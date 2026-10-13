"""
Motor ``'umap'`` del Peeler: clasificación **sin plantillas** con UMAP paramétrico
+ DenStream (`umap-mcqueen <https://github.com/ocelotzin/umap-mcqueen>`_).

Se elige como cualquier otro motor::

    peeler.change_params(catalogue=catalogo, engine='umap', umap_modelo='modelos/s1')
    OnlinePeeler.configure(..., engine='umap', umap_modelo='modelos/s1')

Qué conserva de tridesclous y qué cambia
-----------------------------------------

Conserva, del catálogo, la **cadena de preprocesado** (filtro + normalización por
mediana/MAD), el **detector de picos** (con su geometría) y la **ventana**
``n_left..n_right``. Por eso el catálogo sigue siendo obligatorio, aunque no
necesita plantillas: vale uno normal o el de :func:`umap_tools.catalogo_sin_plantillas`.

Cambia la clasificación. En vez de buscar la plantilla que mejor encaja y
restarla (peeling), cada pico se recorta, se proyecta con la red de UMAP y se
asigna con DenStream. Consecuencias:

* **Unidades nuevas durante la sesión.** Una neurona que no estaba en el
  entrenamiento recibe un id nuevo en cuanto DenStream la ve densa. Si su forma
  de onda cae, una vez proyectada, sobre una unidad conocida, se fusiona con ella:
  el descubrimiento depende de que la red la separe.
* **Sin peeling.** Dos espigas superpuestas dan una sola forma de onda mezclada;
  el tiempo muerto descarta la segunda detección.
* **Etiquetas de sesión.** Los ids son estables dentro de la sesión (capa de
  identidades en Rust), no entre sesiones, y no corresponden a ningún catálogo.
* ``jitter`` sale siempre ``nan``: no hay plantilla respecto a la que medirlo.

Modos
-----

``umap_modo='asincrono'`` (por defecto): el hilo del peeler sólo detecta, recorta
y encola; un hilo aparte llama a Rust por lotes. ``process_one_chunk`` devuelve
las espigas ya clasificadas, que pueden ser de chunks anteriores (igual que el
motor geométrico devuelve las de borde del chunk previo). Mantiene el presupuesto
de tiempo real aunque la GPU tarde.

``umap_modo='sincrono'``: clasifica dentro de ``process_one_chunk``. Determinista;
es el que conviene fuera de línea (``Peeler.run``) y en pruebas.
"""

import queue
import threading
import time
from collections import deque

import numpy as np

from .peakdetector import get_peak_detector_class
from .peeler_tools import LABEL_ALIEN, _dtype_spike
from .signalpreprocessor import signalpreprocessor_engines
from .umap_tools import aplana_formas, carga_descripcion, ruta_descripcion

# Con ``umap_campo_canal=True`` cada espiga lleva también el canal (índice
# dentro del grupo) de mayor amplitud. Fuera de línea se deja en False: el
# ``DataIO`` guarda las espigas con ``_dtype_spike`` exacto.
_dtype_spike_con_canal = list(_dtype_spike) + [('channel', 'int64')]

_CLAVES_NECESARIAS = ('signal_preprocessor_params', 'signals_medians', 'signals_mads',
                      'peak_detector_params', 'n_left', 'n_right')

_FIN = object()


class _Vaciar:
    """Petición al hilo de trabajo: clasifica lo pendiente y avisa."""

    def __init__(self):
        self.hecho = threading.Event()


def importa_motor_rust():
    """El módulo compilado de ``umap_rust/``. Error claro si no está instalado."""
    try:
        import tdc_umap
    except ImportError as exc:
        raise ImportError(
            "engine='umap' necesita el módulo compilado `tdc_umap`. Desde el fork:\n"
            "    git submodule update --init umap_rust/umap-mcqueen\n"
            "    cd umap_rust && pip install maturin && maturin develop --release\n"
            f"(error original: {exc})") from exc
    return tdc_umap


class _Trabajador:
    """Hilo que agrupa formas de onda y llama al clasificador de Rust.

    Sólo guarda referencias a la cola, al clasificador y a sus propios
    resultados, no al motor: así el motor puede recolectarse aunque el hilo viva.
    """

    def __init__(self, clasificador, lote, vaciar_cada_s, tam_cola):
        self.clasificador = clasificador
        self.lote = int(lote)
        self.vaciar_cada_s = float(vaciar_cada_s)
        self.cola = queue.Queue(maxsize=int(tam_cola))
        self.resultados = deque()
        self.estadisticas = dict(clasificadas=0, descartadas=0, errores=0, ms=deque(maxlen=200),
                                 fuera=deque(maxlen=200), n_micro=0, n_unidades=0)
        self._hilo = threading.Thread(target=self._bucle, name='tdc-umap', daemon=True)
        self._hilo.start()

    def envia(self, indices, canales, formas):
        """No bloquea nunca: si la cola está llena, se descarta y se cuenta."""
        try:
            self.cola.put_nowait((indices, canales, formas))
        except queue.Full:
            self.estadisticas['descartadas'] += len(indices)

    def recoge(self):
        salida = []
        while self.resultados:
            salida.append(self.resultados.popleft())
        return salida

    def vacia(self, timeout=30.0):
        """Clasifica todo lo pendiente y espera a que termine."""
        peticion = _Vaciar()
        try:
            self.cola.put(peticion, timeout=timeout)
        except queue.Full:
            return False
        return peticion.hecho.wait(timeout)

    def para(self, timeout=5.0):
        if self._hilo.is_alive():
            try:
                self.cola.put(_FIN, timeout=timeout)
            except queue.Full:
                pass
            self._hilo.join(timeout)

    @property
    def en_cola(self):
        return self.cola.qsize()

    def _bucle(self):
        pend, n_pend, t_primero = [], 0, None
        while True:
            try:
                item = self.cola.get(timeout=self.vaciar_cada_s if pend else 0.5)
            except queue.Empty:
                item = None
            if item is _FIN:
                self._clasifica(pend)
                return
            if isinstance(item, _Vaciar):
                self._clasifica(pend)
                pend, n_pend = [], 0
                item.hecho.set()
                continue
            if item is not None:
                if not pend:
                    t_primero = time.monotonic()
                pend.append(item)
                n_pend += len(item[0])
            if pend and (n_pend >= self.lote or time.monotonic() - t_primero >= self.vaciar_cada_s):
                self._clasifica(pend)
                pend, n_pend = [], 0

    def _clasifica(self, pend):
        if not pend:
            return
        indices = np.concatenate([p[0] for p in pend])
        try:
            canales = np.concatenate([p[1] for p in pend])
            formas = np.concatenate([p[2] for p in pend])
            t0 = time.perf_counter()
            etiquetas, _encaje, fuera, n_micro, n_uni = self.clasificador.clasifica(formas)
            e = self.estadisticas
            e['ms'].append((time.perf_counter() - t0) * 1000.0)
            e['fuera'].append(float(fuera))
            e['n_micro'], e['n_unidades'] = int(n_micro), int(n_uni)
            e['clasificadas'] += indices.size
            self.resultados.append((indices, np.asarray(etiquetas, dtype='int64'), canales))
        except Exception as exc:  # el hilo no puede morir en silencio
            self.estadisticas['errores'] += 1
            if self.estadisticas['errores'] <= 5:
                print(f"[PeelerEngineUmap] Error clasificando {indices.size} formas: {exc!r}")


class PeelerEngineUmap:
    """Motor del Peeler que clasifica con umap-mcqueen en vez de con plantillas.

    Implementa la misma interfaz que ``PeelerEngineGeometrical`` (``change_params``,
    ``initialize``, ``initialize_before_each_segment``, ``process_one_chunk``,
    ``get_remaining_spikes``, ``total_spike``, ``signalpreprocessor``), que es lo
    que usan ``Peeler`` y ``OnlinePeeler``.
    """

    def change_params(self, catalogue=None, chunksize=1024, internal_dtype='float32',
                      save_bad_label=False,
                      umap_modelo=None,
                      umap_denstream=None,
                      umap_modo='asincrono',
                      umap_lote=64,
                      umap_vaciar_cada_s=0.05,
                      umap_tam_cola=20000,
                      umap_tiempo_muerto_ms=0.5,
                      umap_campo_canal=False,
                      umap_clasificador=None,
                      **ignorados):
        """
        Parameters
        ----------
        catalogue: dict
            Cualquier catálogo de tridesclous, o el de ``catalogo_sin_plantillas``.
            Sólo se usan preprocesado, detección, ruido y ventana.
        umap_modelo: str
            Ruta base de ``umap-mcqueen --modelo`` (``<base>.bin`` + ``<base>.json``).
        umap_denstream: dict
            Para ``tdc_umap.Clasificador``: epsilon, min_pts, beta, lambda_, mu,
            macro_epsilon, poda_cada, micro_aislado_es_unidad, tolerancia, olvido_lotes.
        umap_modo: 'asincrono' | 'sincrono'
        umap_lote, umap_vaciar_cada_s, umap_tam_cola:
            Sólo modo asíncrono: formas por llamada a Rust, latencia máxima y
            formas en espera antes de descartar.
        umap_tiempo_muerto_ms: float
            Dos picos más cercanos que esto (en cualquier canal) son la misma espiga.
        umap_campo_canal: bool
            Añade el campo ``channel`` a cada espiga (sólo en línea).
        umap_clasificador:
            Objeto con ``clasifica(formas)`` ya construido; para pruebas o para
            reutilizar un modelo cargado.
        """
        if catalogue is None:
            raise ValueError("engine='umap' necesita un catálogo (preprocesado y detección)")
        faltan = [k for k in _CLAVES_NECESARIAS if k not in catalogue]
        if faltan:
            raise ValueError(f"al catálogo le faltan {faltan} para engine='umap'")
        if umap_modelo is None and umap_clasificador is None:
            raise ValueError("engine='umap' necesita umap_modelo (ruta base del modelo)")
        if umap_modo not in ('asincrono', 'sincrono'):
            raise ValueError("umap_modo debe ser 'asincrono' o 'sincrono'")
        if ignorados:
            print(f"[PeelerEngineUmap] Parámetros del motor de plantillas ignorados: "
                  f"{sorted(ignorados)}")

        self.catalogue = catalogue
        self.chunksize = int(chunksize)
        self.internal_dtype = internal_dtype
        self.save_bad_label = bool(save_bad_label)
        self.umap_modelo = umap_modelo
        self.umap_denstream = dict(umap_denstream or {})
        self.umap_modo = umap_modo
        self.umap_lote = umap_lote
        self.umap_vaciar_cada_s = umap_vaciar_cada_s
        self.umap_tam_cola = umap_tam_cola
        self.umap_tiempo_muerto_ms = float(umap_tiempo_muerto_ms)
        self.campo_canal = bool(umap_campo_canal)
        self.clasificador = umap_clasificador
        self._trabajador = None

        self.n_left = int(catalogue['n_left'])
        self.n_right = int(catalogue['n_right'])
        self.peak_width = self.n_right - self.n_left

    # ------------------------------------------------------------ initialize

    def initialize(self, sample_rate=None, nb_channel=None, source_dtype=None, geometry=None,
                   already_processed=False, processor_engine=None):
        self.sample_rate = sample_rate
        self.nb_channel = nb_channel
        self.source_dtype = source_dtype
        self.geometry = geometry
        self.already_processed = already_processed

        # Preprocesado: idéntico al del motor geométrico (peeler_engine_base).
        if not already_processed:
            p = dict(self.catalogue['signal_preprocessor_params'])
            motor = p.pop('engine')
            if processor_engine is not None:
                motor = processor_engine
            clase = signalpreprocessor_engines[motor]
            self.signalpreprocessor = clase(sample_rate, nb_channel, self.chunksize, source_dtype)
            p['normalize'] = True
            p['signals_medians'] = self.catalogue['signals_medians']
            p['signals_mads'] = self.catalogue['signals_mads']
            self.signalpreprocessor.change_params(**p)
            self.internal_dtype = self.signalpreprocessor.output_dtype
            assert self.chunksize > self.signalpreprocessor.pad_width, \
                f"pad_width ({self.signalpreprocessor.pad_width}) >= chunksize ({self.chunksize})"
        else:
            self.signalpreprocessor = None
            self.internal_dtype = source_dtype

        # Detector: el mismo que el motor geométrico.
        pdp = dict(self.catalogue['peak_detector_params'])
        self.peak_sign = pdp.get('peak_sign', '-')
        self.n_span = max(1, int(sample_rate * pdp['peak_span_ms'] / 1000.) // 2)

        # La fifo guarda el chunk nuevo más lo justo para que cualquier pico de la
        # zona que toca evaluar tenga su ventana completa y su vecindad de detección.
        self._margen_izq = max(-self.n_left, self.n_span) + 1
        self._margen_der = max(self.n_right, self.n_span)
        self.extra_size = self._margen_izq + self._margen_der
        self.fifo_size = self.chunksize + self.extra_size

        motor_det = pdp.pop('engine')
        metodo = pdp.pop('method')
        clase_det = get_peak_detector_class(metodo, motor_det)
        self.peakdetector = clase_det(sample_rate, nb_channel, self.fifo_size - 2 * self.n_span,
                                      self.internal_dtype, geometry)
        self.peakdetector.change_params(**pdp)

        alien = self.catalogue.get('clean_peaks_params', {}).get('alien_value_threshold')
        if alien is not None and (alien < 0. or np.isnan(alien)):
            alien = None
        self.alien_value_threshold = alien

        self.tiempo_muerto = max(1, int(round(self.umap_tiempo_muerto_ms * sample_rate / 1000.)))
        self.n_dim = nb_channel * self.peak_width
        self._comprueba_descripcion()

        if self.clasificador is None:
            tdc_umap = importa_motor_rust()
            self.clasificador = tdc_umap.Clasificador(self.umap_modelo, self.n_dim,
                                                      **self.umap_denstream)
        if self.umap_modo == 'asincrono' and self._trabajador is None:
            self._trabajador = _Trabajador(self.clasificador, self.umap_lote,
                                           self.umap_vaciar_cada_s, self.umap_tam_cola)

        self.estadisticas = dict(detectadas=0, aliens=0, emitidas=0, ruido=0,
                                 ms_sincrono=deque(maxlen=200), fuera=deque(maxlen=200),
                                 n_micro=0, n_unidades=0)
        self.initialize_before_each_segment(already_processed=already_processed)

    def _comprueba_descripcion(self):
        """Compara el catálogo con lo que se usó para entrenar (``<base>.tdc.json``)."""
        if self.umap_modelo is None:
            return
        desc = carga_descripcion(self.umap_modelo)
        if desc is None:
            print(f"[PeelerEngineUmap] ⚠ No hay {ruta_descripcion(self.umap_modelo)}: no se "
                  f"puede comprobar que el modelo se entrenó con esta ventana y este ruido. "
                  f"Exporta el entrenamiento con umap_tools.exporta_formas_para_umap.")
            return
        malas = []
        for k, v in (('n_left', self.n_left), ('n_right', self.n_right),
                     ('nb_channel', self.nb_channel)):
            if desc.get(k) != v:
                malas.append(f"{k}: entrenamiento {desc.get(k)}, ahora {v}")
        if malas:
            raise ValueError("El modelo UMAP se entrenó con otra extracción:\n  - "
                             + "\n  - ".join(malas))
        mads_ent = np.asarray(desc.get('signals_mads', []), dtype=float)
        mads = np.asarray(self.catalogue['signals_mads'], dtype=float).ravel()
        if mads_ent.shape == mads.shape and mads.size:
            cambio = np.abs(mads / np.where(mads_ent > 0, mads_ent, 1.0) - 1.0)
            if np.max(cambio) > 0.2:
                print(f"[PeelerEngineUmap] ⚠ El ruido (MAD) del catálogo difiere hasta un "
                      f"{100 * np.max(cambio):.0f} % del de entrenamiento: las formas de onda "
                      f"normalizadas no tendrán la misma escala que las que vio la red.")

    def initialize_before_each_segment(self, already_processed=False):
        self.total_spike = 0
        self.fifo_residuals = np.zeros((self.fifo_size, self.nb_channel), dtype=self.internal_dtype)
        if self.signalpreprocessor is not None:
            self.signalpreprocessor.initialize_stream()
        self.peakdetector.initialize_stream()
        self.already_processed = already_processed
        self._recientes = np.zeros(0, dtype='int64')  # picos aceptados cerca del borde

    # ---------------------------------------------------------------- chunk

    def _aplica_procesador(self, pos, sigs_chunk):
        if self.already_processed:
            abs_head_index, chunk = pos, sigs_chunk
        else:
            abs_head_index, chunk = self.signalpreprocessor.process_buffer_stream(pos, sigs_chunk)
        n = chunk.shape[0]
        if 0 < n < self.fifo_size:
            self.fifo_residuals[:-n] = self.fifo_residuals[n:]
            self.fifo_residuals[-n:] = chunk
        elif n >= self.fifo_size:
            self.fifo_residuals[:] = chunk[-self.fifo_size:]
        return abs_head_index, chunk

    def _detecta(self, n_nuevo, desplazamiento):
        """Picos a evaluar en este chunk: índices locales en la fifo y canal.

        La zona evaluada mide exactamente lo que avanzó la fifo y termina donde
        un pico aún tiene su ventana completa, así que cada muestra se evalúa
        una sola vez a lo largo del flujo.
        """
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        if mask.ndim == 1:
            s = np.flatnonzero(mask) + self.n_span
            vals = self.fifo_residuals[s]
            if self.peak_sign == '-':
                c = np.argmin(vals, axis=1)
            elif self.peak_sign == '+':
                c = np.argmax(vals, axis=1)
            else:
                c = np.argmax(np.abs(vals), axis=1)
        else:
            s, c = np.nonzero(mask)
            s = s + self.n_span

        hi = self.fifo_size - self._margen_der
        lo = max(hi - n_nuevo, self._margen_izq)
        dentro = (s >= lo) & (s < hi)
        s, c = s[dentro], c[dentro]
        if s.size == 0:
            self._olvida_recientes(hi + desplazamiento)
            return s, c

        # Una espiga puede ser pico en varios canales: se queda la de mayor
        # amplitud y se descartan las que caen dentro del tiempo muerto.
        orden = np.argsort(np.abs(self.fifo_residuals[s, c]))[::-1]
        aceptados = list(self._recientes)
        keep = []
        for k in orden:
            a = s[k] + desplazamiento
            if all(abs(a - b) > self.tiempo_muerto for b in aceptados):
                aceptados.append(a)
                keep.append(k)
        keep = np.sort(np.asarray(keep, dtype='int64'))
        s, c = s[keep], c[keep]
        self._recientes = np.asarray(aceptados, dtype='int64')
        self._olvida_recientes(hi + desplazamiento)
        orden_t = np.argsort(s)
        return s[orden_t], c[orden_t]

    def _olvida_recientes(self, fin_abs):
        self._recientes = self._recientes[self._recientes >= fin_abs - self.tiempo_muerto]

    def _espigas(self, indices, etiquetas, canales):
        mantener = (etiquetas >= 0) | self.save_bad_label
        indices, etiquetas, canales = indices[mantener], etiquetas[mantener], canales[mantener]
        sp = np.zeros(indices.size, dtype=_dtype_spike_con_canal if self.campo_canal else _dtype_spike)
        sp['index'] = indices
        sp['cluster_label'] = etiquetas
        sp['jitter'] = np.nan
        if self.campo_canal:
            sp['channel'] = canales
        return sp

    def _vacias(self):
        return np.zeros(0, dtype=_dtype_spike_con_canal if self.campo_canal else _dtype_spike)

    def process_one_chunk(self, pos, sigs_chunk):
        abs_head_index, preprocessed_chunk = self._aplica_procesador(pos, sigs_chunk)
        desplazamiento = abs_head_index - self.fifo_size  # índice local → absoluto
        s, c = self._detecta(preprocessed_chunk.shape[0], desplazamiento)

        piezas = []
        if s.size:
            self.estadisticas['detectadas'] += s.size
            ventanas = np.stack([self.fifo_residuals[i + self.n_left:i + self.n_right] for i in s])
            indices = (s + desplazamiento).astype('int64')
            canales = c.astype('int64')
            if self.alien_value_threshold is not None:
                alien = np.any(np.abs(ventanas) > self.alien_value_threshold, axis=(1, 2))
                if alien.any():
                    self.estadisticas['aliens'] += int(alien.sum())
                    piezas.append((indices[alien], np.full(alien.sum(), LABEL_ALIEN, 'int64'),
                                   canales[alien]))
                    ventanas, indices, canales = ventanas[~alien], indices[~alien], canales[~alien]
            if indices.size:
                formas = aplana_formas(ventanas)
                if self._trabajador is not None:
                    self._trabajador.envia(indices, canales, formas)
                else:
                    piezas.append((indices, self._clasifica_sincrono(formas), canales))

        if self._trabajador is not None:
            piezas.extend(self._trabajador.recoge())

        espigas = self._junta(piezas)
        return abs_head_index, preprocessed_chunk, self.total_spike, espigas

    def _clasifica_sincrono(self, formas):
        t0 = time.perf_counter()
        etiquetas, _encaje, fuera, n_micro, n_uni = self.clasificador.clasifica(formas)
        e = self.estadisticas
        e['ms_sincrono'].append((time.perf_counter() - t0) * 1000.0)
        e['fuera'].append(float(fuera))
        e['n_micro'], e['n_unidades'] = int(n_micro), int(n_uni)
        return np.asarray(etiquetas, dtype='int64')

    def _junta(self, piezas):
        if not piezas:
            return self._vacias()
        indices = np.concatenate([p[0] for p in piezas])
        etiquetas = np.concatenate([p[1] for p in piezas])
        canales = np.concatenate([p[2] for p in piezas])
        self.estadisticas['ruido'] += int(np.sum(etiquetas == -1))
        espigas = self._espigas(indices, etiquetas, canales)
        espigas = espigas.take(np.argsort(espigas['index'], kind='stable'))
        self.total_spike += espigas.size
        self.estadisticas['emitidas'] += espigas.size
        return espigas

    def get_remaining_spikes(self):
        """Fuera de línea, o al parar: lo que aún estaba en el hilo de trabajo."""
        if self._trabajador is None:
            return None
        self._trabajador.vacia()
        espigas = self._junta(self._trabajador.recoge())
        return espigas if espigas.size else None

    # ---------------------------------------------------------------- estado

    def estado(self):
        """Resumen para los informes del nodo."""
        e = dict(self.estadisticas)
        if self._trabajador is not None:
            t = self._trabajador.estadisticas
            e.update(en_cola=self._trabajador.en_cola, descartadas=t['descartadas'],
                     errores=t['errores'], clasificadas=t['clasificadas'],
                     n_micro=t['n_micro'], n_unidades=t['n_unidades'])
            ms, fuera = t['ms'], t['fuera']
        else:
            e.update(en_cola=0, descartadas=0, errores=0)
            ms, fuera = e['ms_sincrono'], e['fuera']
        e['ms_por_lote'] = float(np.mean(ms)) if ms else 0.0
        e['fuera_de_rango'] = float(np.mean(fuera)) if fuera else 0.0
        e.pop('ms_sincrono', None)
        e.pop('fuera', None)
        return e

    def cerrar(self):
        if self._trabajador is not None:
            self._trabajador.para()
            self._trabajador = None

    def __del__(self):
        try:
            self.cerrar()
        except Exception:
            pass
