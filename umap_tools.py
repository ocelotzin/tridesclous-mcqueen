"""
Herramientas para el motor ``'umap'`` del Peeler (UMAP paramétrico + DenStream,
de `umap-mcqueen <https://github.com/ocelotzin/umap-mcqueen>`_).

El motor clasifica formas de onda sin plantillas, pero la red de UMAP sólo sabe
proyectar vectores **construidos igual que los de su entrenamiento**: misma
cadena de preprocesado (filtro + normalización por mediana/MAD), misma ventana
``n_left..n_right``, mismos canales y mismo orden al aplanar. Este módulo
garantiza esa igualdad haciendo que el entrenamiento salga del mismo sitio que
la clasificación:

* :func:`exporta_formas_para_umap` saca las formas de onda de un
  ``CatalogueConstructor`` (las mismas que ve tridesclous) y escribe el CSV que
  lee ``umap-mcqueen --modelo``, más un archivo ``<base>.tdc.json`` con la
  ventana y el ruido. El motor compara ese archivo con el catálogo al arrancar.
* :func:`aplana_formas` es el único aplanado que existe; lo usan el exportador
  y el motor.
* :func:`catalogo_sin_plantillas` construye el catálogo mínimo que necesita el
  motor (preprocesado, detección, ventana), sin clustering.

Uso típico::

    cc = CatalogueConstructor(dataio=DataIO('tdc_entreno'), chan_grp=0)
    # ... set_preprocessor_params, estimate_signals_noise, run_signalprocessor,
    #     set_waveform_extractor_params ...
    exporta_formas_para_umap(cc, 'modelos/sesion1', n_max=20000)
    #   $ umap-mcqueen modelos/sesion1.csv --modelo modelos/sesion1 --train 2000
    catalogo = catalogo_sin_plantillas(cc)
    cc.dataio.save_catalogue(catalogo, name='umap')

y en línea::

    OnlinePeeler.configure(catalogue=catalogo, engine='umap',
                           umap_modelo='modelos/sesion1', ...)
"""

import json
import os

import numpy as np

# Versión del archivo <base>.tdc.json; súbela si cambia su significado.
FORMATO_TDC_JSON = 1


def aplana_formas(formas):
    """(n, ancho, canales) → (n, canales·ancho), canal por canal, en float64.

    Orden: [c0 t0..tN, c1 t0..tN, ...]. Es el que ve un humano al concatenar las
    formas de onda de cada canal, y es el ÚNICO que se usa en el entrenamiento y
    en línea. Cambiarlo invalida todos los modelos ya entrenados.
    """
    formas = np.asarray(formas)
    if formas.ndim != 3:
        raise ValueError(f"se esperaban formas (n, ancho, canales) y llegó {formas.shape}")
    n = formas.shape[0]
    return np.ascontiguousarray(formas.transpose(0, 2, 1), dtype=np.float64).reshape(n, -1)


def descripcion_de_extraccion(n_left, n_right, nb_channel, signals_medians, signals_mads,
                              peak_detector_params=None, preprocessor_params=None):
    """Lo que tiene que coincidir entre el entrenamiento y la sesión en línea."""
    return {
        'formato': FORMATO_TDC_JSON,
        'n_left': int(n_left),
        'n_right': int(n_right),
        'nb_channel': int(nb_channel),
        'n_dim': int(nb_channel * (n_right - n_left)),
        'orden': 'por_canal',
        'signals_medians': [float(x) for x in np.asarray(signals_medians).ravel()],
        'signals_mads': [float(x) for x in np.asarray(signals_mads).ravel()],
        'peak_detector_params': _a_json(peak_detector_params),
        'preprocessor_params': _a_json(preprocessor_params),
    }


def _a_json(d):
    if d is None:
        return None
    return {str(k): (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dict(d).items()}


def ruta_descripcion(base):
    return f"{os.path.splitext(base)[0]}.tdc.json"


def carga_descripcion(base):
    ruta = ruta_descripcion(base)
    if not os.path.exists(ruta):
        return None
    with open(ruta) as f:
        return json.load(f)


def exporta_formas_para_umap(cc, base, n_max=None, semilla=0, lote=5000):
    """Escribe ``<base>.csv`` y ``<base>.tdc.json`` a partir de un CatalogueConstructor.

    Args:
        cc: ``CatalogueConstructor`` con ``run_signalprocessor`` ya hecho (picos
            detectados) y ``set_waveform_extractor_params`` aplicado.
        base: Ruta base del modelo, la misma que se dará a ``--modelo``.
        n_max: Máximo de formas de onda; ``None`` = todos los picos.
        semilla: Para la selección y el barajado.
        lote: Formas de onda leídas por vez (acota la memoria).

    Las filas se **barajan**: ``umap-mcqueen`` entrena con las primeras
    ``--train`` filas, y sin barajar sólo vería el principio de la grabación.

    Returns:
        (ruta_csv, n_formas, n_dim)
    """
    if cc.nb_peak == 0:
        raise ValueError("el CatalogueConstructor no tiene picos: ejecuta run_signalprocessor")
    ext = cc.info.get('extract_waveforms')
    if ext is None:
        raise ValueError("falta set_waveform_extractor_params(...) en el CatalogueConstructor")

    rng = np.random.default_rng(semilla)
    indices = np.arange(cc.nb_peak)
    if n_max is not None and n_max < indices.size:
        indices = rng.choice(indices, size=n_max, replace=False)
    rng.shuffle(indices)

    base = os.path.splitext(base)[0]
    carpeta = os.path.dirname(base)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    ruta_csv = base + '.csv'
    n_dim = None
    with open(ruta_csv, 'w') as f:
        for a in range(0, indices.size, lote):
            trozo = np.sort(indices[a:a + lote])  # dataio lee mejor en orden
            planas = aplana_formas(cc.get_some_waveforms(peaks_index=trozo))
            planas = planas[rng.permutation(planas.shape[0])]
            n_dim = planas.shape[1]
            np.savetxt(f, planas, delimiter=',', fmt='%.7g')

    desc = descripcion_de_extraccion(
        ext['n_left'], ext['n_right'], cc.nb_channel, cc.signals_medians, cc.signals_mads,
        cc.info.get('peak_detector'), cc.info.get('preprocessor'))
    with open(ruta_descripcion(base), 'w') as f:
        json.dump(desc, f, indent=2)
    return ruta_csv, int(indices.size), int(n_dim)


def catalogo_sin_plantillas(cc):
    """El catálogo mínimo para ``engine='umap'``: preprocesado, detección y ventana.

    No necesita clustering ni ``make_catalogue``. Se guarda con
    ``cc.dataio.save_catalogue(catalogo, name='umap')`` y se carga como cualquier
    otro. Lleva ``clusters`` y ``cluster_labels`` vacíos para que el código que
    espera esas claves (p. ej. el mapeo de réplicas NEU-244) no falle.

    Un catálogo normal de tridesclous también sirve: el motor sólo lee estas
    claves e ignora las plantillas.
    """
    ext = cc.info['extract_waveforms']
    return {
        'chan_grp': cc.chan_grp,
        'mode': 'dense',
        'n_left': int(ext['n_left']),
        'n_right': int(ext['n_right']),
        'n_left_long': int(ext.get('n_left_long', ext['n_left'])),
        'n_right_long': int(ext.get('n_right_long', ext['n_right'])),
        'signal_preprocessor_params': dict(cc.info['preprocessor']),
        'peak_detector_params': dict(cc.info['peak_detector']),
        'clean_peaks_params': dict(cc.info['clean_peaks']) if 'clean_peaks' in cc.info else {},
        'signals_medians': np.array(cc.signals_medians, copy=True),
        'signals_mads': np.array(cc.signals_mads, copy=True),
        'clusters': np.zeros(0, dtype=[('cluster_label', 'int64')]),
        'cluster_labels': np.zeros(0, dtype='int64'),
        'sin_plantillas': True,
    }


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Exporta las formas de onda de un directorio de tridesclous para "
                    "entrenar umap-mcqueen, y guarda el catálogo sin plantillas.")
    ap.add_argument('dirname', help='directorio DataIO de tridesclous')
    ap.add_argument('base', help='ruta base del modelo (la de --modelo)')
    ap.add_argument('--chan-grp', type=int, default=0)
    ap.add_argument('--n-max', type=int, default=None)
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--nombre-catalogo', default='umap')
    a = ap.parse_args()

    from .dataio import DataIO
    from .catalogueconstructor import CatalogueConstructor
    cc = CatalogueConstructor(dataio=DataIO(a.dirname), chan_grp=a.chan_grp)
    ruta, n, n_dim = exporta_formas_para_umap(cc, a.base, n_max=a.n_max, semilla=a.semilla)
    cc.dataio.save_catalogue(catalogo_sin_plantillas(cc), name=a.nombre_catalogo)
    print(f"{n} formas de onda de dimensión {n_dim} → {ruta}")
    print(f"Catálogo sin plantillas guardado como '{a.nombre_catalogo}'")
    print(f"Siguiente: umap-mcqueen {ruta} --modelo {os.path.splitext(a.base)[0]} --train <n>")


if __name__ == '__main__':
    _main()
