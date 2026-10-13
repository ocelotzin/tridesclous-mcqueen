"""Pruebas del motor ``engine='umap'`` sin necesidad del módulo de Rust.

Se sustituye ``tdc_umap.Clasificador`` por un clasificador falso que guarda lo
que recibe. Lo que se prueba es la tubería de tridesclous: que el motor detecta
donde están las espigas, que recorta EXACTAMENTE las mismas formas de onda que
el exportador de entrenamiento, que los modos síncrono y asíncrono coinciden y
que la ruta en línea (preprocesado en flujo) da lo mismo que la fuera de línea.

    pytest tests/test_peeler_engine_umap.py -v
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

import tridesclous

if getattr(tridesclous, '__file__', None) is None:
    # `tridesclous` resolvió a una carpeta sin __init__.py (paquete de espacio
    # de nombres), no al paquete instalado. En este fork pasa con la carpeta
    # `tridesclous/` de la raíz cuando el paquete no está instalado en el
    # intérprete que corre pytest, o cuando se instaló con `pip install -e .`
    # (el `package_dir={'tridesclous': '.'}` de setup.py no funciona en modo
    # editable).
    pytest.exit(
        "`import tridesclous` encontró " + str(list(tridesclous.__path__)) + ", no el paquete.\n"
        "Instala el fork (no editable) en el entorno con el que corres pytest:\n"
        "    pip install .\n"
        "y repite `pip install .` tras cada cambio.", returncode=4)

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.peeler import Peeler
from tridesclous.umap_tools import (aplana_formas, catalogo_sin_plantillas,
                                    exporta_formas_para_umap, ruta_descripcion)

FS = 30000.
NCH = 4
DUR_S = 30.


class ClasificadorFalso:
    """Etiqueta = canal de mayor energía. Guarda todo lo que recibe."""

    def __init__(self, nb_channel):
        self.nb_channel = nb_channel
        self.recibidas = []

    def clasifica(self, formas):
        formas = np.asarray(formas, dtype=np.float64)
        self.recibidas.append(formas.copy())
        n = formas.shape[0]
        energia = (formas.reshape(n, self.nb_channel, -1) ** 2).sum(axis=2)
        etiquetas = energia.argmax(axis=1).astype('int64')
        return etiquetas, np.zeros((n, 2), 'float32'), 0.0, 3, 3


def _senal_sintetica(semilla=0):
    rng = np.random.default_rng(semilla)
    n = int(FS * DUR_S)
    x = rng.normal(0, 10, (n, NCH)).astype('float32')
    t = np.arange(-20, 40)
    forma = -np.exp(-(t / 5.) ** 2) + 0.35 * np.exp(-((t - 12) / 8.) ** 2)
    huellas = np.array([[150, 60, 15, 5], [20, 160, 70, 10], [5, 25, 90, 170]], 'float32')
    verdad = []
    for u in range(3):
        tiempos = np.cumsum(rng.exponential(FS / 6., size=int(DUR_S * 10)))
        tiempos = tiempos[(tiempos > 200) & (tiempos < n - 200)].astype(int)
        for ti in tiempos:
            x[ti - 20:ti + 40] += np.outer(forma, huellas[u]).astype('float32')
            verdad.append((ti, u))
    verdad = np.array(sorted(verdad))
    return x, verdad


@pytest.fixture(scope='module')
def entorno():
    d = tempfile.mkdtemp()
    x, verdad = _senal_sintetica()
    raw = os.path.join(d, 'senal.raw')
    x.tofile(raw)
    dataio = DataIO(dirname=os.path.join(d, 'tdc'))
    dataio.set_data_source(type='RawData', filenames=[raw], dtype='float32',
                           total_channel=NCH, sample_rate=FS)
    dataio.add_one_channel_group(channels=list(range(NCH)), chan_grp=0)

    cc = CatalogueConstructor(dataio=dataio, chan_grp=0)
    cc.set_global_params(chunksize=1024, memory_mode='memmap', internal_dtype='float32')
    cc.set_preprocessor_params(engine='numpy', highpass_freq=300., lowpass_freq=5000.)
    cc.estimate_signals_noise(seg_num=0, duration=10.)
    cc.set_peak_detector_params(method='global', engine='numpy', peak_sign='-',
                                relative_threshold=5., peak_span_ms=0.5)
    cc.run_signalprocessor(duration=DUR_S, detect_peak=True)
    cc.set_waveform_extractor_params(wf_left_ms=-1.0, wf_right_ms=1.5)

    base = os.path.join(d, 'modelos', 's1')
    exporta_formas_para_umap(cc, base)
    catalogo = catalogo_sin_plantillas(cc)
    yield dict(dir=d, x=x, verdad=verdad, cc=cc, dataio=dataio, base=base, catalogo=catalogo)
    shutil.rmtree(d)


def _peeler_offline(e, **umap):
    clf = ClasificadorFalso(NCH)
    p = Peeler(e['dataio'])
    p.change_params(engine='umap', catalogue=e['catalogo'], chunksize=1024,
                    umap_modelo=e['base'], umap_clasificador=clf, **umap)
    p.run(progressbar=False)
    return p, clf, e['dataio'].get_spikes(seg_num=0, chan_grp=0)


def _peeler_online(e, **umap):
    """Igual que OnlinePeeler: dataio=None, preprocesado en flujo, pos = fin del chunk."""
    clf = ClasificadorFalso(NCH)
    p = Peeler(dataio=None)
    p.change_params(engine='umap', catalogue=e['catalogo'], chunksize=1024,
                    umap_modelo=e['base'], umap_clasificador=clf, **umap)
    p.initialize_online_loop(sample_rate=FS, nb_channel=NCH, source_dtype='float32',
                             geometry=e['dataio'].get_geometry(0))
    trozos = []
    x = e['x']
    for a in range(0, x.shape[0] - 1024 + 1, 1024):
        _, _, _, sp = p.process_one_chunk(a + 1024, x[a:a + 1024])
        if sp.size:
            trozos.append(sp)
    resto = p.peeler_engine.get_remaining_spikes()
    if resto is not None:
        trozos.append(resto)
    return p, clf, np.concatenate(trozos) if trozos else np.zeros(0)


def test_registrado():
    from tridesclous.peeler import peeler_engines
    assert 'umap' in peeler_engines


def test_exportacion(entorno):
    cc, base = entorno['cc'], entorno['base']
    filas = np.loadtxt(base + '.csv', delimiter=',')
    ext = cc.info['extract_waveforms']
    assert filas.shape == (cc.nb_peak, NCH * (ext['n_right'] - ext['n_left']))
    with open(ruta_descripcion(base)) as f:
        desc = json.load(f)
    assert desc['n_dim'] == filas.shape[1] and desc['nb_channel'] == NCH


def test_offline_detecta_las_espigas_reales(entorno):
    _, _, espigas = _peeler_offline(entorno, umap_modo='sincrono')
    verdad = entorno['verdad'][:, 0]
    d = np.abs(verdad[:, None] - espigas['index'][None, :]).min(axis=1)
    recall = np.mean(d <= 3)
    falsas = np.mean(np.abs(espigas['index'][:, None] - verdad[None, :]).min(axis=1) > 3)
    print(f"recall {recall:.3f}, falsas {falsas:.3f}, {espigas.size} espigas")
    assert recall > 0.95 and falsas < 0.02


def test_misma_forma_de_onda_que_el_entrenamiento(entorno):
    """Lo que ve la red en línea es exactamente lo que vio al entrenar."""
    cc = entorno['cc']
    _, clf, espigas = _peeler_online(entorno, umap_modo='sincrono')
    recibidas = np.concatenate(clf.recibidas)
    # Sin aliens y con el clasificador falso (etiquetas >= 0), cada espiga emitida
    # corresponde, en orden, a una forma de onda recibida.
    assert recibidas.shape[0] == espigas.size
    picos = cc.all_peaks['index']
    comunes, i_motor, i_cc = np.intersect1d(espigas['index'], picos, return_indices=True)
    print(f"{comunes.size} picos comunes de {picos.size} del constructor")
    assert comunes.size > 0.9 * picos.size
    formas_cc = aplana_formas(cc.get_some_waveforms(peaks_index=i_cc))
    formas_motor = recibidas[i_motor]
    # Preprocesado en flujo (en línea) contra la señal procesada que guardó el constructor.
    err = np.abs(formas_motor - formas_cc).max()
    print(f"error máximo entre extracción en línea y de entrenamiento: {err:.2e}")
    assert np.allclose(formas_motor, formas_cc, atol=1e-3, rtol=1e-4)


def test_asincrono_igual_a_sincrono(entorno):
    _, _, s = _peeler_online(entorno, umap_modo='sincrono')
    _, _, a = _peeler_online(entorno, umap_modo='asincrono', umap_lote=50)
    assert np.array_equal(s['index'], a['index'])
    assert np.array_equal(s['cluster_label'], a['cluster_label'])


def test_campo_canal(entorno):
    _, _, sp = _peeler_online(entorno, umap_modo='sincrono', umap_campo_canal=True)
    assert 'channel' in sp.dtype.names
    assert set(np.unique(sp['channel'])) <= set(range(NCH))


def test_rechaza_otra_ventana(entorno):
    cat = dict(entorno['catalogo'])
    cat['n_right'] = cat['n_right'] + 5
    p = Peeler(dataio=None)
    p.change_params(engine='umap', catalogue=cat, chunksize=1024,
                    umap_modelo=entorno['base'], umap_clasificador=ClasificadorFalso(NCH))
    with pytest.raises(ValueError, match='otra extracción'):
        p.initialize_online_loop(sample_rate=FS, nb_channel=NCH, source_dtype='float32',
                                 geometry=entorno['dataio'].get_geometry(0))
