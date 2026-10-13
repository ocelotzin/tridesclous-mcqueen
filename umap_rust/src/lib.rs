//! `tdc_umap`: el clasificador de umap-mcqueen, importable desde Python para el
//! motor `engine='umap'` del Peeler de tridesclous.
//!
//! ## Cómo se usa umap-mcqueen sin modificarlo
//!
//! umap-mcqueen es un binario (sólo `main.rs`), así que no se puede depender de
//! él como biblioteca. En vez de copiar su código, se compilan **sus mismos
//! archivos** desde el submódulo con `#[path]`:
//!
//! * `normaliza.rs` — las referencias fijas de normalización (entrada y encaje).
//! * `modelo.rs` — `carga_verificada`, que compara el manifiesto con la red.
//!
//! `modelo.rs` usa `crate::normaliza`, y aquí `normaliza` también está en la
//! raíz del crate, así que compila sin tocar una línea.
//!
//! La configuración de UMAP **no se duplica**: se reconstruye del manifiesto que
//! escribió umap-mcqueen al entrenar. Sólo `hidden_sizes`, `n_components` y
//! `input_size` definen la arquitectura (`FittedUmap::load`), y los tres están
//! en el manifiesto. Así es imposible que este crate y `main.rs` discrepen.

#[path = "../umap-mcqueen/src/normaliza.rs"]
pub mod normaliza;

#[path = "../umap-mcqueen/src/modelo.rs"]
pub mod modelo;

pub mod identidades;
pub mod motor;

use std::sync::Mutex;

use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::modelo::{Manifiesto, RutasDelModelo};
use crate::motor::{Motor, ParamsDenStream};

/// El mismo backend que `main.rs` de umap-mcqueen.
pub type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

/// Lee el manifiesto `<base>.json` que escribió umap-mcqueen.
pub fn lee_manifiesto(base: &str) -> Result<Manifiesto, String> {
    let ruta = RutasDelModelo::de(base).manifiesto;
    let texto = std::fs::read_to_string(&ruta)
        .map_err(|e| format!("no se pudo leer el manifiesto {}: {e}", ruta.display()))?;
    serde_json::from_str(&texto).map_err(|e| format!("manifiesto {} ilegible: {e}", ruta.display()))
}

/// Inversa de `modelo::nombre_de_metrica`.
fn metrica_de(nombre: &str) -> Result<Metric, String> {
    Ok(match nombre {
        "Euclidean" => Metric::Euclidean,
        "EuclideanKNN" => Metric::EuclideanKNN,
        "Manhattan" => Metric::Manhattan,
        "Cosine" => Metric::Cosine,
        "Minkowski" => Metric::Minkowski,
        otra => return Err(format!("métrica desconocida en el manifiesto: {otra}")),
    })
}

/// La configuración con la que se entrenó, reconstruida del manifiesto.
pub fn configuracion_de(m: &Manifiesto) -> Result<UmapConfig, String> {
    Ok(UmapConfig {
        n_components: m.n_components,
        hidden_sizes: m.hidden_sizes.clone(),
        graph: GraphParams {
            n_neighbors: m.n_neighbors,
            metric: metrica_de(&m.metrica)?,
            ..Default::default()
        },
        ..Default::default()
    })
}

/// `Mutex` para que la clase sea `Sync` sin exigírselo al backend de GPU, y para
/// poder soltar el GIL mientras corre la red: con el GIL tomado, el hilo del
/// peeler no podría ejecutar Python durante la proyección.
#[pyclass(name = "Clasificador", module = "tdc_umap")]
struct PyClasificador {
    motor: Mutex<Motor>,
    info: String,
}

#[pymethods]
impl PyClasificador {
    #[new]
    #[pyo3(signature = (
        base, n_dim,
        epsilon=0.1, min_pts=3, beta=0.5, lambda_=0.01, mu=1.0,
        macro_epsilon=None, poda_cada=None, micro_aislado_es_unidad=true,
        tolerancia=0.15, olvido_lotes=200,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base: &str,
        n_dim: usize,
        epsilon: f32,
        min_pts: usize,
        beta: f64,
        lambda_: f64,
        mu: f64,
        macro_epsilon: Option<f32>,
        poda_cada: Option<usize>,
        micro_aislado_es_unidad: bool,
        tolerancia: f32,
        olvido_lotes: u64,
    ) -> PyResult<Self> {
        let params = ParamsDenStream {
            epsilon, min_pts, beta, lambda: lambda_, mu,
            macro_epsilon, poda_cada, micro_aislado_es_unidad,
        };
        let motor = Motor::carga(base, n_dim, params, tolerancia, olvido_lotes)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let info = motor.info.clone();
        Ok(Self { motor: Mutex::new(motor), info })
    }

    /// Clasifica un lote `(n, n_dim)` de float64.
    ///
    /// Devuelve `(etiquetas int64 (n,), encaje float32 (n, k), fuera_de_rango,
    /// n_micro, n_unidades)`. Etiqueta -1 = ruido.
    #[allow(clippy::type_complexity)]
    fn clasifica<'py>(
        &self,
        py: Python<'py>,
        formas: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f32>>, f64, usize, usize)> {
        let lote: Vec<Vec<f64>> = formas.as_array().outer_iter().map(|f| f.to_vec()).collect();
        let n = lote.len();
        let r = py
            .allow_threads(|| {
                let mut m = self
                    .motor
                    .lock()
                    .map_err(|_| "el motor quedó envenenado por un pánico anterior".to_string())?;
                m.clasifica(lote)
            })
            .map_err(PyRuntimeError::new_err)?;

        let k = r.encaje.first().map(|f| f.len()).unwrap_or(2);
        let plano: Vec<f32> = r.encaje.into_iter().flatten().collect();
        let encaje = numpy::ndarray::Array2::from_shape_vec((n, k), plano)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            r.etiquetas.into_pyarray(py),
            encaje.into_pyarray(py),
            r.fuera_de_rango,
            r.n_micro,
            r.n_unidades,
        ))
    }

    #[getter]
    fn n_dim(&self) -> PyResult<usize> {
        Ok(self.motor.lock().map_err(|_| PyRuntimeError::new_err("motor envenenado"))?.n_dim)
    }

    fn __repr__(&self) -> String {
        format!("tdc_umap.Clasificador({})", self.info)
    }
}

#[pymodule]
fn tdc_umap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClasificador>()?;
    Ok(())
}
