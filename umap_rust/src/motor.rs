//! El clasificador en línea.
//!
//! Por lote hace lo mismo que `procesa_lote` en `main.rs` de umap-mcqueen:
//! normaliza la entrada con la referencia de entrenamiento, proyecta con la red,
//! normaliza el encaje con la referencia fija y alimenta DenStream (con los
//! mismos parámetros por defecto). Lo que añade es lo que el Peeler necesita y
//! `main.rs` no hace: **una etiqueta estable por forma de onda**.

use std::collections::HashMap;
use std::error::Error;

use clump::DenStream;
use fast_umap::FittedUmap;

use crate::identidades::Identidades;
use crate::modelo;
use crate::normaliza::{Normalizador, NormalizadorDeEntrada};
use crate::{configuracion_de, lee_manifiesto, MyAutodiffBackend};

/// `clump` marca el ruido con `usize::MAX` (su constante `NOISE`).
const RUIDO: usize = usize::MAX;
/// Etiqueta que sale al stream para una forma de onda sin unidad. Coincide con
/// `LABEL_TRASH` de tridesclous, así el resto del pipeline la trata igual.
pub const ETIQUETA_RUIDO: i64 = -1;

#[derive(Debug, Clone)]
pub struct ParamsDenStream {
    /// Radio de micro-cluster. OJO: `clump` compara la distancia **al cuadrado**
    /// contra este valor al decidir si absorbe, y el radio sin elevar al cuadrado.
    pub epsilon: f32,
    pub min_pts: usize,
    pub beta: f64,
    pub lambda: f64,
    pub mu: f64,
    /// `None` deja el valor de `clump` (2 × epsilon).
    pub macro_epsilon: Option<f32>,
    /// `None` deja el valor de `clump` (100 actualizaciones).
    pub poda_cada: Option<usize>,
    /// Un micro-cluster que DBSCAN deja como ruido (sin vecinos suficientes) se
    /// trata como unidad propia. Sin esto, una neurona compacta que cabe entera
    /// en un solo micro-cluster sale **toda** como ruido.
    pub micro_aislado_es_unidad: bool,
}

impl Default for ParamsDenStream {
    /// Los valores que ya usabas en `main.rs`.
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            min_pts: 3,
            beta: 0.5,
            lambda: 0.01,
            mu: 1.0,
            macro_epsilon: None,
            poda_cada: None,
            micro_aislado_es_unidad: true,
        }
    }
}

impl ParamsDenStream {
    /// Avisos sobre combinaciones que desactivan parte del algoritmo.
    pub fn advertencias(&self) -> Vec<String> {
        let mut v = Vec::new();
        // Un micro-cluster nuevo nace con peso 1.0. Si beta·mu <= 1, nace ya como
        // «potencial» y la ruta de outliers no se usa nunca: no hay filtro de ruido.
        if self.beta * self.mu <= 1.0 {
            v.push(format!(
                "beta·mu = {:.2} <= 1: cada punto nuevo nace como micro-cluster potencial, \
                 así que DenStream NO separa ruido. Sube mu (p. ej. mu=10 con beta=0.5 \
                 exige ~5 puntos cercanos antes de crear una unidad).",
                self.beta * self.mu
            ));
        }
        v
    }

    fn construye(&self) -> DenStream {
        let mut ds = DenStream::new(self.epsilon, self.min_pts)
            .with_beta(self.beta)
            .with_lambda(self.lambda)
            .with_mu(self.mu);
        if let Some(e) = self.macro_epsilon {
            ds = ds.with_macro_epsilon(e);
        }
        if let Some(t) = self.poda_cada {
            ds = ds.with_pruning_period(t);
        }
        ds
    }
}

#[derive(Debug, Clone, Default)]
pub struct Resultado {
    /// Una por forma de onda, en el mismo orden. `ETIQUETA_RUIDO` si no tiene unidad.
    pub etiquetas: Vec<i64>,
    /// Coordenadas normalizadas del encaje, por si la interfaz quiere pintarlas.
    pub encaje: Vec<Vec<f32>>,
    /// Fracción del lote fuera de [0,1]: si crece, el modelo ya no representa lo que llega.
    pub fuera_de_rango: f64,
    pub n_micro: usize,
    pub n_unidades: usize,
}

pub struct Motor {
    fitted: FittedUmap<MyAutodiffBackend>,
    norm: Normalizador,
    norm_entrada: NormalizadorDeEntrada,
    ds: DenStream,
    params: ParamsDenStream,
    identidades: Identidades,
    pub n_dim: usize,
    /// Resumen legible del modelo cargado, para los registros.
    pub info: String,
}

impl Motor {
    /// Carga un modelo guardado con `--modelo`. `n_dim` es la dimensión de las
    /// formas de onda que se le van a dar; `carga_verificada` la compara con la
    /// del entrenamiento, así que un extractor mal configurado falla aquí y no
    /// en mitad de la sesión.
    pub fn carga(
        base: &str,
        n_dim: usize,
        params: ParamsDenStream,
        tolerancia: f32,
        olvido_lotes: u64,
    ) -> Result<Self, Box<dyn Error>> {
        // La configuración sale del propio manifiesto: no hay una segunda copia
        // de `configuracion()` que pueda divergir de la de umap-mcqueen.
        let config = configuracion_de(&lee_manifiesto(base)?)?;
        let (fitted, manifiesto) =
            modelo::carga_verificada::<MyAutodiffBackend>(base, config, n_dim, Default::default())?;
        let norm_entrada = manifiesto
            .normalizacion_de_entrada
            .clone()
            .ok_or("el manifiesto no trae referencia de normalización de ENTRADA")?;
        let norm = manifiesto
            .normalizacion
            .ok_or("el manifiesto no trae referencia de normalización del encaje")?;
        for a in params.advertencias() {
            eprintln!("[tdc_umap] Aviso: {a}");
        }
        let info = format!(
            "{base}: entrada {n_dim}, capas {:?}, salida {}, entrenado con {} formas (huella {}…)",
            manifiesto.hidden_sizes,
            manifiesto.n_components,
            manifiesto.n_entrenamiento,
            &manifiesto.huella_datos[..manifiesto.huella_datos.len().min(12)],
        );
        Ok(Self {
            fitted,
            norm,
            norm_entrada,
            ds: params.construye(),
            params,
            identidades: Identidades::new(tolerancia, olvido_lotes),
            n_dim,
            info,
        })
    }

    pub fn clasifica(&mut self, lote: Vec<Vec<f64>>) -> Result<Resultado, String> {
        if lote.is_empty() {
            return Ok(Resultado::default());
        }
        if let Some(f) = lote.iter().find(|f| f.len() != self.n_dim) {
            return Err(format!("forma de onda de dimensión {} y el modelo espera {}", f.len(), self.n_dim));
        }

        // `fast-umap` NO normaliza en `transform` aunque sí al entrenar
        // (ver normaliza::NormalizadorDeEntrada).
        let entrada = self.norm_entrada.aplica(lote)?;
        let encaje = self.norm.aplica(self.fitted.transform(entrada));
        let fuera_de_rango = self.norm.fuera_de_rango(&encaje);

        let puntos: Vec<Vec<f32>> = encaje
            .iter()
            .map(|fila| fila.iter().map(|&x| x as f32).collect())
            .collect();

        // DenStream rechaza NaN/inf con error y tiraría el lote entero; esos
        // puntos se apartan y salen como ruido.
        let validos: Vec<usize> = (0..puntos.len())
            .filter(|&i| puntos[i].iter().all(|x| x.is_finite()))
            .collect();
        let mut etiquetas = vec![ETIQUETA_RUIDO; puntos.len()];
        if validos.is_empty() {
            return Ok(Resultado { etiquetas, encaje: puntos, fuera_de_rango, ..Default::default() });
        }
        let puntos_validos: Vec<Vec<f32>> = validos.iter().map(|&i| puntos[i].clone()).collect();

        // 1) Aprender. Los índices que devuelve `update_batch` NO sirven para
        //    etiquetar: si la poda salta a mitad del lote, `retain` recorre el
        //    `Vec` y los índices de los primeros puntos ya apuntan a otro sitio.
        self.ds.update_batch(&puntos_validos).map_err(|e| e.to_string())?;

        // 2) Etiquetar contra el estado FINAL, donde todos los índices son coherentes.
        let micro: Vec<usize> = if self.ds.n_clusters() == 0 {
            vec![RUIDO; puntos_validos.len()]
        } else {
            self.ds.predict_batch(&puntos_validos).map_err(|e| e.to_string())?
        };

        // 3) Micro → grupo. DBSCAN sobre centroides da los macro-clusters.
        let macro_de_micro = self.ds.macro_cluster().unwrap_or_default();
        let centroides = self.ds.centroids();
        let cuentas = self.ds.counts();

        // Clave de grupo: k >= 0 para un macro-cluster de DBSCAN; -(j+2) para un
        // micro-cluster aislado que se promueve a unidad propia.
        let aislado_es_unidad = self.params.micro_aislado_es_unidad;
        let clave_de_micro = |j: usize| -> Option<i64> {
            match macro_de_micro.get(j) {
                Some(&k) if k != RUIDO => Some(k as i64),
                Some(_) if aislado_es_unidad => Some(-(j as i64) - 2),
                _ => None,
            }
        };

        // Centroide de cada grupo, ponderado por el número de puntos de cada micro.
        let mut sumas: HashMap<i64, (Vec<f64>, f64)> = HashMap::new();
        for (j, c) in centroides.iter().enumerate() {
            if let Some(clave) = clave_de_micro(j) {
                let w = cuentas.get(j).copied().unwrap_or(1).max(1) as f64;
                let e = sumas.entry(clave).or_insert_with(|| (vec![0.0; c.len()], 0.0));
                for (s, &x) in e.0.iter_mut().zip(c) {
                    *s += w * x as f64;
                }
                e.1 += w;
            }
        }
        let mut claves: Vec<i64> = sumas.keys().copied().collect();
        claves.sort_unstable(); // orden determinista; el id no depende de él
        let centroides_grupo: Vec<Vec<f32>> = claves
            .iter()
            .map(|k| {
                let (s, w) = &sumas[k];
                s.iter().map(|x| (x / w) as f32).collect()
            })
            .collect();

        // 4) Grupo → id estable entre lotes.
        let ids = self.identidades.asigna(&centroides_grupo);
        let id_de_clave: HashMap<i64, i64> = claves.into_iter().zip(ids).collect();

        for (pos, &i) in validos.iter().enumerate() {
            let m = micro[pos];
            if m == RUIDO {
                continue;
            }
            if let Some(id) = clave_de_micro(m).and_then(|k| id_de_clave.get(&k)) {
                etiquetas[i] = *id;
            }
        }

        Ok(Resultado {
            etiquetas,
            encaje: puntos,
            fuera_de_rango,
            n_micro: self.ds.n_clusters(),
            n_unidades: self.identidades.n_unidades(),
        })
    }
}
