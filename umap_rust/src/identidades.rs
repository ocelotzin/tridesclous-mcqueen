//! Identidades estables para las unidades que encuentra DenStream.
//!
//! **Por qué hace falta.** `clump::DenStream` no tiene identificadores: lo que
//! devuelven `update`, `predict` y `macro_cluster` es la *posición* del
//! micro-cluster dentro de un `Vec`. Esa posición cambia cuando la poda hace
//! `retain` (todo lo que estaba detrás se recorre) y cuando un outlier se promueve
//! (`push` al final). DBSCAN, además, numera los macro-clusters en el orden en que
//! los encuentra, y ese orden cambia de una llamada a otra.
//!
//! Resultado: sin esta capa, la etiqueta 3 de este lote y la 3 del siguiente
//! pueden ser neuronas distintas, y el logger las guardaría como la misma.
//!
//! **Qué hace.** Cada lote se describe cada unidad por su centroide en el encaje
//! normalizado. Se emparejan con las unidades conocidas por cercanía (voraz, de
//! menor a mayor distancia) dentro de `tolerancia`. Lo emparejado conserva su id;
//! lo que no, recibe uno nuevo. Los ids **nunca se reutilizan**: una neurona que
//! desaparece y vuelve más lejos que `tolerancia` aparece como unidad nueva, que
//! es un error visible y corregible fuera de línea, en vez de una fusión silenciosa.

#[derive(Debug, Clone)]
struct Unidad {
    id: i64,
    centroide: Vec<f32>,
    ultimo_lote: u64,
}

#[derive(Debug, Clone)]
pub struct Identidades {
    unidades: Vec<Unidad>,
    siguiente: i64,
    /// Distancia euclídea máxima (en el encaje normalizado) para considerar que
    /// dos centroides de lotes distintos son la misma unidad.
    pub tolerancia: f32,
    /// Lotes sin ver una unidad antes de olvidarla. Su id no se reutiliza.
    pub olvido_lotes: u64,
    lote: u64,
}

impl Identidades {
    pub fn new(tolerancia: f32, olvido_lotes: u64) -> Self {
        Self { unidades: Vec::new(), siguiente: 0, tolerancia, olvido_lotes, lote: 0 }
    }

    /// Asigna un id estable a cada centroide de este lote, en el mismo orden.
    pub fn asigna(&mut self, centroides: &[Vec<f32>]) -> Vec<i64> {
        self.lote += 1;
        let mut ids = vec![-1i64; centroides.len()];

        // Todas las parejas (nuevo, conocido) dentro de tolerancia, de cerca a lejos.
        let mut parejas: Vec<(f32, usize, usize)> = Vec::new();
        for (i, c) in centroides.iter().enumerate() {
            for (j, u) in self.unidades.iter().enumerate() {
                let d = distancia(c, &u.centroide);
                if d <= self.tolerancia {
                    parejas.push((d, i, j));
                }
            }
        }
        parejas.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut nuevo_usado = vec![false; centroides.len()];
        let mut conocido_usado = vec![false; self.unidades.len()];
        for (_, i, j) in parejas {
            if nuevo_usado[i] || conocido_usado[j] {
                continue;
            }
            nuevo_usado[i] = true;
            conocido_usado[j] = true;
            ids[i] = self.unidades[j].id;
            // El centroide sigue a la unidad: así tolera la deriva lenta.
            self.unidades[j].centroide = centroides[i].clone();
            self.unidades[j].ultimo_lote = self.lote;
        }

        for (i, c) in centroides.iter().enumerate() {
            if !nuevo_usado[i] {
                let id = self.siguiente;
                self.siguiente += 1;
                self.unidades.push(Unidad { id, centroide: c.clone(), ultimo_lote: self.lote });
                ids[i] = id;
            }
        }

        let (lote, olvido) = (self.lote, self.olvido_lotes);
        self.unidades.retain(|u| lote - u.ultimo_lote <= olvido);
        ids
    }

    /// Unidades vivas ahora mismo.
    pub fn n_unidades(&self) -> usize {
        self.unidades.len()
    }

    /// Ids entregados en toda la sesión (vivos u olvidados).
    pub fn n_ids_emitidos(&self) -> i64 {
        self.siguiente
    }
}

fn distancia(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

#[cfg(test)]
mod pruebas {
    use super::*;

    #[test]
    fn conserva_ids_aunque_cambie_el_orden() {
        let mut ids = Identidades::new(0.1, 10);
        let a = ids.asigna(&[vec![0.1, 0.1], vec![0.9, 0.9]]);
        // Mismo par de unidades, en orden inverso y con algo de deriva.
        let b = ids.asigna(&[vec![0.88, 0.91], vec![0.12, 0.09]]);
        assert_eq!(a, vec![0, 1]);
        assert_eq!(b, vec![1, 0]);
    }

    #[test]
    fn unidad_nueva_recibe_id_nuevo() {
        let mut ids = Identidades::new(0.1, 10);
        ids.asigna(&[vec![0.1, 0.1]]);
        let b = ids.asigna(&[vec![0.1, 0.1], vec![0.5, 0.5]]);
        assert_eq!(b, vec![0, 1]);
    }

    #[test]
    fn dos_nuevos_no_se_quedan_con_el_mismo_id() {
        let mut ids = Identidades::new(0.2, 10);
        ids.asigna(&[vec![0.5, 0.5]]);
        // Una unidad que se partió en dos: la más cercana hereda el id.
        let b = ids.asigna(&[vec![0.45, 0.5], vec![0.52, 0.5]]);
        assert_eq!(b, vec![1, 0]);
    }

    #[test]
    fn olvida_pero_no_reutiliza() {
        let mut ids = Identidades::new(0.1, 1);
        ids.asigna(&[vec![0.1, 0.1]]);
        ids.asigna(&[]);
        ids.asigna(&[]);
        assert_eq!(ids.n_unidades(), 0);
        assert_eq!(ids.asigna(&[vec![0.1, 0.1]]), vec![1]);
    }
}
