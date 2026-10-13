# umap_rust — puente tridesclous ↔ umap-mcqueen

Compila el módulo de Python `tdc_umap`, que usa el motor `engine='umap'` del
Peeler (`peeler_engine_umap.py`).

umap-mcqueen **no se modifica ni se copia**: va como submódulo en
`umap-mcqueen/` y aquí se compilan sus `src/modelo.rs` y `src/normaliza.rs` tal
cual (`#[path]` en `src/lib.rs`). Lo único propio de este crate es:

* `src/motor.rs` — normaliza → `transform` → normaliza → DenStream, por lote,
  y devuelve **una etiqueta por forma de onda**.
* `src/identidades.rs` — ids estables entre lotes (DenStream sólo da posiciones
  en un `Vec`, que cambian con cada poda).
* `src/lib.rs` — la clase de Python y la configuración de UMAP, reconstruida del
  manifiesto del modelo (no hay una segunda copia de `configuracion()`).

## Compilar

La primera vez, al aplicar el parche, se añade el submódulo (queda fijado al
commit actual de umap-mcqueen):

    git submodule add https://github.com/ocelotzin/umap-mcqueen.git umap_rust/umap-mcqueen

En cualquier clon posterior del fork basta con:

    git submodule update --init umap_rust/umap-mcqueen

Luego:

    cd umap_rust
    cargo test                       # pruebas de identidades.rs y normaliza.rs
    pip install maturin
    maturin develop --release        # instala tdc_umap en el entorno activo
    python -c "import tdc_umap; print(tdc_umap.Clasificador)"

`build.rs` falla con un mensaje claro si el submódulo no está, o si las
versiones de fast-umap, burn, cubecl o clump no coinciden con las de
umap-mcqueen (cambiarían el formato de los pesos).

## Actualizar umap-mcqueen

    cd umap_rust/umap-mcqueen && git pull && cd ../..
    git add umap_rust/umap-mcqueen && git commit -m "umap-mcqueen: actualizar"

Si `modelo.rs` cambia su API pública (`carga_verificada`, `Manifiesto`,
`RutasDelModelo`), habrá que ajustar `motor.rs`/`lib.rs`; nada más.
