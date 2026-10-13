//! Comprueba que el submódulo de umap-mcqueen está y que las versiones de las
//! dependencias que tocan el formato de los pesos coinciden con las suyas.

use std::path::Path;

const FUENTES: [&str; 2] = ["umap-mcqueen/src/modelo.rs", "umap-mcqueen/src/normaliza.rs"];
const DEBEN_COINCIDIR: [&str; 4] = ["fast-umap", "burn", "cubecl", "clump"];

fn version_de(toml: &str, crate_: &str) -> Option<String> {
    toml.lines()
        .map(str::trim)
        .find(|l| l.starts_with(crate_) && l[crate_.len()..].trim_start().starts_with('='))
        .and_then(|l| l.split('"').nth(1).map(str::to_string))
}

fn main() {
    for f in FUENTES {
        println!("cargo:rerun-if-changed={f}");
        if !Path::new(f).exists() {
            panic!(
                "\n\nFalta {f}.\numap-mcqueen va como submódulo; desde la raíz del fork:\n    \
                 git submodule update --init umap_rust/umap-mcqueen\n\n"
            );
        }
    }
    println!("cargo:rerun-if-changed=umap-mcqueen/Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let suyo = std::fs::read_to_string("umap-mcqueen/Cargo.toml").unwrap_or_default();
    let nuestro = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
    for c in DEBEN_COINCIDIR {
        let (a, b) = (version_de(&suyo, c), version_de(&nuestro, c));
        if a.is_some() && a != b {
            panic!(
                "\n\n{c}: umap-mcqueen usa {a:?} y tdc-umap {b:?}.\nIguala la versión en \
                 umap_rust/Cargo.toml; si no, los pesos guardados por umap-mcqueen pueden no \
                 cargarse igual aquí.\n\n"
            );
        }
    }
}
