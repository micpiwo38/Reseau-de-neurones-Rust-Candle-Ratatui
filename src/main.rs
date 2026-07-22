use std::io;

pub mod app;
pub mod dataset;
pub mod model;
pub mod tokenizer;

use crate::app::{App};

fn main() -> io::Result<()> {
    // Force NVCC à passer l'option au préprocesseur MSVC
    println!("cargo:rustc-env=NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor");
    // Optionnel : passer aussi la macro de suppression au cas où
    // println!("cargo:rustc-env=NVCC_PREPEND_FLAGS=-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING");
    App::run().expect("Erreur de création de l'application MIC_AI !!!");
    println!("Merci d'avoir utilisé le LLM MIC_IA !");

    Ok(())
}
