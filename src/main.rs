use std::io;

pub mod app;
pub mod dataset;
pub mod model;
pub mod tokenizer;

use crate::app::{App};

fn main() -> io::Result<()> {
    App::run().expect("Erreur de création de l'application MIC_AI !!!");
    println!("Merci d'avoir utilisé le LLM MIC_IA !");

    Ok(())
}
