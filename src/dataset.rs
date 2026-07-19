use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::Path;
use candle_core::{Tensor, Device, Result as CandleResult};
pub struct TextDataset{
    pub raw_text: String,
    pub vocabulaire: Vec<char>, //Tableau Vecteur de caractère
}

impl TextDataset{
    //1. Charger le fichier de texte brut et extraire le vocabulaire unique
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self>{
        let mut file = File::open(path)?;
        let mut raw_text = String::new();
        file.read_to_string(&mut raw_text)?;

        //Extraire les caractères 1 a 1
        let mut char_set = HashSet::new();
        //Boucle de parcors de chaque caractère de chaque mot
        for c in raw_text.chars(){
            char_set.insert(c);
        }

        let mut vocabulaire: Vec<char> = char_set.into_iter().collect();
        vocabulaire.sort(); //Tri dans un ordre deterministe
        Ok(Self{raw_text, vocabulaire})
    }

    //Taille du vecteur voculaire = (nombre de tokens uniques possible)
    pub fn vocabulaire_length(&self) -> usize{
        self.vocabulaire.len()
    }
    //Generer des lot (Batch) alatoire de données d'entrainement sous forme de tenseur crate Candle
    pub fn generate_batch_tensor(&self, batch_size: usize, seq_len: usize, device: &Device) -> CandleResult<(Tensor, Tensor)> {
        let text_encode = self.encoder(&self.raw_text);
        //Longueur du texte encoder
        let n = text_encode.len();
        //On creer 2 instances des vecteurs
        let mut x_vectors = Vec::new();
        let mut y_vectors = Vec::new();
        //On pioche des index vecteur au hasard dans les données texte pour creer un batch
        for i in 0..batch_size{
            let index_start = (i * 12345) % (n - seq_len - 1);
            let x_seq = &text_encode[index_start..index_start + seq_len];
            let y_seq = &text_encode[index_start + 1..index_start + seq_len + 1];

            x_vectors.extend_from_slice(x_seq); //Extrait des tranches vecteur random
            y_vectors.extend_from_slice(y_seq);
        }
        //Tranformation des vecteur Rust en tenseur Candle = Matrice BI-Dimenssion
        let x_tensor = Tensor::from_vec(x_vectors, (batch_size, seq_len), device)?;
        let y_tensor = Tensor::from_vec(y_vectors, (batch_size, seq_len), device)?;

        Ok((x_tensor, y_tensor))
    }
    //Encoder une chaine de caractère en liste ID (entier)
    pub fn encoder(&self, text: &str) -> Vec<u32>{
        text.chars().map(|c|{
            self.vocabulaire.iter().position(|&v| v == c).unwrap_or(0) as u32
        })
            .collect()
    }

    //L'inverse : on decode une liste d'entier ID en caractère lisible
    pub fn decoder(&self, ids: &[u32]) -> String{
        ids.iter().map(|&id| self.vocabulaire.get(id as usize).cloned().unwrap_or('?')).collect()
    }
}