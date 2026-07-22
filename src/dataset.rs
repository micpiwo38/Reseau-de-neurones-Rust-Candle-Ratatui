
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use rand::Rng;
use tokenizers::Tokenizer;
use crate::tokenizer::{build_bpe_tokenizer, format_chatml, Sample};

pub struct TextDataset{
    pub raw_text: String,
    pub tokenizer: Tokenizer,
    pub vocab_size:usize,
    pub token_ids: Vec<u32>,
    pub vocab: Vec<String>,
}

impl TextDataset{

    //Taille du vecteur voculaire = (nombre de tokens uniques possible)
    pub fn vocabulaire_length(&self) -> usize{
        self.vocab_size
    }

    pub fn load_dataset<P: AsRef<Path>>(path: P, target_vocab_size: usize) -> io::Result<Self>{
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut combined_text = String::new();

        for (index, line) in reader.lines().enumerate() {
            let line_content = line?;
            if line_content.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<Sample>(&line_content) {
                Ok(sample) => {
                    let formatted = format_chatml(&sample);
                    combined_text.push_str(&formatted);
                }
                Err(e) => {
                    eprintln!("Erreur de parsing JSONL ligne {} : {}", index + 1, e);
                }
            }
        }
        //Entrainer le tokenizer BPE (Byte Pair encoding) sur le texte formaté
        let tokenizer = build_bpe_tokenizer(&combined_text, target_vocab_size)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        let vocab_size = tokenizer.get_vocab_size(true);
        //Pré encoder tout le texte en IDs de tokens BPE
        let encoding = tokenizer
            .encode(combined_text.as_str(), true)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let token_ids = encoding.get_ids().to_vec();
        println!("Le dataset BPE (Byte-Pair Encoding) est prêt : {:?} : tokens total ! ", token_ids.len());
        // 1. Récupérer le vocabulaire du tokenizer (HashMap<String, u32>)
        let vocab_map = tokenizer.get_vocab(true);

        // 2. Créer un Vec de la taille du vocabulaire
        let mut vocab = vec![String::new(); vocab_size];

        // 3. Remplir le Vec aux bons indices (id)
        for (token_str, id) in vocab_map {
            if (id as usize) < vocab_size {
                vocab[id as usize] = token_str;
            }
        }

        Ok(Self{
            raw_text: combined_text,
            tokenizer,
            vocab_size,
            token_ids,
            vocab
        })

    }

    // Générer les lots (batches) sous forme de Tenseurs Candle à partir des token_ids
    // Renvoie simplement x (indices des tokens) et y (targets) sous forme de u32
    pub fn generate_batch_tensor(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &candle_core::Device,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        let mut x_batch = Vec::with_capacity(batch_size * seq_len);
        let mut y_batch = Vec::with_capacity(batch_size * seq_len);

        for _ in 0..batch_size {
            // Récupère une séquence d'indices de ton dataset
            let (x_seq, y_seq) = self.get_random_sample(seq_len);
            x_batch.extend(x_seq);
            y_batch.extend(y_seq);
        }

        // Tenseurs d'indices direct en U32 sur le GPU !
        let x_tensor = candle_core::Tensor::from_vec(x_batch, (batch_size, seq_len), device)?;
        let y_tensor = candle_core::Tensor::from_vec(y_batch, (batch_size, seq_len), device)?;

        Ok((x_tensor, y_tensor))
    }

    pub fn get_random_sample(&self, seq_len: usize) -> (Vec<u32>, Vec<u32>) {
        let mut rng = rand::thread_rng();

        // Index de départ aléatoire dans le dataset
        let max_start = self.token_ids.len().saturating_sub(seq_len + 1);
        let start_idx = rng.gen_range(0..=max_start);

        // X = les tokens originaux (longueur = seq_len)
        let x = self.token_ids[start_idx..start_idx + seq_len].to_vec();

        // Y = les mêmes tokens décalés de 1 (longueur = seq_len)
        let y = self.token_ids[start_idx + 1..=start_idx + seq_len].to_vec(); // <-- Remarque le '=' ici

        (x, y)
    }

    // Encoder une chaîne de caractères en IDs de tokens avec le Tokenizer BPE
    pub fn encoder(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode(text, true)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    //L'inverse : on decode une liste d'entier ID en caractère lisible
    pub fn decoder(&self, tokens: &[u32]) -> String {
        let mut texte = String::new();

        for &id in tokens {
            if let Some(token_str) = self.vocab.get(id as usize) {
                texte.push_str(token_str);
            }
        }

        // --- C'EST ICI QU'IL FAUT AJOUTER LE NETTOYAGE ---
        texte = texte
            .replace('Ġ', " ")   // Remplace 'Ġ' par un vrai espace
            .replace('Ċ', "\n")  // Remplace 'Ċ' par un saut de ligne
            .replace('ĉ', "\t"); // Remplace 'ĉ' par une tabulation

        texte
    }
}