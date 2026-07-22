use std::fs;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use serde::Deserialize;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{Result as TokenizerResult, Tokenizer};



#[derive(Deserialize, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug)]
pub struct Sample {
    pub messages: Vec<Message>,
}

// Fonction pour formater chaque message au format ChatML si besoin
pub(crate) fn format_chatml(sample: &Sample) -> String {
    let mut text = String::new();
    for msg in &sample.messages {
        text.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    text
}

pub fn load_from_jsonl<P: AsRef<Path>>(path: P) -> io::Result<String> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut combined_text = String::new();
    let mut parsed_count = 0;

    for (index, line) in reader.lines().enumerate() {
        let line_content = line?;
        if line_content.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<Sample>(&line_content) {
            Ok(sample) => {
                let formatted = format_chatml(&sample);
                combined_text.push_str(&formatted);
                parsed_count += 1;
            }
            Err(e) => {
                eprintln!("Erreur de parsing JSONL ligne {} : {}", index + 1, e);
            }
        }
    }

    if combined_text.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Aucune ligne valide n'a pu être lue (0/{} lignes lues). Vérifie la structure de tes structs Serde !", parsed_count)
        ));
    }

    println!("Dataset chargé avec succès : {} échantillons lus.", parsed_count);
    Ok(combined_text)
}

//Entrainer un tokenizer BPE (Byte pair encoding) directement sur le texte formaté
// 3. Entraîne un Tokenizer BPE directement sur ton texte formaté
pub fn build_bpe_tokenizer(text: &str, vocab_size: usize) -> TokenizerResult<Tokenizer> {
    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));

    let special_tokens = vec![
        tokenizers::AddedToken::from("<|im_start|>", true),
        tokenizers::AddedToken::from("<|im_end|>", true),
        tokenizers::AddedToken::from("<|unk|>", true),
    ];

    let mut bpe_trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(special_tokens)
        .build();

    let mut trainer: TrainerWrapper = bpe_trainer.into();

    // 1. On sauvegarde temporairement le texte combiné dans un fichier
    let temp_file = "temp_train_corpus.txt";
    fs::write(temp_file, text).expect("Impossible d'écrire le fichier de corpus temporaire");

    // 2. On entraîne le tokenizer à partir de ce fichier
    tokenizer.train_from_files(&mut trainer, vec![temp_file.to_string()])?;

    // 3. On nettoie le fichier temporaire
    let _ = fs::remove_file(temp_file);

    println!(
        "Tokenizer BPE créé avec un vocabulaire de {} tokens !",
        tokenizer.get_vocab_size(true)
    );
    Ok(tokenizer)
}