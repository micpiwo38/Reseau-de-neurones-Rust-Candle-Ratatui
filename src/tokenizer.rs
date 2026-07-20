use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use serde::Deserialize;

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