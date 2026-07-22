use candle_core::{Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, GRU, RNN};
use std::default::Default;
use std::sync::mpsc::Sender;
use crate::app::IAMessage;

pub struct Model {
    pub embed: Embedding,
    pub gru: GRU, // Gated Recurrent Unit
    fc: Linear,
    pub vocab_size: usize,
    pub hidden_dim: usize,
}

impl Model {
    pub fn new(
        vb: VarBuilder,
        vocab_size: usize,
        embed_dim: usize,
        hidden_dim: usize,
    ) -> CandleResult<Self> {
        let embed = candle_nn::embedding(vocab_size, embed_dim, vb.pp("embed"))?;
        let gru = candle_nn::gru(embed_dim, hidden_dim, Default::default(), vb.pp("gru"))?;
        let fc = candle_nn::linear(hidden_dim, vocab_size, vb.pp("fc"))?;

        Ok(Self {
            embed,
            gru,
            fc,
            vocab_size,
            hidden_dim,
        })
    }

    fn sample_multinomial(probs: &[f32]) -> usize {
        let mut rng = rand::thread_rng();
        let sample: f32 = rand::Rng::r#gen(&mut rng);

        let mut cumulative_prob = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative_prob += p;
            if sample <= cumulative_prob {
                return i;
            }
        }
        probs.len().saturating_sub(1)
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len) = x.dims2()?;

        let x_embedded = self.embed.forward(x)?;

        let mut state = self.gru.zero_state(batch_size)?;
        let mut outputs = Vec::with_capacity(seq_len);

        for s in 0..seq_len {
            let current_input = x_embedded
                .narrow(1, s, 1)?
                .squeeze(1)?
                .contiguous()?;

            state = self.gru.step(&current_input, &state)?;
            outputs.push(state.h.clone());
        }

        let states_tensor = Tensor::stack(&outputs, 1)?;
        let flattened_states = states_tensor
            .reshape((batch_size * seq_len, self.hidden_dim))?
            .contiguous()?;

        let output = self.fc.forward(&flattened_states)?;
        output.reshape((batch_size, seq_len, self.vocab_size))
    }

    /// Generer une réponse à partir du prompt utilisateur
    /// Formate l'entrée, élimine la répétition du prompt et gère le streaming UI
    pub fn generate_response(
        &self,
        cmd: &str,
        longueur: usize,
        temperature: f32,
        dataset: &crate::dataset::TextDataset,
        device: &Device,
        tx_ia: &Sender<IAMessage>, // Passer l'émetteur pour le streaming UI
    ) -> CandleResult<String> {
        // 1. Adapter le format du prompt (Ajuste "Question:" / "Réponse:" selon la structure de ton JSONL)
        let prompt_formate = format!("Question: {}\nRéponse:", cmd);

        let mut progression_texte = prompt_formate.clone();
        let mut reponse_seule = String::new();

        for _ in 0..longueur {
            let encode = dataset.encoder(&progression_texte);
            if encode.is_empty() {
                break;
            }

            // Fenêtre glissante de 32 tokens max pour le contexte
            let debut = encode.len().saturating_sub(32);
            let contexte = &encode[debut..];

            let x = Tensor::from_vec(contexte.to_vec(), (1, contexte.len()), device)?;
            let logits = self.forward(&x)?;

            let batch_logits = logits.get(0)?;
            let seq_len_idx = batch_logits.dim(0)?.saturating_sub(1);
            let final_logits = batch_logits.get(seq_len_idx)?;

            let id_u32 = if temperature <= 0.0 {
                let scores = final_logits.to_vec1::<f32>()?;
                let mut max_val = scores[0];
                let mut max_idx = 0u32;
                for (idx, &val) in scores.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = idx as u32;
                    }
                }
                max_idx
            } else {
                let scaled_logits = (&final_logits / (temperature as f64))?;
                let probs_tensor = candle_nn::ops::softmax(&scaled_logits, 0)?;
                let probs = probs_tensor.to_vec1::<f32>()?;

                Self::sample_multinomial(&probs) as u32
            };

            // Décoder uniquement le nouveau token généré
            let nouveau_token = dataset.decoder(&[id_u32]);

            // Si le modèle prédit une fin de ligne/réponse ou un token spécial
            if nouveau_token == "\n" && reponse_seule.len() > 10 {
                break;
            }

            // Mise à jour des chaînes
            progression_texte.push_str(&nouveau_token);
            reponse_seule.push_str(&nouveau_token);

            // 2. Streamer immédiatement le token vers l'interface TUI !
            let _ = tx_ia.send(IAMessage::StreamChunk(nouveau_token));
        }

        // 3. On ne retourne que le texte généré (sans le prompt)
        Ok(reponse_seule)
    }
}