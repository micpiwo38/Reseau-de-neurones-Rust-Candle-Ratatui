
use candle_core::{Device, Result as CandleResult, Tensor, DType, Var, Module};
use candle_nn::{Linear, VarBuilder, VarMap, GRU, RNN};
use rand::Rng;

pub struct Model{
    pub gru: GRU, //Gated Recurent Unit => eviter d'oublier les entrainement
    fc: Linear,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    varmap: VarMap,
}

impl Model{
    //Init d'une nouvelle couche linéaire avec des poids aléatoire
    pub fn new(vocab_size: usize, hidden_dim: usize, device: &Device) -> CandleResult<Self>{
        //VarMap =  initialisation et stockage de nos variables (poids)
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

        //1. La couche GRU gere la memoire, elle prend en entrée un caractère (one-hot vocab_size) et le met en cache (hidden_dim)
        let gru = candle_nn::gru(vocab_size, hidden_dim, Default::default(), vs.pp("gru"))?;
        //2. Une couche lineaire de sortie pour re-transformer la memoire en scores de caractères
        let fc = candle_nn::linear(hidden_dim, vocab_size, vs.pp("fc"))?;

        Ok(Self{
            gru, fc, vocab_size, hidden_dim, varmap
        })
    }
    //Helper
    pub fn variables(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    //Forward Pass prend l'entrée X at applique les poids et le bias
    /// Le "Forward Pass" corrigé avec détection de rang flexible
    /// Le Forward Pass avec mémoire temporelle
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len) = x.dims2()?;

        // 1. One-Hot encoding de l'entrée [batch_size, seq_len] -> [batch_size, seq_len, vocab_size]
        let mut on_hot_data = vec![0.0f32; batch_size * seq_len * self.vocab_size];
        let x_ids = x.to_dtype(candle_core::DType::U32)?.to_vec2::<u32>()?;
        for b in 0..batch_size {
            for s in 0..seq_len {
                let id = x_ids[b][s] as usize;
                if id < self.vocab_size {
                    on_hot_data[b * self.vocab_size * seq_len + s * self.vocab_size + id] = 1.0;
                }
            }
        }

        let x_one_hot = Tensor::from_vec(on_hot_data, (batch_size, seq_len, self.vocab_size), x.device())?;

        // 2. Traitement manuel de la séquence
        let mut outputs = Vec::with_capacity(seq_len);

        // Initialisation de l'état initial
        let mut state = self.gru.zero_state(batch_size)?;

        // On parcourt la séquence étape par étape
        for s in 0..seq_len {
            let current_input = x_one_hot.narrow(1, s, 1)?.squeeze(1)?;

            // state est un GRUState
            state = self.gru.step(&current_input, &state)?;

            // L'astuce magique : state.h permet d'extraire le vrai Tensor caché dans le GRUState !
            outputs.push(state.h.clone());
        }

        // Maintenant, outputs contient uniquement des Tensor, Tensor::stack va adorer !
        let states_tensor = Tensor::stack(&outputs, 1)?;

        // 3. Projection linéaire finale
        let flattened_states = states_tensor.reshape((batch_size * seq_len, self.hidden_dim))?;
        let output = self.fc.forward(&flattened_states)?;

        // 4. Forme de sortie finale = tenseur a 3 dimensions
        output.reshape((batch_size, seq_len, self.vocab_size))
    }


    //Generer une reponse a partir du prompt utilisateur
    /// Génère du texte avec échantillonnage par Température
    pub fn generate_response(
        &self,
        amorce: &str,
        longueur: usize,
        dataset: &crate::dataset::TextDataset,
        device: &Device,
    ) -> CandleResult<String> {
        let mut progression_texte = amorce.to_string();

        for _ in 0..longueur {
            let encode = dataset.encoder(&progression_texte);
            // On prend une fenêtre glissante des 32 derniers caractères pour la mémoire du GRU
            let debut = encode.len().saturating_sub(32);
            let contexte = &encode[debut..];
            // Dans ta fonction generer / generate_response, au milieu de la boucle for :
            let x = Tensor::from_vec(contexte.to_vec(), (1, contexte.len()), device)?;
            let logits = self.forward(&x)?; // Forme obtenue : [1, seq_len, vocab_size]

            // 1. On extrait le premier (et seul) batch -> Forme : [seq_len, vocab_size]
            let batch_logits = logits.get(0)?; // [seq_len, vocab_size]
            // 2. On extrait la DERNIÈRE étape temporelle de la séquence -> Forme : [vocab_size] (1D)
            let seq_len_idx = batch_logits.dim(0)? -1;
            let final_logits = batch_logits.get(seq_len_idx)?; // [vocab_size]
            // 3. On convertit ce tenseur 1D en Vec<f32> pour l'échantillonnage
            let scores = final_logits.to_vec1::<f32>()?;
            let mut max_val = scores[0];
            let mut id_u32 = 0u32;
            for (idx, &val) in scores.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    id_u32 = idx as u32;
                }
            }

            let caractere_suivant = dataset.decoder(&[id_u32]);
            progression_texte.push_str(&caractere_suivant);
        }

        Ok(progression_texte)
    }
}