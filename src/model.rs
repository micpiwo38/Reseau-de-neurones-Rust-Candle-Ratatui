
use candle_core::{Device, Result as CandleResult, Tensor, DType, Var};
use rand::Rng;

pub struct Model{
    pub weights: Var,
    pub bias: Var,
}

impl Model{
    //Init d'une nouvelle couche linéaire avec des poids aléatoire
    pub fn new(vocab_size: usize, device: &Device) -> CandleResult<Self>{
        //Poids aléatoire
        let w_init = Tensor::randn(0.0f32, 0.1f32, (vocab_size, vocab_size), device)?;
        //Par defaut le biais est a 0
        let b_init = Tensor::zeros((1, vocab_size), DType::F32, device)?;
        //1. Matrice de poids de taille [vocab_taille, vocan_taille]
        //On utilise rand N pour casser la symetrie
        let weights = Var::from_tensor(&w_init)?;
        //Un bias est un écart entre la vraie valeur d'une variable inobservable et la valeur estimée statistiquement
        //2. Un bias pour chaque caractères du vecteur vocubulaire [1 et vocab taille]
        let bias = Var::from_tensor(&b_init)?;

        Ok(Self{weights, bias})
    }

    //Forward Pass prend l'entrée X at applique les poids et le bias
    /// Le "Forward Pass" corrigé avec détection de rang flexible
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Au lieu de forcer dims2(), on analyse la forme de manière flexible
        let dims = x.dims();
        let (batch_size, seq_len) = match dims.len() {
            1 => (1, dims[0]), // Si c'est un tenseur 1D [seq_len]
            2 => (dims[0], dims[1]), // Si c'est un tenseur 2D [batch_size, seq_len]
            _ => return Err(candle_core::Error::Msg(format!("Tenseur d'entrée invalide, dimensions reçues : {:?}", dims))),
        };

        let vocab_size = self.bias.as_tensor().dim(1)?; // 97

        // 1. Convertir les IDs en vecteurs de taille 'vocab_size' (One-hot encoding)
        let mut on_hot_data = vec![0.0f32; batch_size * seq_len * vocab_size];

        // On aplatit x temporairement en un vecteur 1D pour lire facilement ses IDs
        let x_flat = x.flatten_all()?.to_dtype(candle_core::DType::U32)?.to_vec1::<u32>()?;

        for b in 0..batch_size {
            for s in 0..seq_len {
                let flat_idx = b * seq_len + s;
                let id = x_flat[flat_idx] as usize;
                if id < vocab_size {
                    on_hot_data[b * seq_len * vocab_size + s * vocab_size + id] = 1.0;
                }
            }
        }

        // On transforme nos données one-hot en Tenseur Candle [batch_size * seq_len, vocab_size]
        let x_one_hot = Tensor::from_vec(on_hot_data, (batch_size * seq_len, vocab_size), x.device())?;

        // 2. Équation linéaire : Y = X * W + B
        let mut y = x_one_hot.matmul(&self.weights.as_tensor())?;
        y = y.broadcast_add(&self.bias.as_tensor())?;

        // 3. On redonne à la sortie sa forme 3D d'origine : [batch_size, seq_len, vocab_size]
        y.reshape((batch_size, seq_len, vocab_size))
    }

    //Recuperer la liste de toutes les variables a mettre a jour
    pub fn variables(&self) -> Vec<Var>{
        vec![self.weights.clone(), self.bias.clone()]
    }

    //Generer une reponse a partir du prompt utilisateur
    /// Génère du texte avec échantillonnage par Température
    pub fn generate_response(
        &self,
        amorce: &str,
        longueur: usize,
        temperature: f32, // Nouvelle variable ! (ex: 0.7)
        dataset: &crate::dataset::TextDataset,
        device: &Device,
    ) -> CandleResult<String> {
        let mut progression_texte = amorce.to_string();
        let mut rng = rand::thread_rng();

        for _ in 0..longueur {
            let encode = dataset.encoder(&progression_texte);
            let debut = encode.len().saturating_sub(16);
            let contexte = &encode[debut..];

            let x = Tensor::from_vec(contexte.to_vec(), (1, contexte.len()), device)?;
            let logits = self.forward(&x)?;

            let last_step_logits = logits.get(0)?;
            let final_logits = last_step_logits.get(last_step_logits.dim(0)? - 1)?;

            // Convertit le tenseur de scores en Vec Rust
            let scores = final_logits.to_vec1::<f32>()?;
            let vocab_size = scores.len();

            // --- APPLICATION DE LA TEMPÉRATURE & SOFTMAX MANUEL ---
            // 1. Division des scores par la température et passage à l'exponentielle
            let mut exp_scores: Vec<f32> = scores
                .iter()
                .map(|&s| (s / temperature).exp())
                .collect();

            // 2. Calcul de la somme pour normaliser (Somme des probabilités = 1.0)
            let somme_exp: f32 = exp_scores.iter().sum();

            // 3. Transformation en probabilités
            let probabilites: Vec<f32> = exp_scores
                .iter_mut()
                .map(|val| *val / somme_exp)
                .collect();

            // --- TIRAGE AU SORT PONDÉRÉ (Hasard contrôlé) ---
            let mut tirage: f32 = rng.r#gen(); // Nombre aléatoire entre 0.0 et 1.0
            let mut id_u32 = 0u32;
            let mut somme_cumulee = 0.0;

            for (idx, &prob) in probabilites.iter().enumerate() {
                somme_cumulee += prob;
                if tirage <= somme_cumulee {
                    id_u32 = idx as u32;
                    break;
                }
            }

            // Décodage et ajout du caractère
            let caractere_suivant = dataset.decoder(&[id_u32]);
            progression_texte.push_str(&caractere_suivant);
        }

        Ok(progression_texte)
    }
}