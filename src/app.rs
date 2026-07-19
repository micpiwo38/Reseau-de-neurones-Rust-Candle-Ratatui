use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, ListItem, Gauge, List, Paragraph },
};

use std::{
    io::{self, stdout},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};


//-----------------STRUCTURE UI DE APPLICATION--------------------//
//Enumeration des messages que le moteur IA peut envoyer a Ratatui
pub enum IAMessage{
    ProgressionEntainement(u16), // % de 0 a 100%
    ResponseChat(String), //Texte générer par le LLM
}

pub struct App{
    pub input: String, //Input utilisateur
    pub historique_chat: Vec<String>, //Historique de discussion
    pub progression: u16, //Progression de l'entrainement
    pub rx_ia: Receiver<IAMessage>, //Recepteur de message de l'IA
    pub tx_ia_cmd: Sender<String>, //Emetteur pour envoyer les ordre a l'IA
    pub cursor_position: usize, //Position du curseur dans le prompt
}

impl App{
    pub fn new(rx_ia: Receiver<IAMessage>, tx_ia_cmd: Sender<String>) -> Self{
        Self{
            input: String::new(),
            historique_chat: vec![
                "MIC_IA : Bienvenue dans le LLM Mic-IA !".to_string(),
                "MIC_IA : En attente du lancement de l'entraînement ...".to_string(),
            ],
            progression: 0,
            rx_ia,
            tx_ia_cmd,
            cursor_position: 0,
        }
    }

    //------------------------------------UI----------------------------//
    pub fn ui(frame: &mut Frame, app: &App){
        //Decoupe de l'ecran: ZOne proincipale en haut et barre de progression en bas
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(frame.area());

        //Decoupe de la zone principale, le chat en haut et input utilisateur en bas
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),
                Constraint::Length(3),
            ]).split(chunks[0]);

        //La Zone de chat
        let chat_items: Vec<ListItem> = app
            .historique_chat
            .iter()
            .map(|msg|{
                let style = if msg.starts_with("Vous : "){
                    Style::default().fg(Color::LightGreen)
                }else if msg.starts_with("MIC_IA : "){
                    Style::default().fg(Color::LightRed)
                }else{
                    Style::default().fg(Color::LightYellow).add_modifier(Modifier::ITALIC)
                };
                ListItem::new(Line::from(Span::styled(msg, style)))
            })
            .collect();

        let chat_list = List::new(chat_items)
            .block(Block::default().borders(Borders::ALL).title("Discussion avec LLM MIC_IA"));
        frame.render_widget(chat_list, main_chunks[0]);

        //Zone de saisie utilisateur
        let input_widget = Paragraph::new(app.input.as_str())
            .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL).title("Votre question ?"));
        frame.render_widget(input_widget, main_chunks[1]);

        // ---- AJOUT DU CURSEUR CLIGNOTANT ----
        // On place le curseur juste après le dernier caractère saisi
        // main_chunks[1].x + 1 pour sauter la bordure gauche du bloc
        // main_chunks[1].y + 1 pour sauter la bordure haute du bloc
        frame.set_cursor_position(Position::new(
            main_chunks[1].x + 1 + app.input.chars().count() as u16,
            main_chunks[1].y + 1,
        ));
        //Position dynamique du curseur
        frame.set_cursor_position(Position::new(
            main_chunks[1].x + 1 + app.cursor_position as u16,
            main_chunks[1].y + 1,
        ));

        //Barre de progression de l'entainement
        let titre_gauge = format!("Entraînement du modèle a partir du dataset.txt ... {}", app.progression);
        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(titre_gauge))
            .gauge_style(Style::default().fg(Color::LightGreen).bg(Color::Gray).add_modifier(Modifier::ITALIC))
            .percent(app.progression);
        frame.render_widget(gauge, chunks[1]);

    }

    //Test simulation entrainement
    pub fn training_simulation(tx_ia: Sender<IAMessage>, rx_cmd: Receiver<String>){
        thread::spawn(move ||{
            //Chemin absolu de la racine du projet
            let project_root = env!("CARGO_MANIFEST_DIR");
            //Construire un chemin propre vers le fichier de données
            let dataset_path = std::path::Path::new(project_root)
                .join("dataset")
                .join("php_mysql.txt");
            //1. Charger le dataset fichier php_mysql.txt
            let dataset = match crate::dataset::TextDataset::load_from_file(dataset_path) {
                Ok(dataset) => dataset,
                Err(e) => {
                    let _ = tx_ia.send(IAMessage::ResponseChat(format!("MIC_IA : Erreur de chargement et d'analyse de la base de données : {}", e)));
                    return;
                }
            };
            //2. Init du periphérique de calcul Candle tensor Batch avec le GPU
            let device = candle_core::Device::Cpu;
            let vocab_size = dataset.vocabulaire_length();

            let _ = tx_ia.send(IAMessage::ResponseChat(format!("MIC_IA : Données chargées ! Vocbulaire de {} de caractères unique.", vocab_size)));

            let hidden_dim = 128;
            //Initialisé le modele
            let model = match crate::model::Model::new(dataset.vocab_size, hidden_dim, &device) {
                Ok(model) => model,
                Err(e) => {
                    let _ = tx_ia.send(IAMessage::ResponseChat(format!("Erreur de chargement du modele et d'initialisation du reseau de neurone : {}", e)));
                    return;
                }
            };

            let _ = tx_ia.send(IAMessage::ResponseChat(
                "MIC_IA : Réseau de neurones initialisé avec succès !".to_string()
            ));

            //3. Boucle d'entrainement
            let total_step = 50;
            let batch_size = 8;
            let seq_len = 16;
            let learning_rate = 0.1f32; // Vitesse d'aprentissage

            for step in 1..=total_step{
                //1. Generation des lots (batchs) = X entrée, Y_true (ce que IA doit deviner)
                if let Ok((x, y_true)) = dataset.generate_batch_tensor(batch_size, seq_len, &device){
                    //2. FORWARD PASS = Calcul des prédictions
                    if let Ok(predictions) = model.forward(&x){
                        //3. Calcul de la perte LOSS (les erreurs de prédictions)
                        let pred_flat = match predictions.reshape((batch_size * seq_len, vocab_size)) {
                            Ok(predictions) => predictions,
                            Err(_) => continue,
                        };
                        //Applatire les cibles
                        let targets_flat = match y_true.reshape(batch_size * seq_len) {
                            Ok(targets) => targets,
                            Err(_) => continue,
                        };
                        //L'entropie caractérise l'aptitude de l'énergie contenue dans un système à fournir du travail,
                        // et donc également son incapacité à le faire : plus cette grandeur est élevée, plus l'énergie est dispersée, homogénéisée et donc moins utilisable
                        // Utilisation de la CrossEntropy native de Candle (Loss d'évaluation)
                        if let Ok(loss) = candle_nn::loss::cross_entropy(&pred_flat, &targets_flat){
                            //4. BACKWARD PASS = calcul automatique des gradients
                            //Gradient = champ de vecteurs qui combine en chaque point les différentes dérivées partielles
                            // et donne ainsi à la fois la direction de la variation la plus forte[1] localement et l’intensité de cette variation.
                            if let Ok(grads) = loss.backward() {

                                // 5. DESCENTE DE GRADIENT (Mise à jour des poids manuelle pour comprendre le mécanisme)
                                let vars = model.variables();
                                for var in vars {
                                    if let Some(grad_tensor) = grads.get(&var) {
                                        // Formule : W = W - (learning_rate * grad)
                                        if let Ok(update) = grad_tensor.clone() * (learning_rate as f64) {
                                            if let Ok(new_val) = var.as_tensor().sub(&update) {
                                                let _ = var.set(&new_val); // On applique les nouveaux poids !
                                            }
                                        }
                                    }
                                }
                            }
                            //Affichage regulier de la perte (Loss) dans ratatui pour suivre la progression
                            if step % 20 == 0 || step == 1{
                                if let Ok(loss_val) = loss.to_vec0::<f32>(){
                                    let _ = tx_ia.send(IAMessage::ResponseChat(format!(
                                        "Etape : {:03}/{} | Perte (Loss) : {:.4}",
                                        step, total_step, loss_val
                                    )));
                                }
                            }
                        }
                    }
                }
                let pourcentage = (step * 100 / total_step) as u16;
                let _ = tx_ia.send(IAMessage::ProgressionEntainement(pourcentage));
                thread::sleep(Duration::from_millis(20));
            }

            //Envoie du message dans le chat cpour confimer le chargent du fichier de données
            let _ = tx_ia.send(IAMessage::ResponseChat(
                "MIC_IA : Entraînement terminé avec succès ! Écris une amorce de code PHP et j'essaierai de la compléter.".to_string()
            ));
            //Attente de la question de l'utilisateur
            while let Ok(cmd) = rx_cmd.recv() {
                let _ = tx_ia.send(IAMessage::ResponseChat("MIC_IA : En cours de génération ...".to_string()));
                let temperature = 0.7f32; //Pertinence de la réponse en 0.1 et 1.0
                //Generer 50 caractères a partir du prompt utilisateur
                match model.generate_response(&cmd, 80, &dataset, &device) {
                    Ok(generated_response) => {
                        let _ = tx_ia.send(IAMessage::ResponseChat(format!(
                            "Votre réponse :\n{}", generated_response
                        )));
                        }
                    Err(e) => {
                        let _ = tx_ia.send(IAMessage::ResponseChat(format!(
                            "MIC_IA : Erreur lors de la génération de la réponse ! {}", e
                        )));
                    }
                }
            }
        });
    }

    //----------------------------------INIT IA-------------------------//
    pub fn run() -> io::Result<()>{
        //1.  Init des chanels de communication
        let (tx_ia, rx_ia) = mpsc::channel::<IAMessage>();
        let (tx_ia_cmd, rx_ia_cmd) = mpsc::channel::<String>();
        //2.  Lancement de la simulation d'entrainement
        Self::training_simulation(tx_ia, rx_ia_cmd);
        //3. COnfig du terminal ratatui
        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

        //4. Instance de l'application
        let mut app = App::new(rx_ia, tx_ia_cmd);
        //Timer
        let tick_rate = Duration::from_millis(250);
        //Dernière frame
        let mut last_tick = Instant::now();

        //5. Boucle principale du rendu
        loop {
            terminal.draw(|mut f| App::ui(&mut f, &app))?;
            //Gestion du timer pour ne pas saturer le CPU
            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press{
                        match key.code {
                            KeyCode::Esc => break, //Touche Echap quitte application
                            //Fleche de gauche et droite pour déplacer le curseur
                            KeyCode::Left => {
                                if app.cursor_position > 0{
                                    app.cursor_position -= 1;
                                }
                            }
                            KeyCode::Right => {
                                if app.cursor_position < app.input.chars().count(){
                                    app.cursor_position += 1;
                                }
                            }
                            //Inserer un caractère précisement ou ce situe le curseur
                            KeyCode::Char(c) => {
                                let byte_index = app.input
                                    .char_indices()
                                    .nth(app.cursor_position)
                                    .map(|(i, _)| i)
                                    .unwrap_or_else(|| app.input.len());

                                app.input.insert(byte_index, c);
                                app.cursor_position += 1;
                            }
                            KeyCode::Backspace => {
                                if app.cursor_position > 0{
                                    app.cursor_position -= 1;
                                }
                                if let Some((byte_index, _)) = app.input.char_indices().nth(app.cursor_position) {
                                    app.input.remove(byte_index);
                                }
                            }
                            KeyCode::Enter => {
                                if !app.input.is_empty(){
                                    let message = app.input.drain(..).collect::<String>(); //Stock de la question utilisateur
                                    app.historique_chat.push(format!("Vous : {}", message));
                                    //Envoie la commande au fil thread IA
                                    let _ = app.tx_ia_cmd.send(message);
                                    //Effacer le prompt + position du curseur
                                    app.input.clear();
                                    app.cursor_position = 0;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            //6. Verification des messages envoyé par IA en arrière plan
            while let Ok(msg) = app.rx_ia.try_recv() {
                match msg {
                    IAMessage::ProgressionEntainement(p) => {
                        app.progression = p;
                        if p == 100 && app.progression != p{
                            app.historique_chat.push("MIC_IA : Mon entraînement est terminé, je suis pret a te réponde ...".to_string());
                        }
                    }
                    IAMessage::ResponseChat(texte) => {
                        app.historique_chat.push(format!("MIC_IA : {}",texte));
                    }
                }
            }
            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }
        }
        //Restauration du terminal apres fermeture
        disable_raw_mode()?;
        stdout().execute(LeaveAlternateScreen)?;
        Ok(())
    }
}
