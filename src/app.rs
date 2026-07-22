use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Gauge, Paragraph},
};

use std::{
    io::{self, stdout},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};
use candle_core::{DType, Device};
use candle_nn::{Optimizer, ParamsAdamW, VarBuilder, VarMap};
use ratatui::widgets::Wrap;
use crate::model::Model;

//-----------------STRUCTURE UI DE APPLICATION--------------------//
pub enum IAMessage {
    ProgressionEntainement(u16), // % de 0 a 100%
    ResponseChat(String),        // Texte généré par le LLM
    StreamChunk(String),
}

pub struct App {
    pub input: String,                // Input utilisateur
    pub historique_chat: Vec<String>, // Historique de discussion
    pub progression: u16,             // Progression de l'entrainement
    pub rx_ia: Receiver<IAMessage>,   // Récepteur de message de l'IA
    pub tx_ia_cmd: Sender<String>,    // Émetteur pour envoyer les ordres a l'IA
    pub cursor_position: usize,      // Position du curseur dans le prompt
    pub scroll_offset: usize,
}

impl App {
    pub fn new(rx_ia: Receiver<IAMessage>, tx_ia_cmd: Sender<String>) -> Self {
        Self {
            input: String::new(),
            historique_chat: vec![
                "MIC_IA : LLM Mic-IA PHP Master !".to_string(),
                "MIC_IA : En attente du lancement de l'entraînement ...".to_string(),
            ],
            progression: 0,
            rx_ia,
            tx_ia_cmd,
            cursor_position: 0,
            scroll_offset: 0,
        }
    }

    //-------------------------Helper--------------------------------------//
    /*
    pub fn variables(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }
    */

    //------------------------------------UI----------------------------//
    pub fn ui(frame: &mut Frame, app: &App) {
        // Découpe de l'écran : Zone principale en haut et barre de progression en bas
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(3)])
            .split(frame.area());

        // Découpe de la zone principale : chat en haut et input utilisateur en bas
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(3)])
            .split(chunks[0]);

        // La Zone de chat
        let chat_items: Vec<Line> = app
            .historique_chat
            .iter()
            .map(|msg| {
                let style = if msg.starts_with("Vous : ") {
                    Style::default().fg(Color::LightGreen)
                } else if msg.starts_with("MIC_IA : ") {
                    Style::default().fg(Color::LightRed)
                } else {
                    Style::default()
                        .fg(Color::LightYellow)
                        .add_modifier(Modifier::ITALIC)
                };
                Line::from(Span::styled(msg, style))
            })
            .collect();

        let box_height = main_chunks[0].height as usize;
        let max_scroll = chat_items.len().saturating_sub(box_height.saturating_sub(2));
        let current_scroll = (app.scroll_offset as u16).min(max_scroll as u16);

        let chat_list = Paragraph::new(chat_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Discussion avec LLM MIC_IA MASTER PHP"),
            )
            .wrap(Wrap { trim: true })
            .scroll((current_scroll, 0));
        frame.render_widget(chat_list, main_chunks[0]);

        // Zone de saisie utilisateur
        let input_widget = Paragraph::new(app.input.as_str())
            .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL).title("Votre question ?"));
        frame.render_widget(input_widget, main_chunks[1]);

        // Placement dynamique du curseur
        frame.set_cursor_position(Position::new(
            main_chunks[1].x + 1 + app.cursor_position as u16,
            main_chunks[1].y + 1,
        ));

        // Barre de progression de l'entraînement
        let titre_gauge = format!("Entraînement du modèle : {} %", app.progression);
        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(titre_gauge))
            .gauge_style(
                Style::default()
                    .fg(Color::LightGreen)
                    .bg(Color::Gray)
                    .add_modifier(Modifier::ITALIC),
            )
            .percent(app.progression);
        frame.render_widget(gauge, chunks[1]);
    }

    // Moteur d'entraînement et de génération IA
    pub fn training_simulation(tx_ia: Sender<IAMessage>, rx_cmd: Receiver<String>) {
        thread::spawn(move || {
            let project_root = env!("CARGO_MANIFEST_DIR");
            let dataset_path = std::path::Path::new(project_root)
                .join("dataset")
                .join("php_mysql_QR.jsonl");

            if !dataset_path.exists() {
                let err_msg = format!("Fichier dataset introuvable : {:?}", dataset_path);
                let _ = tx_ia.send(IAMessage::ResponseChat(format!("MIC_IA : Erreur : {}", err_msg)));
                return;
            }

            // 1. Charger le dataset
            let dataset = match crate::dataset::TextDataset::load_dataset(&dataset_path, 2048) {
                Ok(dataset) => dataset,
                Err(e) => {
                    let _ = tx_ia.send(IAMessage::ResponseChat(format!("MIC_IA : Erreur dataset : {}", e)));
                    return;
                }
            };

            // 2. Détection Device
            let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
            if device.is_cuda() {
                let _ = tx_ia.send(IAMessage::ResponseChat(
                    "MIC_IA : Accélération matérielle CUDA activée !".to_string(),
                ));
            }

            if dataset.vocab_size == 0 {
                let _ = tx_ia.send(IAMessage::ResponseChat("MIC_IA : Dataset vide !".to_string()));
                return;
            }

            let _ = tx_ia.send(IAMessage::ResponseChat(format!(
                "MIC_IA : Données chargées ! Vocabulaire de {} tokens.", dataset.vocab_size
            )));

            // 3. Initialisation du modèle (UNE SEULE FOIS)
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

            let embed_dim = 128;
            let hidden_dim = 384;

            let model = match Model::new(vb, dataset.vocab_size, embed_dim, hidden_dim) {
                Ok(m) => m,
                Err(e) => {
                    let _ = tx_ia.send(IAMessage::ResponseChat(format!("Erreur création modèle: {}", e)));
                    return;
                }
            };

            let _ = tx_ia.send(IAMessage::ResponseChat(
                "MIC_IA : Réseau de neurones initialisé avec succès !".to_string(),
            ));

            // 4. Initialisation d'AdamW
            let params = ParamsAdamW {
                lr: 0.0003,
                ..Default::default()
            };

            let mut opt = match candle_nn::AdamW::new(varmap.all_vars(), params) {
                Ok(opt) => opt,
                Err(e) => {
                    let _ = tx_ia.send(IAMessage::ResponseChat(format!("Erreur AdamW: {}", e)));
                    return;
                }
            };

            // 5. Boucle d'entraînement
            let total_step = 5000;
            let batch_size = 32;
            let seq_len = 128;
            let mut last_percent = 0;

            for step in 1..=total_step {
                let (x, y_true) = match dataset.generate_batch_tensor(batch_size, seq_len, &device) {
                    Ok(res) => res,
                    Err(e) => {
                        let _ = tx_ia.send(IAMessage::ResponseChat(format!("Erreur batch: {}", e)));
                        break;
                    }
                };

                let predictions = match model.forward(&x) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = tx_ia.send(IAMessage::ResponseChat(format!("Erreur forward: {}", e)));
                        break;
                    }
                };

                let pred_flat = match predictions.reshape((batch_size * seq_len, dataset.vocab_size)) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = tx_ia.send(IAMessage::ResponseChat(format!("Erreur reshape pred: {}", e)));
                        break;
                    }
                };

                let targets_flat = match y_true.reshape(batch_size * seq_len) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx_ia.send(IAMessage::ResponseChat(format!("Erreur reshape target: {}", e)));
                        break;
                    }
                };

                match candle_nn::loss::cross_entropy(&pred_flat, &targets_flat) {
                    Ok(loss) => {
                        if let Err(e) = opt.backward_step(&loss) {
                            let _ = tx_ia.send(IAMessage::ResponseChat(format!("Erreur opt: {}", e)));
                            break;
                        }

                        if step % 100 == 0 || step == 1 {
                            if let Ok(loss_val) = loss.to_vec0::<f32>() {
                                let _ = tx_ia.send(IAMessage::ResponseChat(format!(
                                    "Step [{:05}/{}] | Loss: {:.4}", step, total_step, loss_val
                                )));
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx_ia.send(IAMessage::ResponseChat(format!("Erreur loss: {}", e)));
                        break;
                    }
                }

                let pourcentage = ((step as f32 / total_step as f32) * 100.0) as u16;
                if pourcentage != last_percent {
                    last_percent = pourcentage;
                    let _ = tx_ia.send(IAMessage::ProgressionEntainement(pourcentage));
                }
            }

            let _ = tx_ia.send(IAMessage::ResponseChat(
                "MIC_IA : Entraînement terminé ! Écris une question ou du code PHP à compléter.".to_string(),
            ));

            // 6. Boucle de génération
            while let Ok(cmd) = rx_cmd.recv() {
                let _ = tx_ia.send(IAMessage::ResponseChat(
                    "MIC_IA : Génération de la réponse en cours...".to_string(),
                ));

                match model.generate_response(&cmd, 80, 0.7f32, &dataset, &device, &tx_ia) {
                    Ok(generated_response) => {
                        let _ = tx_ia.send(IAMessage::ResponseChat(format!(
                            "MIC_IA : {}", generated_response
                        )));
                    }
                    Err(e) => {
                        let _ = tx_ia.send(IAMessage::ResponseChat(format!(
                            "MIC_IA : Erreur génération : {}\n", e
                        )));
                    }
                }
            }
        });
    }

    //----------------------------------INIT UI ET BOUCLE EVENT-------------------------//
    pub fn run() -> io::Result<()> {
        let (tx_ia, rx_ia) = mpsc::channel::<IAMessage>();
        let (tx_ia_cmd, rx_ia_cmd) = mpsc::channel::<String>();

        Self::training_simulation(tx_ia, rx_ia_cmd);

        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

        let mut app = App::new(rx_ia, tx_ia_cmd);
        let tick_rate = Duration::from_millis(100);
        let mut last_tick = Instant::now();

        loop {
            terminal.draw(|mut f| App::ui(&mut f, &app))?;

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Esc => break,
                            KeyCode::Left => {
                                if app.cursor_position > 0 {
                                    app.cursor_position -= 1;
                                }
                            }
                            KeyCode::Right => {
                                if app.cursor_position < app.input.chars().count() {
                                    app.cursor_position += 1;
                                }
                            }
                            KeyCode::Char(c) => {
                                let byte_index = app
                                    .input
                                    .char_indices()
                                    .nth(app.cursor_position)
                                    .map(|(i, _)| i)
                                    .unwrap_or_else(|| app.input.len());

                                app.input.insert(byte_index, c);
                                app.cursor_position += 1;
                            }
                            KeyCode::Backspace => {
                                if app.cursor_position > 0 {
                                    app.cursor_position -= 1;
                                    let byte_index = app
                                        .input
                                        .char_indices()
                                        .nth(app.cursor_position)
                                        .map(|(i, _)| i)
                                        .unwrap_or_else(|| app.input.len());
                                    app.input.remove(byte_index);
                                }
                            }
                            KeyCode::Up => {
                                app.scroll_offset = app.scroll_offset.saturating_sub(1);
                            }
                            KeyCode::Down => {
                                app.scroll_offset = app.scroll_offset.saturating_add(1);
                            }
                            KeyCode::Enter => {
                                if !app.input.is_empty() {
                                    let message = app.input.drain(..).collect::<String>();
                                    app.historique_chat.push(format!("Vous : {}", message));
                                    app.scroll_offset = app.historique_chat.len();
                                    let _ = app.tx_ia_cmd.send(message);
                                    app.cursor_position = 0;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Réception des messages de l'IA en temps réel
            while let Ok(msg) = app.rx_ia.try_recv() {
                match msg {
                    IAMessage::ProgressionEntainement(p) => {
                        app.progression = p;
                    }
                    IAMessage::ResponseChat(texte) => {
                        app.historique_chat.push(texte);
                        app.scroll_offset = app.historique_chat.len();
                    }
                    IAMessage::StreamChunk(chunk) => {
                        //Ajouter un token texte au dernier message
                        if let Some(last_msg) = app.historique_chat.last_mut() {
                            last_msg.push_str(&chunk);
                        }else{
                            app.historique_chat.push(format!("MIC_IA : {}", chunk));
                        }
                        app.scroll_offset = app.historique_chat.len();
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }
        }

        disable_raw_mode()?;
        stdout().execute(LeaveAlternateScreen)?;
        Ok(())
    }
}