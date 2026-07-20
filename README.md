

## Neural Network Architecture

#### Construction : cargo build --release
#### Cible : C:\AI_Training\target\release\Ai_Trainig.exe

## Comment fonctionne un LLM ?

#### L’entrainement initial : C’est la première étape, et la plus gourmande en données. 
#### Les LLM sont entraînés sur des milliards de mots et de phrases pour identifier des schémas et des relations. 
#### Durant cette phase, le LLM apprend à prédire le mot suivant dans une séquence en développant une compréhension statistique du langage. 
#### C’est le fondement de sa capacité à générer du texte cohérent.  
Le fine-tuning supervisé : Une fois entraîné sur une base de données générale, le modèle subit un réglage affiné. 
#### On lui donne des instructions plus spécifiques pour qu’il s’adapte à des tâches précises : répondre à des questions, suivre des instructions ou encore adopter un ton approprié. 
#### Cette étape est cruciale pour “spécialiser” le modèle et le rendre plus performant sur des cas d’usage précis.
#### L’apprentissage par renforcement à partir de retours humains (Reinforcement Learning Human Feedback) : Cette phase est essentielle pour aligner le modèle avec les attentes humaines. Des retours de personnes sont utilisés pour corriger les réponses du modèle. L’objectif est de le guider pour qu’il donne des réponses utiles, pertinentes et sans danger, tout en évitant les contenus faux, inappropriés ou agressifs.

### Input Text → Tokenizer → Embedding → Mean Pooling → Linear → Softmax → Prediction
                            ↓            ↓            ↓         ↓
                        [N, 16]        [16]         [2]       [2]

### Tokenizer -> Convertis des mots en nombre
### Embedding -> Carte des ID des mots pour densifier les vecteurs
### Mean pooling -> Moyenne des vecteurs pour fixé leurs tailles
### Linear -> Classification des calques
### SofMax -> Conversion en probabilités

## ----------------- APPRENTISSAGE -----------------------
### Concept Mathématique Deep Learning : Y = X.W + B
### $X$ est notre tenseur d'entrée (le texte encodé).
### $W$ représente les Poids (Weights) : une matrice de nombres aléatoires au début, qui va s'ajuster pour capter les règles du PHP.
### $B$ représente le Biais (Bias) : une valeur ajoutée pour donner plus de flexibilité aux neurones.

## --------------------TENSEURS-----------------------------

### On creer des batchs aléatoires (extrait des données texte decoupée)
### Un batch est un petit Vecteur
### Exemple : [4, 8, 97] 
### 4 : Le nombre de phrases que l'on traite en parallèle (batch_size).

### 8 : La longueur de la séquence de caractères lue par la machine (seq_len).

### 97 : La taille de ton vocabulaire (vocab_size). 
### Pour chaque caractère de chaque phrase, le réseau a sorti 97 "scores" 
### (un score pour chaque caractère possible du jeu de caractères PHP de ton fichier).

## ----Théorie : Loss, Rétropropagation et Descente de Gradient----

### 1. Forward Pass : Le modèle fait une prédiction.
### 2 .Calcul de la Loss (Perte) : On compare sa prédiction avec la réalité (le vrai caractère qui suit dans le cours PHP). Plus l'erreur est grande, plus la Loss est élevée. On utilise ici l'entropie croisée (Cross Entropy Loss).
### 3. Backward Pass (Rétropropagation) : Candle calcule la dérivée (le gradient) de la perte par rapport à chaque poids du réseau. Cela nous dit dans quelle direction ajuster chaque poids pour diminuer l'erreur.
### 4. Mise à jour des poids (Descente de gradient) : On modifie légèrement les poids dans la bonne direction à l'aide d'un taux d'apprentissage (learning rate $\eta$).

## --Comment fonctionne la génération ?--
### 1. On donne un texte de départ au modèle (ex: "class ").

### 2. On l'encode en nombres et on le passe dans notre modèle (forward).

### 3. Le modèle sort un vecteur de 97 scores (un pour chaque caractère possible).

### 4. On applique une fonction Softmax pour transformer ces scores en probabilités (entre 0 et 100%).

### 5. On pioche (échantillonne) le caractère suivant selon ces probabilités.

### 6. On ajoute ce nouveau caractère à notre texte et on recommence !

#### RNN -> Réseau de Neuronnes Récurent : Structure Gated Recurent Unit pour eviter l'oublis des mots

