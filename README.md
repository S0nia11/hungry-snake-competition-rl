# Hungry Snake Competition RL

Implémentation d'un agent **Double Dueling DQN** pour jouer à Snake en mode compétitif multi-agent.
L'agent apprend à survivre, manger et éliminer des adversaires dans une arène partagée, grâce à l'apprentissage par renforcement profond.

---

## Table des matières

1. [Objectif du projet](#objectif-du-projet)
2. [Structure des fichiers](#structure-des-fichiers)
3. [Description détaillée de chaque fichier](#description-détaillée-de-chaque-fichier)
4. [Architecture du réseau de neurones](#architecture-du-réseau-de-neurones)
5. [Environnement de jeu](#environnement-de-jeu)
6. [Hyperparamètres et variables modifiables](#hyperparamètres-et-variables-modifiables)
7. [Commandes de lancement](#commandes-de-lancement)
8. [Profils d'entraînement](#profils-dentraînement)
9. [Curriculum Learning](#curriculum-learning)
10. [Politiques de baseline](#politiques-de-baseline)
11. [Score d'évaluation composite](#score-dévaluation-composite)
12. [Sorties générées](#sorties-générées)
13. [Dépendances](#dépendances)

---

## Objectif du projet

Ce projet entraîne un agent de reinforcement learning à jouer au **Snake en mode compétitif** :
- L'agent (snake `P`) affronte `n_bots` adversaires contrôlés par une IA heuristique.
- L'objectif est de **maximiser son score** en mangeant de la nourriture tout en **survivant le plus longtemps possible** et en **éliminant les adversaires**.
- L'algorithme utilisé est le **Double Dueling DQN** avec soft target update et curriculum learning optionnel.

---

## Structure des fichiers

```
hungry-snake-competition-rl/
│
├── snake_env.py                  # Environnement Snake multi-agent (Gym-compatible)
├── dqn_agent.py                  # Agent Double Dueling DQN + Replay Buffer
├── train_dqn.py                  # Script d'entraînement principal
├── evaluate_dqn.py               # Évaluation d'un modèle pré-entraîné
├── baseline_policies.py          # Politiques de référence (random, greedy, heuristic…)
├── benchmark_baselines.py        # Comparaison agent DQN vs baselines
├── snake_ui.py                   # Interface graphique Tkinter
├── train_profiles.bat            # Entraînement de 4 profils en séquentiel
│
├── outputs_v3/                   # Sortie par défaut (train_dqn.py sans --output-dir)
├── outputs_curriculum/           # Profil curriculum (train_profiles.bat)
├── outputs_agressif/             # Profil agressif
├── outputs_conservateur/         # Profil conservateur
└── outputs_explorateur/          # Profil explorateur (réseau 512)
    ├── models/
    │   ├── best_eval_model.pt    # Meilleur modèle selon le score composite
    │   ├── last_model.pt         # Modèle à la fin de l'entraînement
    │   └── checkpoint_ep*.pt     # Checkpoints intermédiaires
    ├── logs/
    │   └── training_history.json # Historique complet des métriques
    └── plots/
        ├── rewards.png           # Courbe de reward
        ├── scores.png            # Courbe de score
        ├── win_rate.png          # Taux de victoire
        └── rank.png              # Rang moyen
```

---

## Description détaillée de chaque fichier

### `snake_env.py` — Environnement de jeu

Contient toute la logique du jeu Snake multi-agent. C'est le seul fichier qui gère les règles.

**Classes principales :**

| Classe | Rôle |
|--------|------|
| `FoodType` | Enum des 3 types de nourriture : `NORMAL`, `BONUS`, `RISKY` |
| `FoodSpec` | Dataclass définissant score, reward et croissance pour chaque type de nourriture |
| `Snake` | Dataclass représentant un snake (corps, direction, score, état vivant/mort) |
| `MultiSnakeEnv` | Environnement principal compatible API Gym (reset/step/render) |

**Méthodes clés de `MultiSnakeEnv` :**

| Méthode | Description |
|---------|-------------|
| `reset(seed)` | Réinitialise l'arène, respawn les snakes et la nourriture |
| `step(action)` | Avance d'un step : déplace tous les snakes, gère collisions et récompenses |
| `get_observation()` | Retourne le vecteur d'état 23D du joueur |
| `render()` | Retourne une représentation ASCII de l'arène |
| `_bot_action(snake)` | Heuristique utilisée par les bots adversaires (cherche la nourriture la plus proche) |
| `_safe_actions(snake)` | Retourne la liste des actions qui n'entraînent pas de mort immédiate |
| `_observation_for_snake(snake_id)` | Construit le vecteur d'état pour un snake donné |

**Variables modifiables dans `snake_env.py` :**

| Variable | Valeur par défaut | Description |
|----------|------------------|-------------|
| `FOOD_SPECS[NORMAL]` | score=10, reward=10.0, growth=1 | Propriétés de la nourriture normale |
| `FOOD_SPECS[BONUS]` | score=18, reward=15.0, growth=1 | Propriétés de la nourriture bonus |
| `FOOD_SPECS[RISKY]` | score=25, reward=20.0, growth=2 | Propriétés de la nourriture risquée |
| `food_counts` | NORMAL=2, BONUS=1, RISKY=1 | Nombre de chaque type de nourriture sur l'arène |
| `last_player_positions.maxlen` | 8 | Fenêtre pour détecter les boucles |
| `render_fps` | 5 | FPS pour le rendu (non utilisé en mode texte) |

---

### `dqn_agent.py` — Agent DQN

Contient la définition du réseau de neurones et la logique d'apprentissage.

**Classes :**

| Classe | Rôle |
|--------|------|
| `DuelingQNetwork` | Réseau Dueling DQN (tronc commun + Value stream + Advantage stream) |
| `Transition` | Dataclass stockant un tuple (state, action, reward, next_state, done) |
| `ReplayBuffer` | Buffer circulaire de replay d'expériences |
| `DQNAgent` | Agent complet : act, remember, update, save, load |

**Méthodes clés de `DQNAgent` :**

| Méthode | Description |
|---------|-------------|
| `act(state, greedy)` | Choisit une action (epsilon-greedy si greedy=False) |
| `remember(...)` | Ajoute une transition au replay buffer |
| `update()` | Effectue un pas d'apprentissage (sample + loss + backprop + soft update) |
| `decay_epsilon()` | Multiplie epsilon par epsilon_decay |
| `save(path)` | Sauvegarde les poids de q_net dans un fichier .pt |
| `load(path)` | Charge les poids depuis un fichier .pt |

**Les 2 réseaux de neurones :**

| Réseau | Variable | Rôle |
|--------|----------|------|
| Q-network principal | `self.q_net` | Entraîné par backpropagation à chaque step |
| Q-network cible | `self.target_net` | Mis à jour lentement (soft update, τ=0.01) pour stabiliser l'apprentissage |

---

### `train_dqn.py` — Entraînement

Script principal d'entraînement. Orchestre la boucle épisodique, les évaluations périodiques, les checkpoints et les graphiques.

**Fonctions :**

| Fonction | Description |
|----------|-------------|
| `train(args)` | Boucle d'entraînement principale : épisodes, curriculum, évaluation, sauvegarde |
| `evaluate_agent(agent, args, episodes)` | Évalue l'agent en mode greedy sur N épisodes |
| `build_env(args)` | Instancie `MultiSnakeEnv` depuis les arguments CLI |
| `plot_training_curves(history, plots_dir)` | Génère les 4 graphiques PNG |
| `moving_average(values, window)` | Calcule la moyenne glissante |

**Score composite pour sauvegarder le meilleur modèle :**
```
score = 120 × win_rate + 20 × draw_rate + 0.35 × score_moyen + 0.10 × reward_moyen - 2 × rang_moyen
```

---

### `evaluate_dqn.py` — Évaluation

Charge un modèle `.pt` et l'évalue sur N épisodes. Affiche reward moyenne, score, win rate, draw rate et rang.

**Arguments CLI :**

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--model-path` | **requis** | Chemin vers le fichier `.pt` |
| `--episodes` | 50 | Nombre d'épisodes d'évaluation |
| `--render` | False | Affiche chaque step en ASCII dans le terminal |
| `--n-bots` | 2 | Nombre de bots adversaires |
| `--hidden-dim` | 256 | Taille des couches cachées (doit correspondre au modèle) |
| `--width` / `--height` | 15 | Dimensions de l'arène |
| `--max-steps` | 300 | Steps max par épisode |
| `--device` | auto | `cpu` ou `cuda` |
| `--seed` | 42 | Graine aléatoire |

---

### `baseline_policies.py` — Politiques de référence

Définit 5 politiques utilisables comme joueur ou comme bots adversaires dans le benchmark.

| Politique | Type | Description |
|-----------|------|-------------|
| `random` | Règles | Action aléatoire parmi {0, 1, 2} |
| `safe_random` | Règles | Aléatoire parmi les actions non-suicidaires |
| `greedy` | Règles | Maximise une fonction de valeur manuelle (nourriture, murs, ennemis) |
| `heuristic` | Règles | Cherche la nourriture bonus la plus proche en évitant les collisions |
| `model` | **Entraîné** | Agent DQN chargé depuis un fichier `.pt` |

**Variables modifiables dans `greedy_food_policy` :**

| Variable (dans `_position_value`) | Valeur | Description |
|-----------------------------------|--------|-------------|
| Poids food (`food_weight`) | `score + 2 × reward` | Importance de la nourriture selon son type |
| Facteur RISKY si longueur < 4 | `× 0.8` | Pénalise la nourriture risquée pour les petits snakes |
| `enemy_penalty` (dist ≤ 1) | 4.0 | Forte pénalité si tête ennemie adjacente |
| `enemy_penalty` (dist == 2) | 1.5 | Pénalité modérée si ennemi à 2 cases |
| `wall_bonus` | `0.3 × margin` | Légère préférence pour le centre |

---

### `benchmark_baselines.py` — Benchmark comparatif

Compare toutes les politiques sur N épisodes et sauvegarde les résultats en JSON.

**Arguments CLI :**

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--model-path` | `` (vide) | Chemin modèle DQN (optionnel, skip `model` si absent) |
| `--episodes` | 100 | Épisodes par politique |
| `--policies` | all 5 | Liste des politiques joueur à tester |
| `--bot-policy` | `heuristic` | Politique des bots adversaires |
| `--output-json` | `benchmark_results.json` | Fichier de sortie |
| `--n-bots` | 2 | Nombre de bots adversaires |
| `--hidden-dim` | 256 | Taille des couches cachées du modèle |
| `--seed` | 42 | Graine aléatoire |

**Métriques produites par politique :**
- `reward` : reward totale moyenne
- `score` : score de jeu moyen
- `win_rate` : taux de victoire
- `draw_rate` : taux de match nul
- `avg_rank` : rang moyen (1 = meilleur)
- `avg_steps` : durée moyenne des épisodes

---

### `snake_ui.py` — Interface graphique

Interface Tkinter qui permet de visualiser les parties en temps réel.

**Fonctionnalités :**
- Sélecteur de modèle `.pt` (scan automatique de tous les dossiers `outputs_*/`)
- Bouton **Rafraîchir les modèles** pour mettre à jour après un entraînement
- Contrôle de la vitesse (FPS slider)
- Choix du contrôleur pour chaque snake : `human`, `random`, `safe_random`, `greedy`, `heuristic`, `model`
- Mode humain : touches fléchées ou ZQSD
- Affichage en temps réel : scores, rangs, outcomes (`VICTOIRE`, `DÉFAITE`, `MATCH NUL`)

**Variables de couleur modifiables (en haut du fichier) :**

| Variable | Valeur | Description |
|----------|--------|-------------|
| `CELL_SIZE` | 30 | Taille en pixels de chaque case |
| `PADDING` | 3 | Marge intérieure des cellules |
| `WINDOW_BG` | `#070d1a` | Couleur de fond de la fenêtre |
| `SNAKE_COLORS` | liste de 8 couleurs | Couleurs des snakes (par index) |
| `FOOD_COLORS` | dict par type | Vert=normal, Jaune=bonus, Rouge=risky |
| `DEAD_COLOR` | `#1e2d3d` | Couleur d'un snake mort |

**Arguments CLI :**

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--model-path` | `` | Modèle pré-sélectionné au lancement |
| `--width` / `--height` | 15 | Dimensions de l'arène |
| `--snakes` | 3 | Nombre total de snakes (joueur + bots) |

---

### `train_profiles.bat` — Batch d'entraînement

Lance 4 entraînements séquentiels avec des hyperparamètres différents.

| Profil | Dossier | Épisodes | Particularités |
|--------|---------|----------|----------------|
| **curriculum** | `outputs_curriculum` | 2000 | `--curriculum`, 3 bots, progression graduelle |
| **agressif** | `outputs_agressif` | 2000 | `win_bonus=60`, `elimination_bonus=5`, `death_penalty=-10` |
| **conservateur** | `outputs_conservateur` | 2000 | `survival_reward=0.08`, `death_penalty=-30`, `win_bonus=20` |
| **explorateur** | `outputs_explorateur` | 2500 | `hidden_dim=512`, `buffer=200k`, `batch=256`, exploration lente |

---

## Architecture du réseau de neurones

```
Input : vecteur 23D (état du jeu)
   │
   └─► Feature extractor
         Linear(23 → 256) → ReLU
         Linear(256 → 256) → ReLU
         │
         ├─► Value stream
         │     Linear(256 → 128) → ReLU
         │     Linear(128 → 1)          → V(s)
         │
         └─► Advantage stream
               Linear(256 → 128) → ReLU
               Linear(128 → 3)          → A(s, a)
                     │
                     └─► Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

**Techniques de stabilisation :**

| Technique | Valeur | Effet |
|-----------|--------|-------|
| Double DQN | activé | Réduit la surestimation des Q-values |
| Soft target update | τ = 0.01 | Mise à jour lente du target net à chaque step |
| CosineAnnealingLR | eta_min = lr × 0.01 | Décroissance douce du learning rate |
| SmoothL1Loss (Huber) | — | Loss robuste aux valeurs aberrantes |

---

## Environnement de jeu

### Règles

- Arène de `width × height` cases (défaut : 15×15)
- 1 agent RL (snake `P`) vs `n_bots` bots adversaires (défaut : 2)
- 3 types de nourriture présents en permanence sur l'arène

### Types de nourriture

| Type | Symbole | Points | Reward | Croissance | Caractère |
|------|---------|--------|--------|-----------|-----------|
| Normal | `n` | 10 | +10.0 | +1 | Courant, sans risque |
| Bonus | `b` | 18 | +15.0 | +1 | Meilleur ratio risque/reward |
| Risky | `r` | 25 | +20.0 | +2 | Fort reward mais fait beaucoup grossir |

### Actions possibles

| Code | Action |
|------|--------|
| `0` | Tout droit |
| `1` | Tourner à gauche (par rapport à la direction courante) |
| `2` | Tourner à droite (par rapport à la direction courante) |

### Causes de mort

| Cause | Description |
|-------|-------------|
| `wall` | Collision avec un mur |
| `body` | Collision avec un corps de snake |
| `head_on` | Deux têtes arrivent sur la même case |
| `swap` | Deux snakes s'échangent de place |

### Vecteur d'observation (23 dimensions)

```
Index  0     : danger tout droit (0 ou 1)
Index  1     : danger à gauche   (0 ou 1)
Index  2     : danger à droite   (0 ou 1)
Index  3–6   : direction courante (one-hot : haut, droite, bas, gauche)
Index  7–8   : dx/dy vers la nourriture NORMAL la plus proche (normalisé)
Index  9–10  : dx/dy vers la nourriture BONUS la plus proche (normalisé)
Index 11–12  : dx/dy vers la nourriture RISKY la plus proche (normalisé)
Index 13–16  : distances aux 4 murs (normalisé : haut, droite, bas, gauche)
Index 17–18  : dx/dy vers la tête ennemie la plus proche (normalisé)
Index 19     : distance à la tête ennemie la plus proche (normalisé)
Index 20     : longueur du snake (normalisé par width×height)
Index 21     : score du joueur (divisé par 100)
Index 22     : ratio de snakes encore en vie
```

### Système de récompenses

| Événement | Paramètre | Défaut train | Description |
|-----------|-----------|-------------|-------------|
| Survie | `survival_reward` | +0.03 | Donné à chaque step où le joueur est vivant |
| Mort | `death_penalty` | -20.0 | Appliqué quand le joueur meurt |
| Victoire (élimination) | `win_bonus` | +35.0 | Tous les bots éliminés |
| Victoire (timeout) | `timeout_win_bonus` | +18.0 | Meilleur score au timeout |
| Match nul | `draw_bonus` | +4.0 | Égalité de score au timeout |
| Défaite (timeout) | `timeout_loss_penalty` | -3.0 | Moins bon score au timeout |
| Élimination d'un bot | `elimination_bonus` | +1.5 | Par bot tué |
| Boucle détectée | `loop_penalty` | -1.5 | Si ≤ 3 positions uniques sur les 8 dernières |
| Nourriture mangée | `FoodSpec.reward` | +10/+15/+20 | Selon le type de nourriture |

> **Note :** Les valeurs par défaut de `snake_env.py` diffèrent légèrement des valeurs CLI de `train_dqn.py`. Les valeurs CLI (indiquées ci-dessus) prévalent lors de l'entraînement.

---

## Hyperparamètres et variables modifiables

### Réseau de neurones

| Paramètre CLI | Défaut | Description |
|---------------|--------|-------------|
| `--hidden-dim` | 256 | Taille des couches cachées (profil explorateur : 512) |

### Apprentissage

| Paramètre CLI | Défaut | Description |
|---------------|--------|-------------|
| `--lr` | 3e-4 | Learning rate initial (Adam) |
| `--gamma` | 0.99 | Facteur de discount (importance du futur) |
| `--batch-size` | 128 | Taille des mini-batchs pour la mise à jour |
| `--buffer-capacity` | 100 000 | Capacité maximale du replay buffer |
| `--tau` | 0.01 | Coefficient de soft update du target network |
| `--target-update-freq` | 1 | Fréquence de mise à jour du target net (en steps) |
| `--learning-starts` | 1 000 | Nombre de transitions avant de commencer l'apprentissage |
| `--disable-double-dqn` | False | Désactiver le Double DQN |

### Exploration (epsilon-greedy)

| Paramètre CLI | Défaut | Description |
|---------------|--------|-------------|
| `--epsilon-start` | 1.0 | Epsilon initial (100% exploration aléatoire) |
| `--epsilon-end` | 0.05 | Epsilon minimal (5% exploration résiduelle) |
| `--epsilon-decay` | 0.998 | Facteur multiplicatif appliqué à epsilon après chaque épisode |

> **Exemple :** avec decay=0.998 sur 1500 épisodes : epsilon passe de 1.0 à ~0.05 vers l'épisode 1400.

### Environnement

| Paramètre CLI | Défaut | Description |
|---------------|--------|-------------|
| `--width` / `--height` | 15 | Dimensions de l'arène |
| `--n-bots` | 2 | Nombre de bots adversaires |
| `--max-steps` | 300 | Steps maximum par épisode avant truncation |
| `--seed` | 42 | Graine aléatoire (reproductibilité) |

### Entraînement

| Paramètre CLI | Défaut | Description |
|---------------|--------|-------------|
| `--episodes` | 1500 | Nombre total d'épisodes |
| `--eval-every` | 100 | Évaluer le modèle tous les N épisodes |
| `--eval-episodes` | 20 | Nombre d'épisodes par session d'évaluation |
| `--save-every` | 500 | Sauvegarder un checkpoint tous les N épisodes (0 = désactivé) |
| `--log-every` | 50 | Afficher les logs tous les N épisodes |
| `--ma-window` | 50 | Fenêtre de la moyenne glissante dans les logs |
| `--output-dir` | `outputs_v3` | Dossier de sortie (models, logs, plots) |

### Récompenses (personnalisables via CLI)

| Paramètre CLI | Défaut | Description |
|---------------|--------|-------------|
| `--survival-reward` | 0.03 | Récompense par step survécu |
| `--death-penalty` | -20.0 | Pénalité à la mort |
| `--win-bonus` | 35.0 | Bonus de victoire par élimination |
| `--timeout-win-bonus` | 18.0 | Bonus de victoire au timeout |
| `--draw-bonus` | 4.0 | Bonus de match nul |
| `--timeout-loss-penalty` | -3.0 | Pénalité de défaite au timeout |
| `--elimination-bonus` | 1.5 | Bonus par bot éliminé |
| `--loop-penalty` | -1.5 | Pénalité pour boucler |

### Curriculum

| Paramètre CLI | Défaut | Description |
|---------------|--------|-------------|
| `--curriculum` | False | Activer le curriculum learning |
| `--curriculum-stage-1` | 500 | Dernier épisode de la phase 1 (1 bot) |
| `--curriculum-stage-2` | 1000 | Dernier épisode de la phase 2 (2 bots) |

---

## Commandes de lancement

### Installation des dépendances

```bash
pip install torch numpy matplotlib
```

### Entraînement simple (défaut)

```powershell
python train_dqn.py
```

Sortie dans `outputs_v3/`.

### Entraînement avec logging en direct

```powershell
python -u train_dqn.py --episodes 1500 --curriculum | Tee-Object -FilePath outputs_v3/logs/train_live.log
```

Suivre les logs dans un second terminal :

```powershell
Get-Content outputs_v3/logs/train_live.log -Wait -Tail 20
```

### Entraînement curriculum (recommandé)

```powershell
python train_dqn.py --episodes 2000 --curriculum --n-bots 3
```

### Entraînement avec hyperparamètres personnalisés

```powershell
python train_dqn.py `
  --episodes 3000 `
  --curriculum `
  --curriculum-stage-1 800 `
  --curriculum-stage-2 1800 `
  --n-bots 3 `
  --hidden-dim 256 `
  --lr 3e-4 `
  --gamma 0.99 `
  --epsilon-decay 0.9985 `
  --win-bonus 40.0 `
  --death-penalty -25.0 `
  --output-dir outputs_custom `
  --seed 42
```

### Entraînement des 4 profils en batch

```powershell
.\train_profiles.bat
```

Génère 4 modèles :
- `outputs_curriculum\models\best_eval_model.pt`
- `outputs_agressif\models\best_eval_model.pt`
- `outputs_conservateur\models\best_eval_model.pt`
- `outputs_explorateur\models\best_eval_model.pt`

---

### Évaluation d'un modèle

```powershell
# Évaluation simple
python evaluate_dqn.py --model-path outputs_v3\models\best_eval_model.pt --episodes 100

# Avec rendu ASCII dans le terminal
python evaluate_dqn.py --model-path outputs_v3\models\best_eval_model.pt --render --episodes 5

# Profil explorateur (hidden-dim 512 obligatoire)
python evaluate_dqn.py --model-path outputs_explorateur\models\best_eval_model.pt --hidden-dim 512 --episodes 100

# Tous les profils
python evaluate_dqn.py --model-path outputs_curriculum\models\best_eval_model.pt    --episodes 100
python evaluate_dqn.py --model-path outputs_agressif\models\best_eval_model.pt      --episodes 100
python evaluate_dqn.py --model-path outputs_conservateur\models\best_eval_model.pt  --episodes 100
python evaluate_dqn.py --model-path outputs_explorateur\models\best_eval_model.pt   --hidden-dim 512 --episodes 100
```

---

### Benchmark vs baselines

```powershell
# Toutes les politiques contre des bots heuristiques
python benchmark_baselines.py --model-path outputs_v3\models\best_eval_model.pt --episodes 200

# Bots aléatoires sécurisés pour tester en conditions plus faciles
python benchmark_baselines.py --model-path outputs_v3\models\best_eval_model.pt --bot-policy safe_random --episodes 200

# Sans modèle (compare uniquement les baselines entre elles)
python benchmark_baselines.py --episodes 200
```

Résultats sauvegardés dans `benchmark_results.json`.

---

### Interface graphique

```powershell
# Lancement standard
python snake_ui.py

# Avec modèle pré-sélectionné
python snake_ui.py --model-path outputs_v3\models\best_eval_model.pt

# Arène plus grande avec plus de snakes
python snake_ui.py --width 20 --height 20 --snakes 4
```

---

### Consulter les résultats

```powershell
# Dernières métriques du training
Get-Content outputs_v3\logs\training_history.json | python -m json.tool | Select-String "win_rate|score"

# Ouvrir les graphiques
Invoke-Item outputs_v3\plots\win_rate.png
Invoke-Item outputs_v3\plots\rewards.png
Invoke-Item outputs_v3\plots\scores.png
Invoke-Item outputs_v3\plots\rank.png
```

---

## Profils d'entraînement

### Phase 1 — Curriculum (`outputs_curriculum`)

Progression graduelle en difficulté : commence seul contre 1 bot, puis monte à 3.

```
Episodes 1–600   : 1 bot adversaire
Episodes 601–1200: 2 bots adversaires
Episodes 1201–2000: 3 bots adversaires
```

Paramètres : `lr=3e-4`, `win_bonus=35`, `death_penalty=-20`, `survival_reward=0.03`

---

### Phase 2 — Agressif (`outputs_agressif`)

Favorise les comportements offensifs : tuer les ennemis, gagner coûte que coûte.

Paramètres clés : `win_bonus=60`, `elimination_bonus=5`, `death_penalty=-10`, `survival_reward=0.01`

---

### Phase 3 — Conservateur (`outputs_conservateur`)

Favorise la survie avant tout, évite les risques inutiles.

Paramètres clés : `survival_reward=0.08`, `death_penalty=-30`, `win_bonus=20`, `elimination_bonus=0.5`

---

### Phase 4 — Explorateur (`outputs_explorateur`)

Réseau plus large, exploration lente, plus de capacité de mémoire. Le plus long à entraîner.

Paramètres clés : `hidden_dim=512`, `buffer_capacity=200000`, `batch_size=256`, `episodes=2500`, `epsilon_decay=0.9992`

> **Important :** Ce profil utilise `hidden_dim=512`. Il faut obligatoirement passer `--hidden-dim 512` lors de l'évaluation ou du chargement dans l'UI.

---

## Curriculum Learning

Avec `--curriculum`, l'entraînement progresse automatiquement en 3 phases :

| Phase | Épisodes | Adversaires | Objectif |
|-------|----------|-------------|----------|
| 1 | 1 → `stage_1` (défaut 500) | 1 bot | Apprendre les bases : manger, survivre |
| 2 | `stage_1+1` → `stage_2` (défaut 1000) | 2 bots | Gérer plusieurs adversaires |
| 3 | `stage_2+1` → fin | `n_bots` bots | Compétition complète |

Le passage de phase est affiché dans les logs :
```
[Curriculum] Episode 501: passage à 2 bot(s)
[Curriculum] Episode 1001: passage à 3 bot(s)
```

---

## Politiques de baseline

| Politique | Comportement | Usage typique |
|-----------|-------------|---------------|
| `random` | Action aléatoire parmi {0,1,2} | Borne inférieure de performance |
| `safe_random` | Aléatoire parmi les coups non-suicidaires | Baseline minimale intelligente |
| `greedy` | Maximise une fonction de valeur manuelle (nourriture, murs, ennemis) | Benchmark intermédiaire |
| `heuristic` | Cherche la nourriture bonus/normale la plus proche en sécurité | Bot par défaut des adversaires |
| `model` | Agent DQN chargé depuis `.pt`, mode greedy | Mesure la performance du modèle entraîné |

Ces politiques sont utilisables à la fois comme **joueur** et comme **bots adversaires** dans le benchmark.

---

## Score d'évaluation composite

Le script `train_dqn.py` calcule ce score après chaque évaluation périodique pour décider quel modèle sauvegarder comme `best_eval_model.pt` :

```
score_composite = 120 × win_rate
                +  20 × draw_rate
                + 0.35 × score_moyen
                + 0.10 × reward_moyenne
                -  2.0 × rang_moyen
```

**Logique :** La victoire (win_rate) est très fortement valorisée. Le rang moyen est pénalisé pour éviter de sauvegarder un modèle qui score bien mais perd souvent.

---

## Sorties générées

Après un entraînement, le dossier de sortie contient :

```
outputs_*/
├── models/
│   ├── best_eval_model.pt        # Meilleur modèle selon le score composite
│   ├── last_model.pt             # Modèle à la fin de l'entraînement
│   └── checkpoint_ep500.pt       # Checkpoints intermédiaires (si --save-every > 0)
│
├── logs/
│   └── training_history.json     # Toutes les métriques épisode par épisode
│
└── plots/
    ├── rewards.png               # Courbe reward train + eval
    ├── scores.png                # Courbe score train + eval
    ├── win_rate.png              # Taux de victoire train + eval
    └── rank.png                  # Rang moyen train + eval
```

**Contenu de `training_history.json` :**

| Clé | Description |
|-----|-------------|
| `episode_rewards` | Reward totale par épisode |
| `episode_scores` | Score de jeu par épisode |
| `episode_lengths` | Durée en steps par épisode |
| `episode_wins` | 1.0 si victoire, 0.0 sinon |
| `episode_draws` | 1.0 si match nul, 0.0 sinon |
| `episode_ranks` | Rang du joueur par épisode |
| `losses` | Valeur de la loss à chaque mise à jour |
| `reward_ma` | Moyenne glissante de la reward |
| `score_ma` | Moyenne glissante du score |
| `win_rate_ma` | Moyenne glissante du win rate |
| `eval_checkpoints` | Épisodes où une évaluation a eu lieu |
| `eval_win_rates` | Win rate lors de chaque évaluation |
| `config` | Tous les arguments CLI utilisés |

---

## Dépendances

```bash
pip install torch numpy matplotlib
```

| Package | Utilisation |
|---------|-------------|
| `torch` | Réseau de neurones, optimiseur, tenseurs |
| `numpy` | Vecteurs d'état, replay buffer |
| `matplotlib` | Génération des graphiques de courbes |
| `tkinter` | Interface graphique (inclus avec Python) |
