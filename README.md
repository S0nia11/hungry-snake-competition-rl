# Hungry Snake Competition RL

Implémentation d'un agent **Double Dueling DQN** pour jouer à Snake en mode compétitif multi-agent.

---

## Architecture du projet

```
hungry-snake-competition-rl/
├── snake_env.py            # Environnement Snake multi-agent (Gym-compatible)
├── dqn_agent.py            # Agent Double Dueling DQN + N-Step Replay Buffer
├── train_dqn.py            # Entraînement avec curriculum, LR scheduler, checkpoints
├── evaluate_dqn.py         # Évaluation d'un modèle pré-entraîné
├── baseline_policies.py    # Politiques de référence (random, greedy, heuristic...)
├── benchmark_baselines.py  # Benchmark comparatif agent vs baselines
├── snake_ui.py             # Interface graphique Tkinter avec model picker
├── train_profiles.bat      # Entraînement de 4 profils en séquentiel
├── outputs_v3/             # Sorties d'entraînement par défaut
├── outputs_curriculum/     # Profil curriculum (train_profiles.bat)
├── outputs_agressif/       # Profil agressif
├── outputs_conservateur/   # Profil conservateur
└── outputs_explorateur/    # Profil explorateur (réseau 512)
    ├── models/
    │   ├── best_eval_model.pt
    │   ├── last_model.pt
    │   └── checkpoint_ep*.pt
    ├── logs/
    │   └── training_history.json
    └── plots/
        ├── rewards.png
        ├── scores.png
        ├── win_rate.png
        └── rank.png
```

---

## Environnement

### Règles du jeu
- Arène de `width × height` cases (défaut 15×15)
- 1 joueur RL vs `n_bots` agents IA (défaut 2)
- 3 types de nourriture sur le terrain :

| Type | Points | Reward | Croissance |
|------|--------|--------|-----------|
| Normal (n) | 10 | +10.0 | +1 |
| Bonus (b) | 18 | +15.0 | +1 |
| Risky (r) | 25 | +20.0 | +2 |

### Actions
| Code | Action |
|------|--------|
| 0 | Tout droit |
| 1 | Tourner à gauche |
| 2 | Tourner à droite |

### Observation (vecteur 23D)
```
[danger_devant, danger_gauche, danger_droite,      # 3 — dangers immédiats
 dir_haut, dir_droite, dir_bas, dir_gauche,         # 4 — direction courante (one-hot)
 nourriture_normale_dx/dy,                          # 2
 nourriture_bonus_dx/dy,                            # 2
 nourriture_risky_dx/dy,                            # 2
 mur_haut, mur_droite, mur_bas, mur_gauche,         # 4 — distance aux murs
 ennemi_dx, ennemi_dy, ennemi_dist,                 # 3 — ennemi le plus proche
 longueur, score, nb_snakes_vivants]                # 3
```

### Récompenses configurables
| Paramètre | Défaut | Déclencheur |
|-----------|--------|-------------|
| `survival_reward` | +0.03 | Chaque step vivant |
| `death_penalty` | -20.0 | Collision/mort |
| `win_bonus` | +35.0 | Victoire par élimination |
| `timeout_win_bonus` | +18.0 | Victoire au timeout |
| `draw_bonus` | +4.0 | Match nul |
| `timeout_loss_penalty` | -3.0 | Défaite au timeout |
| `elimination_bonus` | +1.5 | Par bot éliminé |
| `loop_penalty` | -1.5 | Boucle détectée |

---

## Agent

Architecture **Dueling DQN** avec **Double DQN**, **N-Step Returns** et **Soft Target Update** :

```
Input(23)
   └─► Feature: Linear(256) → ReLU → Dropout → Linear(256) → ReLU → Dropout
          ├─► Value stream:     Linear(128) → ReLU → Linear(1)
          └─► Advantage stream: Linear(128) → ReLU → Linear(3)
                   └─► Q(s,a) = V(s) + A(s,a) - mean(A)
```

### Techniques de régularisation
| Technique | Valeur par défaut | Effet |
|-----------|------------------|-------|
| Dropout | 0.1 | Régularisation dans le feature extractor |
| Weight Decay (Adam) | 1e-4 | Pénalisation L2 des poids |
| Grad Clip | 1.0 | Stabilité des mises à jour |
| N-Step Returns | n=3 | Réduction variance, propagation signal plus rapide |
| CosineAnnealingLR | eta_min = lr × 0.01 | Décroissance douce du learning rate |

---

## Workflow (PowerShell)

### 0. Installation
```powershell
pip install torch numpy matplotlib
```

### 1. Entraînement simple
```powershell
# Lancer l'entraînement et enregistrer les logs
python -u train_dqn.py --episodes 1500 --curriculum | Tee-Object -FilePath outputs_v3/logs/train_live.log
```

Suivre les logs en direct dans un second terminal :
```powershell
Get-Content outputs_v3/logs/train_live.log -Wait -Tail 20
```

### 2. Entraînement de 4 profils (batch)
```powershell
.\train_profiles.bat
```

Génère 4 modèles dans leurs dossiers respectifs :
- `outputs_curriculum\models\best_eval_model.pt`
- `outputs_agressif\models\best_eval_model.pt`
- `outputs_conservateur\models\best_eval_model.pt`
- `outputs_explorateur\models\best_eval_model.pt`

### 3. Évaluation
```powershell
# Modèle v3 (défaut)
python evaluate_dqn.py --model-path outputs_v3\models\best_eval_model.pt --episodes 100

# Profils batch
python evaluate_dqn.py --model-path outputs_curriculum\models\best_eval_model.pt --episodes 100
python evaluate_dqn.py --model-path outputs_agressif\models\best_eval_model.pt --episodes 100
python evaluate_dqn.py --model-path outputs_conservateur\models\best_eval_model.pt --episodes 100
python evaluate_dqn.py --model-path outputs_explorateur\models\best_eval_model.pt --hidden-dim 512 --episodes 100
```

> **Note :** Le profil `explorateur` utilise `--hidden-dim 512`, il faut le préciser à l'évaluation.

Avec affichage ASCII pas à pas :
```powershell
python evaluate_dqn.py --model-path outputs_v3\models\best_eval_model.pt --render --episodes 5
```

### 4. Benchmark vs baselines
```powershell
python benchmark_baselines.py --model-path outputs_v3\models\best_eval_model.pt --episodes 200
```

Résultats sauvegardés dans `benchmark_results.json`.

### 5. Interface graphique
```powershell
python snake_ui.py
```

L'UI scanne automatiquement tous les `.pt` du projet. Le bouton **↺ Rafraîchir les modèles** met à jour la liste après un entraînement.

### 6. Consulter les résultats d'entraînement
```powershell
# Voir les dernières lignes du log
Get-Content outputs_v3\logs\training_history.json | python -m json.tool | Select-String "win_rate|score"

# Ouvrir les graphiques
Invoke-Item outputs_v3\plots\win_rate.png
Invoke-Item outputs_v3\plots\rewards.png
Invoke-Item outputs_v3\plots\scores.png
```

---

## Profils d'entraînement (`train_profiles.bat`)

| Profil | Dossier | Stratégie |
|--------|---------|-----------|
| **curriculum** | `outputs_curriculum` | Progression graduelle 1→2→3 bots, 2000 eps |
| **agressif** | `outputs_agressif` | win_bonus=60, elimination_bonus=5, death_penalty=-10 |
| **conservateur** | `outputs_conservateur` | survival_reward=0.08, death_penalty=-30, win_bonus=20 |
| **explorateur** | `outputs_explorateur` | hidden_dim=512, buffer=200k, 2500 eps |

---

## Arguments CLI

### `train_dqn.py`

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--episodes` | 1500 | Nombre total d'épisodes |
| `--width` / `--height` | 15 | Dimensions de l'arène |
| `--n-bots` | 2 | Nombre de bots adversaires |
| `--max-steps` | 300 | Steps max par épisode |
| `--hidden-dim` | 256 | Taille des couches cachées |
| `--gamma` | 0.99 | Facteur d'escompte |
| `--lr` | 3e-4 | Learning rate |
| `--weight-decay` | 1e-4 | Régularisation L2 (Adam) |
| `--dropout` | 0.1 | Dropout dans le réseau |
| `--grad-clip` | 1.0 | Clipping du gradient |
| `--n-step` | 3 | N-step returns |
| `--batch-size` | 128 | Taille du batch |
| `--buffer-capacity` | 100000 | Taille du replay buffer |
| `--tau` | 0.01 | Coefficient soft update |
| `--epsilon-start` | 1.0 | Exploration initiale |
| `--epsilon-end` | 0.05 | Exploration minimale |
| `--epsilon-decay` | 0.998 | Décroissance d'epsilon |
| `--learning-starts` | 1000 | Steps avant début apprentissage |
| `--disable-double-dqn` | — | Désactive le Double DQN |
| `--curriculum` | — | Active le curriculum learning |
| `--curriculum-stage-1` | 500 | Fin du stage 1 (1 bot) |
| `--curriculum-stage-2` | 1000 | Fin du stage 2 (2 bots) |
| `--eval-every` | 100 | Évaluer tous les N épisodes |
| `--eval-episodes` | 50 | Épisodes par évaluation |
| `--save-every` | 500 | Checkpoint tous les N épisodes |
| `--log-every` | 50 | Afficher tous les N épisodes |
| `--output-dir` | outputs_v3 | Dossier de sortie |
| `--seed` | 42 | Graine aléatoire |

### `evaluate_dqn.py`

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--model-path` | *requis* | Chemin vers le fichier `.pt` |
| `--episodes` | 50 | Nombre d'épisodes d'évaluation |
| `--render` | — | Affiche chaque étape (ASCII) |
| `--n-bots` | 2 | Nombre de bots adversaires |
| `--hidden-dim` | 256 | Taille des couches cachées |

### `benchmark_baselines.py`

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--model-path` | — | Chemin modèle (optionnel) |
| `--episodes` | 100 | Épisodes par politique |
| `--policies` | all | Politiques à tester |
| `--bot-policy` | heuristic | Politique des bots adversaires |
| `--output-json` | benchmark_results.json | Fichier résultats |

### `snake_ui.py`

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--model-path` | — | Modèle pré-sélectionné au lancement |
| `--width` / `--height` | 15 | Dimensions de l'arène |
| `--snakes` | 3 | Nombre de snakes total |

---

## Curriculum Learning

Avec `--curriculum`, l'entraînement progresse en 3 stages automatiques :

| Stage | Épisodes | Adversaires |
|-------|----------|-------------|
| 1 | 1 → 500 | 1 bot |
| 2 | 501 → 1000 | 2 bots |
| 3 | 1001 → fin | `n_bots` bots |

---

## Score d'évaluation composite

Le meilleur modèle est sauvegardé selon :

```
score = 120 × win_rate + 20 × draw_rate + 0.35 × score_moyen + 0.10 × reward_moyen - 2 × rang_moyen
```

---

## Politiques de baseline

| Politique | Description |
|-----------|-------------|
| `random` | Action aléatoire |
| `safe_random` | Action aléatoire parmi les coups sûrs |
| `greedy` | Chasse la nourriture la plus proche |
| `heuristic` | IA déterministe (utilisée par les bots en jeu) |
| `model` | Agent DQN chargé depuis fichier `.pt` |

---

## Dépendances

```bash
pip install torch numpy matplotlib
```
