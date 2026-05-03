# Autonomous Driving Multi-Agent AI

## Introduction

Ce projet propose une base logicielle pour l’entraînement et l’inférence de systèmes de conduite autonome dans le simulateur CARLA. Il combine un environnement Gym-like, des agents d’apprentissage par renforcement (PPO / SAC) et un cadre de simulation pour l’expérimentation sur des scénarios de trajectoire et de sécurité.

## Objectif

Permettre à des développeurs et chercheurs de tester des politiques d’agents autonomes dans un environnement CARLA, avec une configuration centralisée, une journalisation des entraînements et des scripts d’inférence.

---

## Fonctionnalités principales

- Entraînement et inférence avec les algorithmes PPO et SAC
- Wrapper CARLA compatible style Gym
- Gestion des récompenses et métriques d’évaluation
- Chargement de configuration YAML personnalisée
- Environnement de simulation avec véhicule ego, trafic et capteurs caméra

---

## Installation

### Prérequis

- Python 3.8+
- CARLA 0.9.15 installé et accessible
- Un environnement virtuel recommandé

### Étapes

1. Cloner le dépôt :

```bash
git clone <repo-url>
cd autonomous-driving-multi-agent-ai
```

2. Créer un environnement Python :

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

4. Vérifier que CARLA tourne sur `localhost:2000` avant de lancer l’entraînement ou l’inférence.

> Si CARLA n’est pas lancé, le projet peut tenter de se connecter plusieurs fois puis arrêter.

---

## Utilisation

### Exécution principale

Le script principal est `run.py`.

```bash
python run.py --mode train --algorithm ppo
python run.py --mode infer --algorithm ppo
python run.py --mode evaluate --algorithm sac
```

### Commandes disponibles

- `--mode train` : lancer l’entraînement
- `--mode infer` : lancer l’inférence avec un modèle enregistré
- `--mode evaluate` : évaluer un modèle sur plusieurs épisodes
- `--algorithm {ppo,sac}` : choisir l’algorithme
- `--config path/to/config.yaml` : charger un fichier de configuration personnalisé
- `--model-path path/to/model.pt` : charger un modèle spécifique

### Utiliser les scripts dédiés

- Entraînement :

```bash
python src/scripts/train.py --algorithm ppo --episodes 50
```

- Inférence :

```bash
python src/scripts/infer.py --algorithm sac --episodes 5 --model-path models/sac_agent.pth
```

### Exemple avec configuration personnalisée

```bash
python run.py --mode train --algorithm ppo --config config/config.yaml
```

---

## Architecture du projet

Le code est organisé pour séparer la configuration, l’environnement, les agents et la logique d’entraînement.

### Structure principale

- `src/` : code source du projet
  - `config.py` : gestion des paramètres et du YAML
  - `environment/` : wrapper CARLA et gestion de l’environnement
  - `reward/` : fonction de récompense
  - `state_representation/` : construction de l’état pour l’agent
  - `rl/` : implémentations PPO et SAC
  - `training/` : pipeline d’entraînement et d’évaluation
  - `evaluation/` : métriques et rapports
  - `visualization/` : génération de graphiques d’entraînement
  - `scripts/` : points d’entrée dédiés pour train/infer
- `config/` : fichiers YAML de configuration
- `requirements.txt` : dépendances Python
- `LICENSE` : licence du projet

### Flux général

1. Charger la configuration YAML
2. Initialiser l’environnement CARLA
3. Construire l’état à partir des capteurs
4. Sélectionner l’action avec l’agent RL
5. Exécuter l’action dans CARLA
6. Calculer la récompense et les métriques
7. Enregistrer les checkpoints et les logs

---

## Bonnes pratiques de développement

- Favoriser la lisibilité du code : noms explicites, fonctions courtes et commentaires clairs
- Respecter la modularité : séparer l’environnement, la logique d’agent et le pipeline d’entraînement
- Utiliser la configuration YAML pour éviter les constantes codées en dur
- Ajouter des tests ou des vérifications lors de l’ajout de nouvelles fonctionnalités
- Maintenir des logs et des checkpoints pour l’analyse des résultats

---

## Technologies utilisées

- Python 3.8+
- CARLA simulator
- PyTorch
- Gymnasium
- NumPy, Pandas, Matplotlib
- OpenCV
- YAML
- Ray RLlib (pour support multi-agent)

---

## Contribution

Vous pouvez contribuer en suivant ces étapes :

1. Créer une branche dédiée
2. Vérifier le style et la cohérence du code
3. Ajouter des tests si nécessaire
4. Soumettre une Pull Request

### Bonnes règles pour les contributions

- Faire des commits atomiques et bien décrits
- Garder les modifications ciblées
- Documenter les nouveaux paramètres de configuration
- Relire le code avant la validation

---

## Licence

Ce projet est distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## Notes importantes

- `config/config.yaml` contient les paramètres de CARLA, la récompense et l’entraînement.
- Le fichier `src/main.py` est un exemple simple de connexion CARLA.
- Les checkpoints sont sauvegardés dans le dossier `checkpoints/`.
- Les modèles sont enregistrés par défaut dans `models/ppo_agent.pth` ou `models/sac_agent.pth`.

# CARLA Multi-Agent Reinforcement Learning Project

A production-grade CARLA RL pipeline with PPO/SAC training, evaluation, and inference.

## What this project includes

- PPO and SAC reinforcement learning implemented in PyTorch
- A Gym-like CARLA environment wrapper with ego vehicle, traffic, and camera perception
- Configurable reward shaping, checkpointing, and evaluation metrics
- Training, inference, and evaluation entry points
- Optional custom config path support

## Requirements

- Python 3.8+
- CARLA 0.9.15 server running on `localhost:2000`
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python run.py --mode train --algorithm ppo
python run.py --mode infer --algorithm ppo
```

The script will:

1. Load `config/config.yaml`
2. Train or evaluate the selected agent
3. Save checkpoints to `checkpoints/`
4. Save a training history plot to `checkpoints/training_history.png`

## Configuration

Edit `config/config.yaml` to change CARLA settings, training hyperparameters, and reward weights.

Use a custom config file:

```bash
python run.py --config config/custom_config.yaml --mode train --algorithm sac
```

## Notes

- If CARLA is not running, start it before launching the script.
- To enable automatic CARLA launch, set `carla.auto_launch: true` and configure `carla_root` or `launch_command`.


# With custom CARLA settings
python src/multi_agent_main.py --host localhost --port 2000 --duration 500
```

#### Legacy Single-Agent System
```bash
# Run the original fall detection demo
python src/main.py --visualize
```

## 📁 Project Structure

```
src/
├── agents/                    # Multi-agent system
│   ├── base_agent.py         # Base agent classes and message bus
│   ├── perception_agent.py   # Object detection agent
│   ├── prediction_agent.py   # Behavior prediction agent
│   ├── decision_agent.py     # Decision-making agent
│   ├── planning_agent.py     # Trajectory planning agent
│   └── control_agent.py      # Low-level control agent
├── envs/                     # Environment interfaces
│   ├── carla_env.py          # CARLA environment wrapper
│   └── carla_simulation.py   # CARLA simulation utilities
├── communication/            # Inter-agent communication
├── decision/                 # Decision algorithms
├── models/                   # ML models and weights
├── safety/                   # Safety systems (CBF)
├── utils/                    # Utilities
├── multi_agent_main.py       # Main multi-agent entry point
└── main.py                   # Legacy single-agent entry point
```

## 🎯 System Features

### Perception Capabilities
- Real-time object detection (vehicles, pedestrians, traffic signs)
- Distance estimation and 3D positioning
- Confidence scoring and uncertainty estimation

### Prediction Features
- Short-term trajectory prediction (2-5 seconds)
- Intention recognition (lane change, stopping, turning)
- Multi-modal prediction with confidence scores

### Decision Making
- Rule-based safety decisions
- ML-enhanced decision making
- Emergency response protocols

### Planning & Control
- Smooth trajectory generation
- PID-based control systems
- Safety constraint satisfaction

## 🔧 Configuration

### Agent Parameters
Edit agent initialization in `multi_agent_main.py`:
```python
# Perception agent settings
perception_agent = PerceptionAgent(message_bus, confidence_threshold=0.5)

# Planning agent settings
planning_agent = PlanningAgent(message_bus, planning_horizon=5.0)
```

### CARLA Settings
```python
simulation = CarlaSimulation(
    host="localhost",
    port=2000,
    map_name="Town03",
    synchronous=True
)
```

## 📊 Monitoring & Visualization

The system provides real-time status updates:
- Agent states and health
- Detection counts and confidence
- Control commands and safety status
- Performance metrics

## 🛡️ Safety Systems

- **Control Barrier Functions (CBF)**: Mathematical safety guarantees
- **Emergency braking**: Automatic collision avoidance
- **Safety overrides**: Control command validation
- **Monitoring**: Continuous system health checks

## 🔬 Research & Development

### Current Capabilities
- ✅ Multi-agent communication framework
- ✅ Real-time perception pipeline
- ✅ Basic prediction and planning
- ✅ CARLA integration
- ✅ Safety monitoring

### Future Enhancements
- 🚧 End-to-end learning pipeline
- 🚧 Advanced prediction models (LSTM/Transformer)
- 🚧 Reinforcement learning integration
- 🚧 Multi-vehicle coordination
- 🚧 HD map integration

## 👥 Team

* **Zakariae** - Multi-agent architecture
* **Meryam** - Perception and computer vision

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎮 Quick Start Example

```bash
# 1. Start CARLA simulator
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run multi-agent system
python src/multi_agent_main.py --visualize

# Expected output:
# 🤖 Initializing Multi-Agent System...
# ✅ All agents initialized successfully
# ✅ CARLA simulation initialized
# 📊 System Status [t=2.1s]:
#   • perception: processing
#   • prediction: processing
#   • decision: processing
#   • planning: processing
#   • control: processing
```
3. Lancez le script principal :
   ```bash
   python src/main.py --visualize
   ```
4. Pour changer la carte ou le nombre de frames :
   ```bash
   python src/main.py --map Town03 --frames 400 --visualize
   ```

> Le script démarre une simulation CARLA, crée un véhicule ego, un piéton, puis simule un événement de chute. La détection est affichée en console et, si `--visualize` est activé, dans une fenêtre OpenCV.


