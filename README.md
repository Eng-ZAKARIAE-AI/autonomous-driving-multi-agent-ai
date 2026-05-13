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

---

## Architecture du projet

- `src/` : code source du projet
  - `config.py` : gestion des paramètres et du YAML
  - `environment/` : wrapper CARLA et gestion de l’environnement
  - `reward/` : fonction de récompense
  - `state_representation/` : construction de l’état pour l’agent
  - `rl/` : implémentations PPO et SAC
  - `training/` : pipeline d’entraînement et d’évaluation
  - `evaluation/` : métriques et rapports
  - `visualization/` : génération de graphiques d’entraînement
- `config/` : fichiers YAML de configuration
- `requirements.txt` : dépendances Python

---

## Licence

Ce projet est distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus de détails.
