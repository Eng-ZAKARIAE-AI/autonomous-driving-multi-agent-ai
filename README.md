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
