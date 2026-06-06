# Rapport d'Avancement — Projet de Fin d'Année
**Simulateur de Conduite Autonome — Apprentissage par Renforcement Multi-Agent**

| | |
|---|---|
| **Étudiant** | Zakariae El Haddouchi |
| **Email** | zakariaeelhaddouchi@ump.ac.ma |
| **Date** | 04 Juin 2026 |
| **Branche Git** | `develop` |

| | |
|---|---|
| **Étudiante** | Meryam El Aiboudi |
| **Email** | meryemelaiboudi@ump.ac.ma |
| **Date** | 04 Juin 2026 |
| **Branche Git** | `develop` |

---

---


## 1. Tâches Terminées

### Architecture des Agents RL
- Implémentation complète de deux algorithmes d'apprentissage par renforcement :
  - **PPO** (Proximal Policy Optimization) avec GAE et objectif surrogate clipé
  - **SAC** (Soft Actor-Critic) avec double Q-network, Gaussian policy et soft target update
- Encodeurs partagés entre les deux agents (`backend/src/agents/common.py`) :
  - `ImageEncoder` : 3 couches Conv → Flatten → 2 FC → vecteur 128D
  - `VectorEncoder` : 2 FC → vecteur 64D
  - État fusionné : `{image: (3, 84, 84), vector: (5,)}` → 192D

### Environnement de Simulation
- Environnement CARLA compatible Gymnasium (`simulator/envs/carla_env.py`) :
  - Connexion au simulateur avec retry automatique (configurable)
  - Capteur caméra RGB (84×84) + capteur de collision
  - Mode synchrone pour reproductibilité
  - Objectif de navigation avec calcul de distance et d'angle

### Pipeline de Traitement d'État
- `StateBuilder` (`backend/src/models/state_builder.py`) : fusion multi-modale
  - `CameraProcessor` : redimensionnement et normalisation de l'image
  - Normalisation des features cinématiques : vitesse, écart de voie, distance objectif, angle, collision

### Fonction de Récompense
- Récompense multi-objectif pondérée (`backend/src/reward/reward_function.py`) :
  - Récompense de vitesse (maintien de la vitesse cible)
  - Récompense de centrage sur la voie
  - Récompense de progression vers l'objectif
  - Pénalité de confort (variation d'action)
  - Pénalité de collision
  - Récompense de succès (arrivée à destination)

### Pipeline d'Entraînement
- `Trainer` unifié (`backend/src/training/trainer.py`) :
  - Modes : `train`, `evaluate`, `infer`, `auto`
  - Logging CSV avec métriques par épisode (reward, steps, collision, vitesse, écart)
  - Sauvegarde de checkpoints périodique (`.pt`)
  - Génération automatique de courbe d'entraînement en fin de session
- CLI principal `run.py` avec arguments : `--mode`, `--algorithm`, `--episodes`, `--config`, `--model-path`

### Backend API
- Serveur FastAPI (port 8000) avec WebSocket `/ws/telemetry`
- Diffusion temps réel des métriques d'entraînement vers le frontend

### Frontend Dashboard
- Application React 19 + TypeScript + Tailwind CSS + Recharts + Vite (`frontend/web/`)
- Pages implémentées : **Dashboard**, **Training**, **Agents**, **Simulation**, **Telemetry**, **Settings**
- Composants : `MetricCard`, `RewardChart`, `Sidebar`, `Topbar`, `Layout`
- Hook `useTelemetry.ts` : souscription WebSocket au backend avec reconnexion automatique
- Affichage des données live + fallback mock si le backend est hors ligne

### Infrastructure
- `docker-compose.yml` orchestrant 3 services : CARLA (port 2000), backend (port 8000), frontend (port 3000)
- Système de configuration en cascade : `DEFAULT_CONFIG` → `config.yaml` → variables d'environnement (`CARLA_HOST`, `CARLA_PORT`)

---

## 2. Bugs Résolus

| Bug | Symptôme | Solution |
|-----|----------|----------|
| **Conflits de merge** | `trainer.py`, `carla_env.py`, `requirements.txt` et `config.yaml` avaient des marqueurs `<<<<<<<` | Résolution manuelle des conflits, stabilisation de l'environnement RL (`commit: d30bfdb`) |
| **Mauvais fichier modèle chargé** | L'agent SAC chargeait le modèle PPO (et inversement) au démarrage | Correction dans `Trainer.__init__` : détection et remplacement automatique du nom de fichier selon l'algorithme |
| **Actions hors bornes** | `throttle` et `brake` sortaient de `[0, 1]` car le réseau sort dans `[-1, 1]` | Remapping dans `_format_action` : `(x + 1) / 2` + logique d'exclusion mutuelle throttle/brake |
| **Instabilité SAC** | Divergence des Q-values en début d'entraînement | Utilisation de Clipped Double-Q (min des deux critiques) pour les cibles Bellman |
| **Frontend inutilisable sans backend** | Crash au démarrage si WebSocket indisponible | Ajout d'un reducer `INITIALIZE_MOCK` dans `Dashboard.tsx` + gestion de reconnexion dans `useTelemetry.ts` |
| **Seed non reproductible** | Résultats différents à chaque run | `_set_seed()` dans le Trainer initialise `random`, `numpy` et `torch` (CPU + GPU) |

---

## 3. Ce Qui Reste à Faire

### Court terme (avant soutenance)
- [ ] **Tests automatisés** : le dossier `tests/` est vide — écrire au minimum des tests smoke pour `Trainer` et `RewardFunction`
- [ ] **Métriques live côté frontend** : les métriques d'évaluation sont calculées (`evaluation/metrics.py`) mais non diffusées via WebSocket — les connecter au dashboard
- [ ] **Run d'entraînement de référence** : lancer un entraînement SAC complet (100+ épisodes) sur Town03 pour obtenir un modèle de base documenté

### Moyen terme
- [ ] **Vrai multi-agent** : plusieurs véhicules RL simultanés dans CARLA (actuellement un seul ego-vehicle)
- [ ] **Généralisation maps** : tester sur Town05 et Town10 en plus de Town03
- [ ] **Hyperparameter tuning** : documenter l'impact des poids de récompense et des hyperparamètres SAC/PPO
- [ ] **Nettoyage code legacy** : supprimer `backend/src/system.py` et `backend/src/envs/` (chemin obsolète conservé pour référence)

---

## 4. Prochaines Étapes Immédiates

1. **Connecter les métriques au WebSocket** — modifier `multi_agent_main.py` pour diffuser les résultats de `compute_metrics()` après chaque épisode
2. **Écrire les tests smoke** — tester `RewardFunction.compute()` et `Trainer._format_action()` sans CARLA (mock d'environnement)
3. **Lancer l'entraînement** — `python run.py --mode train --algorithm sac --episodes 100` sur machine avec CARLA, sauvegarder les courbes
4. **Rédiger le rapport PFA final** en s'appuyant sur les courbes d'entraînement et les métriques collectées

---

## 5. Stack Technique Résumée

| Couche | Technologies |
|--------|-------------|
| Simulateur | CARLA 0.9.x (Unreal Engine) |
| Agents RL | PyTorch, PPO, SAC |
| Perception | Conv Neural Network (ImageEncoder) |
| Backend API | FastAPI, WebSocket, Python 3.10+ |
| Frontend | React 19, TypeScript, Tailwind CSS, Recharts, Vite |
| Infrastructure | Docker Compose, YAML config |

---

*Rapport généré le 04/06/2026 — projet sur branche `develop`, dernier commit : `649d9af`*
