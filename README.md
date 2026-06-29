# Intelligent Decision-Making System for Autonomous Driving using Multi-Agent AI

Projet de Fin d'Année (PFA) — Architecture modulaire à 5 agents communicants pour la conduite
autonome, simulée dans **CARLA 0.9.14+** (Town10HD_Opt, mode synchrone 20 Hz).

---

## Contexte et motivation

L'approche initiale reposait sur un agent RL monolithique (PPO/SAC) entraîné sur une récompense
scalaire. En pratique, deux problèmes se posent : l'opacité du comportement (boîte noire) et
l'impossibilité d'expliquer une décision à un jury ou un homologateur. L'architecture
multi-agents remplace ce pipeline par **5 agents à responsabilités séparées**, chacun testable
et présentable indépendamment.

---

## Architecture : 5 agents + Blackboard

### Diagramme de flux de données

```
                         CARLA Simulator  (20 Hz — tick synchrone)
                                  │
             ┌────────────────────▼──────────────────────────────────┐
             │          Global Route Planner  (calculé à l'init)      │
             │  trace_route(A → B)  →  liste (Location, RoadOption)   │
             │  Publie : blackboard.route  (actif si --global-planner) │
             └────────────────────┬──────────────────────────────────┘
                                  │ itinéraire A→B (une fois)
                                  │
  ┌───────────────────────────────▼──────────────────────────────────────────┐
  │                        BOUCLE PAR TICK                                   │
  │                                                                          │
  │  ┌───────────────────────────────────────────────────────────────────┐   │
  │  │  1.  PERCEPTION                                                    │   │
  │  │      CarlaActorPerception   : obstacles, piétons (world.get_actors)│   │
  │  │      CarlaTrafficLightPerception : état TL + distance ligne arrêt  │   │
  │  │      CarlaStopSignPerception     : distance panneau STOP           │   │
  │  │      Calcule : forward_dist, lateral_offset, v_lead_kmh (3D→2D)   │   │
  │  │      Publie  → blackboard.perception                               │   │
  │  └──────────────────────────────┬────────────────────────────────────┘   │
  │                                 │  détections + TL + STOP                │
  │  ┌──────────────────────────────▼────────────────────────────────────┐   │
  │  │  2.  DECISION  (FSM comportementale)                               │   │
  │  │      Lit     : blackboard.perception, ego_speed, ego_heading       │   │
  │  │                                                                    │   │
  │  │      États FSM :                                                   │   │
  │  │        FOLLOW_LANE ──(obstacle < 15 m)──► SLOW_DOWN               │   │
  │  │                    ──(obstacle < 10 m)──► STOP                     │   │
  │  │                    ──(TL rouge)──────────► RED_LIGHT               │   │
  │  │                    ──(panneau STOP)───────► STOP_SIGN (hold 1 s)   │   │
  │  │                                                                    │   │
  │  │      Car-following (SLOW_DOWN + obstacle détecté) :                │   │
  │  │        v_tgt = v_lead + k_dist·(d − d_secu) − k_vel·(v_ego−v_lead)│   │
  │  │        d_secu = max(10 m, headway_s · v_ego_ms)   [règle 2 s]     │   │
  │  │                                                                    │   │
  │  │      Publie  → blackboard.decision  (fsm_state, target_speed,      │   │
  │  │                                      branch_heading)               │   │
  │  └──────────────────────────────┬────────────────────────────────────┘   │
  │                                 │  target_speed + branch_heading         │
  │  ┌──────────────────────────────▼────────────────────────────────────┐   │
  │  │  3.  PLANNING  (waypoints + suivi itinéraire)                      │   │
  │  │      Lit     : blackboard.decision, blackboard.route, carte CARLA  │   │
  │  │      Génère  : N waypoints (x, y, target_speed_kmh) devant l'ego  │   │
  │  │      Fork    : choisit la branche selon branch_heading              │   │
  │  │                  GoStraightPolicy   — cap courant (défaut)          │   │
  │  │                  TLSeekingPolicy    — branche vers un feu           │   │
  │  │                  RouteFollowingPolicy — suit itinéraire GRP         │   │
  │  │      Publie  → blackboard.planning  (waypoints, branch_in_chain)   │   │
  │  └──────────────────────────────┬────────────────────────────────────┘   │
  │                                 │  liste (x, y, v_kmh)                  │
  │  ┌──────────────────────────────▼────────────────────────────────────┐   │
  │  │  4.  CONTROL  (PID longitudinal + latéral Stanley)                 │   │
  │  │      Lit     : blackboard.planning, ego_speed, ego_heading, pos    │   │
  │  │      Latéral : steer = PID(Δheading)                               │   │
  │  │                      + k_ff · κ · L  (feedforward courbure)        │   │
  │  │                      + arctan(k_cte · CTE / v)  (terme Stanley)    │   │
  │  │      Longit. : PID normalisé → (throttle, brake) ∈ [0, 1]         │   │
  │  │      Publie  → blackboard.control  (steer, throttle, brake)        │   │
  │  └──────────────────────────────┬────────────────────────────────────┘   │
  │                                 │  commandes candidates                  │
  │  ┌──────────────────────────────▼────────────────────────────────────┐   │
  │  │  5.  SAFETY  (veto transversal — PRIORITAIRE sur tous les autres)  │   │
  │  │      Lit     : blackboard.perception  (toutes détections)          │   │
  │  │      Garde 1 : TTC = d_fwd / v_ego  <  2.0 s  →  VETO             │   │
  │  │      Garde 2 : d_fwd  <  7.5 m (même à l'arrêt)  →  VETO         │   │
  │  │      Latch   : maintient veto 20 ticks après levée TTC             │   │
  │  │      Si VETO : écrase blackboard.control → brake=0.7, throttle=0   │   │
  │  │      Logue   : chaque intervention (raison, tick, distance, TTC)   │   │
  │  │      Publie  → blackboard.safety  (override, latch, interventions) │   │
  │  └──────────────────────────────┬────────────────────────────────────┘   │
  │                                 │                                        │
  │              apply_control(steer, throttle, brake)  → ego CARLA         │
  │                          world.tick()                                    │
  └──────────────────────────────────────────────────────────────────────────┘
```

### Communication inter-agents : Blackboard pattern

Tous les agents lisent et écrivent via un seul objet `Blackboard` thread-safe (verrou par slot).
Pas de message-passing ni de bus — chaque agent publie dans son slot et lit les slots des agents
précédents. L'ordre d'exécution séquentiel dans la boucle garantit la cohérence des données au
sein d'un tick.

```
blackboard.perception  ←  CarlaActorPerception
                          CarlaTrafficLightPerception
                          CarlaStopSignPerception
blackboard.decision    ←  DecisionAgent
blackboard.planning    ←  PlanningAgent
blackboard.control     ←  ControlAgent  (puis Safety peut écraser)
blackboard.safety      ←  SafetyAgent
blackboard.route       ←  GlobalPlannerAgent  (calculé une fois à l'init)
blackboard.rl_policy   ←  réservé — extension PPO lane-change (interface prête)
```

---

## Fonctionnalités prouvées

| Fonctionnalité | Scénario de validation | Résultat mesuré |
|---|---|---|
| Contrôle latéral (Stanley + CTE) | spawn-road 900, 200 ticks | CTE < 0.15 m droit, < 0.30 m arc |
| Navigation globale A→B | spawn 28 → 34, GlobalRoutePlanner, --tl-green-time 99 | route_complete=True t=1142, zéro FALLBK (3/3 runs identiques) |
| Feu rouge | spawn 28, cycle naturel | Arrêt < 3 m avant ligne, reprise au vert |
| Panneau STOP | spawn 28, landmark STOP | Hold 20 ticks, reprise propre |
| Véhicule arrêté | --stopped-vehicle-at 15 --force-red-until 0 | SLOW_DOWN, ego décélère 7.4→0 km/h en 1 s, fwd=13.3 m stable, 0 collision |
| Piéton qui traverse | --pedestrian-cross-at 10 | Détection class=walker, STOP pendant traversée, reprise |
| Car-following cas b | --spawn-road 900, --traffic-speed 10, 300 ticks | d_obs 12.7–15.2 m stable > d_secu, 0 veto Safety |
| Car-following cas c | --spawn-road 900, --traffic-brake-at 80 | Lead 10→0 km/h en 10 ticks ; ego stop, d_obs_min=14.8 m, 0 veto |
| Safety veto + latch | `--obstacle-at 20` | VETO déclenché à TTC < 2.0 s, CLEAR automatique après latch |
| Safety benchmark | 28 épisodes, spawn 28 | 0 collision, 0 faux-positif |

---

## Paramètres de configuration clés

**`backend/config/config.yaml` — section `agents.planning` :**

```yaml
target_speed:           16.0   # km/h — vitesse de croisière
slow_speed:              6.0   # km/h — SLOW_DOWN (obstacle ou fork)
obstacle_distance:      15.0   # m    — seuil entrée SLOW_DOWN
obs_stop_m:             10.0   # m    — seuil STOP FSM (avant zone Safety 7.5 m)
obs_slow_hysteresis_m:   5.0   # m    — SLOW_DOWN maintenu jusqu'à d > obstacle_distance + hyst
cf_headway_s:            2.0   # s    — règle des 2 secondes (d_secu dynamique)
cf_k_dist:               0.5   # km/h/m — gain distance car-following
cf_k_vel:                0.5   # sans dim — amortissement vitesse relative (anti-accordéon)
tl_detect_m:            50.0   # m    — horizon armement feu
stop_detect_m:          50.0   # m    — horizon armement STOP
```

**`backend/config/config.yaml` — section `safety` :**

```yaml
ttc_emergency: 2.0    # s    — seuil déclenchement veto TTC (calibré sur d_stop mesuré)
ttc_clear:     3.0    # s    — seuil levée veto (hystérèse)
latch_ticks:   20     # ticks — maintien veto après levée TTC (~1 s à 20 Hz)
brake_force:   0.7    # [0,1] — pression de frein pendant le veto
```

**SafetyAgent — constante interne (non configurable) :**

```
MIN_SAFE_M = 7.5 m   — distance minimale centre-à-centre (veto maintenu même à l'arrêt)
             Calibrée : ego s'arrête à 6.6 m après veto → 7.5 m > 6.6 m (marge active)
```

---

## Limites documentées et domaine de validité

### Ce qui est couvert

- Détection obstacle/piéton fiable sur **route droite jusqu'à 50 m** (lat < 1.5 m)
- Intersection avec TL ou STOP : arrêt correct + reprise
- Car-following sur droit : convergence sans accordéon, freinage brutal géré par le terme k_vel
- Navigation A→B multi-segments sur Town10HD_Opt
- Safety : 2 gardes complémentaires, latch, benchmark 28 épisodes sans collision

### Ce qui n'est pas couvert (extension documentée)

| Limite | Cause racine | Extension identifiée |
|---|---|---|
| **Cécité euclidienne en virage** (limite structurelle principale) | `CarlaActorPerception` projette l'acteur en ligne droite depuis l'ego. En courbe, l'offset latéral euclidien dépasse le seuil de filtre → zone aveugle ~8–10 m en virage. Frappée 5 fois pendant le développement. | Projection **par chaîne de waypoints** (distance curviligne le long de la route) — cohérente avec le CTE de Stanley. Extension prioritaire de prochaine itération. |
| **Car-following validé sur droit uniquement** | `lane_half_width=8 m` est un contournement local pour spawn28 (courbe à ~60°). En virage réel, le lead peut disparaître de la détection. | Même fix que ci-dessus. |
| **Zone aveugle état feu (~42 m)** | L'état du TL est lisible uniquement à < ~42 m. Géré par détection précoce (`tl_detect_m=50 m`) + latch. | Augmenter `tl_detect_m` ou fusion avec perception visuelle caméra. |
| **Speed-limit non démontrable sur Town10** | La limite de vitesse est accessible via l'API CARLA (`waypoint.get_landmarks_of_type`) et publiée dans `PerceptionAgent.lane_geometry["speed_limit_kmh"]`. Town10HD_Opt est une zone-30 uniforme : le mécanisme de lecture fonctionne, mais la FSM DecisionAgent ne l'utilise pas encore comme seuil de vitesse dynamique. | Câbler `speed_limit_kmh` comme cap de `target_speed` dans DecisionAgent ; tester sur Town01/Town03 pour des zones multi-vitesses. |
| **Obstacle en arc serré** | Pas d'état FSM `CAUTIOUS_CURVE`. En virage à faible rayon, l'obstacle n'est pas détecté + la vitesse n'est pas réduite par précaution. | Ajouter état FSM `CAUTIOUS_CURVE` : κ > seuil → réduire target_speed. |
| **Vision YOLO non activée** | VRAM 4 GB insuffisant pour YOLOv8 temps réel simultané avec CARLA. | `USE_YOLO=True` dans `config.yaml` + CPU offload (code présent dans `perception_agent.py`, non validé en live). |
| **RL (PPO/SAC) remplacé** | Architecture multi-agents remplace le pipeline RL monolithique. | Slot `blackboard.rl_policy` réservé. Interface `BranchPolicy` prête pour injection d'une `RLLaneChangePolicy` (PPO) sans modifier les autres agents. |
| **Bug natif CARLA 0.9.14 — TM synchrone + route à fourche** | `tm.set_synchronous_mode(True)` + route avec divergence déclenche un crash 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN dans `carla.pyd`) après ~285 ticks. Diagnostic : le crash n'est pas lié au code du projet — il est reproductible à blanc (script Python minimal). **Non lié à la navigation A→B** (aucun TM activé sur ce scénario). | Contournement validé : car-following isolé sur route droite sans fourche (`--spawn-road 900`, TM async). La navigation A→B n'est pas affectée. |

---

## Structure du code

```
backend/src/
├── agents/
│   ├── blackboard.py                    Blackboard thread-safe (5 slots typés + route)
│   ├── carla_actor_perception.py        Perception obstacles/piétons (ground-truth CARLA)
│   ├── carla_traffic_light_perception.py Feux tricolores (état + distance ligne arrêt)
│   ├── carla_stop_sign_perception.py    Panneaux STOP (distance)
│   ├── decision_agent.py               FSM + car-following IDM-simplifié
│   ├── planning_agent.py               Waypoints + BranchPolicy (3 politiques)
│   ├── control_agent.py                PID longitudinal + PID/Stanley latéral
│   ├── safety_agent.py                 Veto TTC + distance, latch, log interventions
│   └── global_planner.py              GlobalPlannerAgent (trace_route A→B, CARLA API)
├── scripts/
│   ├── demo_control_carla.py           Script principal (tous scénarios)
│   ├── benchmark_agents.py             Benchmark Safety N épisodes
│   └── discover_signs.py              Scan panneaux STOP sur la carte
└── config.py                          Chargement config.yaml centralisé
backend/config/config.yaml             Configuration CARLA + agents + capteurs
simulator/envs/carla_env.py            Connexion CARLA + cycle de vie acteurs
```

---

## Prérequis et lancement rapide

```bash
# 1. Lancer CARLA 0.9.14+
CarlaUE4.exe -windowed -ResX=1280 -ResY=720 -quality-level=Low

# 2. Dépendances Python 3.7
pip install -r requirements.txt

# 3. Depuis la racine du projet
set PYTHONPATH=.          # Windows
# export PYTHONPATH=.     # Linux/macOS

# Démo nominale — spawn 28 (TL + STOP naturels)
python -m backend.src.scripts.demo_control_carla --spawn-index 28 --ticks 400

# Navigation A→B avec Global Route Planner
# --tl-green-time 99 : TL verts 99s (>1400 ticks) — évite les fenêtres rouges
# aléatoires mid-route (déterminisme garanti : 3/3 runs → ATTEINTE t=1142)
python -m backend.src.scripts.demo_control_carla \
    --spawn-index 28 --route-to 34 --global-planner --ticks 1400 \
    --force-red-until 0 --tl-green-time 99

# Scénario 2 — piéton qui traverse à 10 m
python -m backend.src.scripts.demo_control_carla \
    --spawn-index 28 --pedestrian-cross-at 10 --ticks 250

# Scénario 2b — véhicule arrêté à 15 m
python -m backend.src.scripts.demo_control_carla \
    --spawn-index 28 --stopped-vehicle-at 15 --ticks 200 --force-red-until 0

# Scénario 3a — car-following vitesse constante 10 km/h
# (spawn-road 900 = route droite sans fork ; évite le crash natif CARLA 0.9.14
#  du Traffic Manager en mode synchrone après ~285 ticks sur des routes avec fourches)
python -m backend.src.scripts.demo_control_carla \
    --spawn-road 900 --traffic-ahead --traffic-speed 10 --traffic-dist 15 --ticks 300

# Scénario 3b — car-following + freinage brutal au tick 80
python -m backend.src.scripts.demo_control_carla \
    --spawn-road 900 --traffic-ahead --traffic-speed 10 \
    --traffic-dist 15 --traffic-brake-at 80 --ticks 250

# Benchmark Safety (28 épisodes)
python -m backend.src.scripts.benchmark_agents --n-episodes 28
```

---

## Résultats synthétiques

| Métrique | Valeur | Condition de mesure |
|---|---|---|
| CTE sur droit | < 0.15 m | spawn-road 900, 200 ticks |
| CTE en arc | < 0.30 m | Stanley + feedforward courbure |
| Collisions Safety | 0 / 28 épisodes | Benchmark spawn 28 |
| Faux-positifs Safety | 0 / 28 épisodes | Benchmark spawn 28 |
| Veto car-following cas b | 0 | v_lead = 10 km/h constant, 300 ticks |
| Veto car-following cas c | 0 | Freinage brutal lead → 0 km/h |
| d_min cas c (freinage brutal) | 10.1 m | > obs_stop_m = 10.0 m |
| Convergence car-following | 14 → 12 m monotone | Pas d'accordéon (terme k_vel) |
| Arrêt feu rouge | < 3 m avant ligne | tl_brake_margin = 3 m |
| Hold STOP réglementaire | 20 ticks (1.0 s) | stop_hold_ticks configurable |
