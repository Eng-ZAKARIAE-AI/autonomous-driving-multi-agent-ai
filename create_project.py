import os

# Configuration du projet
BASE_DIR = "."

# Structure des dossiers complète (Source: Page 2, 3)
DIRECTORIES = [
    "docker",
    "configs",
    "src/envs",
    "src/agents",
    "src/models",
    "src/safety",
    "src/communication",
    "src/utils",
    "training",
    "tests",
    "results/logs",
    "results/checkpoints"
]

# --- CONTENU DES FICHIERS ---

# Configuration des Scénarios (Source: Page 4)
ENV_CONFIG = """# configs/env_config.yaml
scenarios:
  - name: urban_intersection
    agents: 4
    traffic: moderate
    weather: clear
    goal: navigate intersection without collision
  - name: highway_merge
    agents: 6
    traffic: dense
    goal: merge safely at 80 km/h
"""

# Dockerfile Agents (Source: Page 6)
DOCKERFILE_AGENTS = """FROM python:3.11-slim
WORKDIR /app
# Installation des dépendances système pour OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN pip install carla==0.9.15 gymnasium numpy opencv-python torch stable-baselines3 pettingzoo ray[rllib] redis
COPY . .
CMD ["python", "src/main.py"]
"""

# Docker Compose (Source: Page 6)
DOCKER_COMPOSE = """version: "3.8"
services:
  carla-server:
    image: carlasim/carla:0.9.15
    runtime: nvidia
    ports:
      - "2000-2002:2000-2002"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    command: /bin/bash CarlaUE4.sh -RenderOffScreen -world-port=2000

  agent-trainer:
    build:
      context: .
      dockerfile: docker/Dockerfile.agents
    depends_on:
      - carla-server
    volumes:
      - ./src:/app/src
      - ./results:/app/results
      - ./configs:/app/configs
    environment:
      - CARLA_HOST=carla-server
"""

# Wrapper Gymnasium Minimal (Source: Page 6)
CARLA_ENV_PY = """import gymnasium as gym
import carla
import numpy as np

class CarlaEnv(gym.Env):
    def __init__(self, host='localhost', port=2000):
        super().__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
    def reset(self, seed=None):
        # Logique de reset simplifiée
        return {}, {}

    def step(self, action):
        # Logique de step simplifiée
        return {}, 0.0, False, False, {}
"""

# Sécurité : Control Barrier Function (Source: Page 12)
CBF_PY = """import numpy as np

class ControlBarrierFunction:
    def __init__(self, safety_margin=2.5, alpha=0.5):
        self.d_safe = safety_margin [cite: 323]
        self.alpha = alpha [cite: 323]

    def h(self, ego, obstacle):
        # Distance euclidienne - marge de sécurité
        dist = np.sqrt((ego.x - obstacle.x)**2 + (ego.y - obstacle.y)**2) - self.d_safe [cite: 323]
        return dist
"""

# Main.py : Point d'entrée
MAIN_PY = """import carla
import os
from src.envs.carla_env import CarlaEnv

def main():
    print("🚀 Initialisation du projet Autonomous Driving...")
    host = os.getenv("CARLA_HOST", "localhost")
    try:
        env = CarlaEnv(host=host)
        print("✅ Connexion réussie au serveur CARLA")
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
"""

# --- LOGIQUE DE CRÉATION ---

FILES = {
    "configs/env_config.yaml": ENV_CONFIG,
    "docker/Dockerfile.agents": DOCKERFILE_AGENTS,
    "docker-compose.yml": DOCKER_COMPOSE,
    "src/envs/carla_env.py": CARLA_ENV_PY,
    "src/safety/cbf.py": CBF_PY,
    "src/main.py": MAIN_PY,
    "requirements.txt": "carla==0.9.15\\ngymnasium\\nnumpy\\ntorch\\nstable-baselines3\\npettingzoo\\nray[rllib]\\nredis",
    ".gitignore": "venv/\\n__pycache__/\\nresults/\\n.env\\n*.pyc"
}

def create_project():
    print("🏗️ Création de la structure du projet...")
    
    # Création des répertoires
    for folder in DIRECTORIES:
        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)
        # Ajout d'un .gitkeep pour Git
        with open(os.path.join(path, ".gitkeep"), "w") as f:
            pass
        print(f"  ✔ Dossier: {folder}")

    # Création des fichiers
    for file_path, content in FILES.items():
        full_path = os.path.join(BASE_DIR, file_path)
        if not os.path.exists(full_path):
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✔ Fichier: {file_path}")
        else:
            print(f"  ⚠ Existe déjà: {file_path}")

    print("\n✅ Projet prêt ! Utilisez 'docker-compose up' pour démarrer l'infrastructure.")

if __name__ == "__main__":
    create_project()