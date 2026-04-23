import os

# Dossier racine (déjà dedans)
base_dir = "."

# Dossiers à créer (seulement s'ils n'existent pas)
directories = [
    "data",
    "notebooks",
    "src/agents",
    "src/environment",
    "src/models",
    "src/decision",
    "src/utils",
    "src/carla_env",
    "tests",
    "docs"
]

# Fichiers standards à créer (si absents)
files = {
    "README.md": "# Autonomous Driving Multi-Agent AI\n\nProjet de niveau Master utilisant CARLA.\n",
    "requirements.txt": "",
    ".gitignore": "venv/\n__pycache__/\n.ipynb_checkpoints/\n"
}

# Fichiers Docker (ajoutés uniquement s'ils n'existent pas)
docker_files = {
    ".dockerignore": """venv/
__pycache__/
.ipynb_checkpoints/
.git/
.gitignore
notebooks/
tests/
docs/
""",

    "Dockerfile": """FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

CMD ["python", "src/main.py"]
""",

    "docker-compose.yml": """version: "3.9"

services:
  carla:
    image: carlasim/carla:0.9.15
    ports:
      - "2000:2000"
      - "2001:2001"
    command: /bin/bash ./CarlaUE4.sh -nosound -RenderOffScreen

  ai:
    build: .
    depends_on:
      - carla
    environment:
      - CARLA_HOST=carla
      - CARLA_PORT=2000
"""
}

# main.py (client CARLA minimal)
main_py_content = """import carla
import os

host = os.getenv("CARLA_HOST", "localhost")
port = int(os.getenv("CARLA_PORT", 2000))

client = carla.Client(host, port)
client.set_timeout(10.0)

world = client.get_world()
print("✅ Connecté à CARLA :", world.get_map().name)
"""

print("🚀 Vérification et ajout des éléments manquants...\n")

# Création des dossiers
for directory in directories:
    dir_path = os.path.join(base_dir, directory)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"📁 Dossier créé : {directory}")

        # .gitkeep pour Git
        with open(os.path.join(dir_path, ".gitkeep"), "w"):
            pass
    else:
        print(f"✔ Dossier existe : {directory}")

# Création des fichiers standards
for file, content in files.items():
    file_path = os.path.join(base_dir, file)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"📄 Fichier créé : {file}")
    else:
        print(f"✔ Fichier existe : {file}")

# Docker files
for file, content in docker_files.items():
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"🐳 Docker ajouté : {file}")
    else:
        print(f"✔ Docker existe : {file}")

# main.py
main_py_path = os.path.join("src", "main.py")
if not os.path.exists(main_py_path):
    with open(main_py_path, "w", encoding="utf-8") as f:
        f.write(main_py_content)
    print("🧠 main.py créé (client CARLA)")
else:
    print("✔ src/main.py existe déjà")

print("\n✅ Projet mis à jour sans écraser les fichiers existants")