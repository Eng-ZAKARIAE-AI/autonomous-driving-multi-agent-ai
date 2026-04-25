import os

# On utilise '.' car vous êtes déjà dans le dossier cible
base_dir = "."

# Liste des dossiers à créer
directories = [
    "data",
    "notebooks",
    "src/agents",
    "src/models",
    "src/safety",
    "src/communication",
    "src/utils",
    "tests",
    "results/logs",
    "results/checkpoints"
]

# Liste des fichiers à créer
files = [
    "README.md",
    "requirements.txt",
    ".gitignore"
]

print("Création de l'arborescence dans le dossier actuel...")

# Création des dossiers
for directory in directories:
    dir_path = os.path.join(base_dir, directory)
    os.makedirs(dir_path, exist_ok=True)
    
    # Ajout d'un fichier .gitkeep pour que les dossiers vides soient détectés par Git
    with open(os.path.join(dir_path, ".gitkeep"), 'w') as f:
        pass

# Création des fichiers
for file in files:
    file_path = os.path.join(base_dir, file)
    with open(file_path, 'w', encoding='utf-8') as f:
        # Ajout d'un titre basique dans le README
        if file == "README.md":
            f.write("# Autonomous Driving Multi-Agent AI\n")

print("✅ Arborescence créée avec succès !")