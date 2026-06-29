import torch
from collections import OrderedDict

ckpt = torch.load("backend/weights/transfuser/model_seed1_39.pth", map_location="cpu")

# Strip DataParallel "module." prefix
cleaned = OrderedDict()
for k, v in ckpt.items():
    new_k = k.replace("module.", "", 1)
    cleaned[new_k] = v

# Group by top-level submodule
groups = {}
for k in cleaned:
    top = k.split(".")[0]
    groups.setdefault(top, []).append(k)

print("=== SOUS-MODULES NIVEAU 1 ===")
for g, keys in groups.items():
    sample = cleaned[keys[0]].shape
    print(f"  {g:40s} ({len(keys)} params, ex: {sample})")

print(f"\nTotal params: {len(cleaned)}")

# Afficher toutes les clés uniques pour les têtes de contrôle
print("\n=== CLÉS DES COUCHES FINALES (control head) ===")
for k in cleaned:
    if any(x in k for x in ["head", "output", "pred", "control", "wp", "join", "speed"]):
        print(f"  {k}: {cleaned[k].shape}")
