import torch
from collections import OrderedDict

ckpt = torch.load("backend/weights/transfuser/model_seed1_39.pth", map_location="cpu")

cleaned = OrderedDict()
for k, v in ckpt.items():
    new_k = k.replace("module.", "", 1)
    cleaned[new_k] = v

print("=== TOUTES LES CLÉS _model ===")
for k, v in cleaned.items():
    if k.startswith("_model"):
        print(f"  {k}: {v.shape}")

print("\n=== join + decoder + output ===")
for k, v in cleaned.items():
    if any(k.startswith(x) for x in ["join", "decoder", "output"]):
        print(f"  {k}: {v.shape}")
