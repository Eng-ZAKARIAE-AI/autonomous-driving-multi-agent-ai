import torch

ckpt = torch.load("backend/weights/transfuser/model_seed1_39.pth", map_location="cpu")

if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print("Top-level keys:", keys[:10])
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    param_keys = list(sd.keys())
    print(f"\nTotal params: {len(param_keys)}")
    print("First 20 keys:")
    for k in param_keys[:20]:
        print(f"  {k}: {sd[k].shape}")
else:
    print("Type:", type(ckpt))
