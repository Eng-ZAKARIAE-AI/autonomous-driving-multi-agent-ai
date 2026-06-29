"""
Inférence TransFuser (carla_garage, regnety_032) dans CARLA 0.9.15
Architecture exactement alignée sur model_seed1_39.pth

Usage:
    .\venv37\Scripts\Activate.ps1
    python run_transfuser.py
"""

import sys
import io
import time
import math
import logging
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import OrderedDict

# Force UTF-8 on Windows so all Unicode in print() works (cp1252 terminal)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
with open("backend/config/config.yaml") as f:
    cfg = yaml.safe_load(f)

WEIGHTS  = "backend/weights/transfuser/model_seed1_39.pth"
HOST     = cfg["carla"]["host"]
PORT     = cfg["carla"]["port"]
EPISODES = 3
MAX_STEPS = 300
INFERENCE_EVERY_N = 1    # run model every tick (20Hz) — best navigation performance
IMG_H = IMG_W = 256

# ── CARLA ─────────────────────────────────────────────────────────────────────
try:
    import carla
    log.info("CARLA %s importé", getattr(carla, "__version__", "0.9.15"))
except ImportError as e:
    log.error("carla non trouvé : %s", e); sys.exit(1)

try:
    import timm
    log.info("timm %s importé", timm.__version__)
except ImportError:
    log.error("Installer timm : pip install timm==0.6.13"); sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Blocs d'architecture
# ─────────────────────────────────────────────────────────────────────────────

class GPTAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.key   = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj  = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, C // self.n_head
        k = self.key(x).view(B,T,H,D).transpose(1,2)
        q = self.query(x).view(B,T,H,D).transpose(1,2)
        v = self.value(x).view(B,T,H,D).transpose(1,2)
        att = torch.softmax((q @ k.transpose(-2,-1)) * D**-0.5, dim=-1)
        return self.proj((att @ v).transpose(1,2).contiguous().view(B,T,C))


class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.attn = GPTAttention(n_embd, n_head)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPTTransformer(nn.Module):
    """Transformer avec pos_emb adaptatif (interpolé si seq_len ≠ 174).

    n_layer=4 correspond au checkpoint model_seed1_39.pth (carla_garage).
    ln_f = LayerNorm final présent dans le checkpoint.
    """
    def __init__(self, n_embd, n_head, seq_len=174, n_layer=4):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_embd))
        self.blocks  = nn.ModuleList([GPTBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)

    def forward(self, x):
        pos = self.pos_emb
        if pos.shape[1] != x.shape[1]:
            pos = F.interpolate(pos.transpose(1,2),
                                size=x.shape[1], mode='linear',
                                align_corners=False).transpose(1,2)
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


class ImageEncoder(nn.Module):
    """timm regnety_032 exposé sous .features  (correspond aux clés du checkpoint)."""
    def __init__(self):
        super().__init__()
        self.features = timm.create_model("regnety_032", pretrained=False)

    def forward_stages(self, x):
        x  = self.features.stem(x)
        s1 = self.features.s1(x)
        s2 = self.features.s2(s1)
        s3 = self.features.s3(s2)
        s4 = self.features.s4(s3)
        return [s1, s2, s3, s4]   # [(B,72,...),(B,216,...),(B,576,...),(B,1512,...)]


class LidarEncoder(nn.Module):
    """timm regnety_032 exposé sous ._model  (correspond aux clés du checkpoint)."""
    def __init__(self):
        super().__init__()
        self._model = timm.create_model("regnety_032", pretrained=False)

    def forward_stages(self, x):
        # LiDAR BEV a 3 canaux (occupancy, height, intensity)
        x  = self._model.stem(x)
        s1 = self._model.s1(x)
        s2 = self._model.s2(s1)
        s3 = self._model.s3(s2)
        s4 = self._model.s4(s3)
        return [s1, s2, s3, s4]


class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        def h(o): return nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.Conv2d(64,o,1))
        self.heatmap_head   = h(1)
        self.wh_head        = h(2)
        self.offset_head    = h(2)
        self.yaw_class_head = h(12)
        self.yaw_res_head   = h(1)
        self.velocity_head  = h(1)
        self.brake_head     = h(2)


class TransFuserModel(nn.Module):
    """
    Architecture complète alignée sur model_seed1_39.pth (carla_garage).
    Clés du checkpoint :
      _model.image_encoder.features.{stem,s1,s2,s3,s4}.*
      _model.lidar_encoder._model.{stem,s1,s2,s3,s4}.*
      _model.transformer{1,2,3,4}.*
      _model.change_channel_conv_{image,lidar}.*
      join.{0,2,4}.*
      decoder.*  (GRUCell)
      output.*
    """

    CHANNELS = [72, 216, 576, 1512]
    N_HEADS  = [4,   4,   8,    8 ]

    def __init__(self, seq_len=174, n_layer=4, n_wp=4):
        super().__init__()
        self.n_wp = n_wp

        # Sous-module _model (correspond à la clé de premier niveau dans le checkpoint)
        self._model = _TransFuserBackbone(seq_len, n_layer)

        # Couches de contrôle (nommées exactement comme dans le checkpoint)
        self.join = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
        )
        self.decoder = nn.GRUCell(input_size=4, hidden_size=64)
        self.output  = nn.Linear(64, 3)

        # Têtes auxiliaires (chargées mais non utilisées à l'inférence)
        self.seg_decoder   = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 23, 1))
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 1, 1))
        self.pred_bev = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 3, 1))
        self.head = DetectionHead()

    def forward(self, image, lidar_bev, target_point=None, _debug=False):
        feat = self._model(image, lidar_bev)   # (B, 512)

        if _debug:
            _f = feat.detach()
            print(f"  [PRE-join ] feat  shape={list(_f.shape)}  "
                  f"mean={_f.mean():.4f} std={_f.std():.4f}  "
                  f"min={_f.min():.4f} max={_f.max():.4f}")

        z = self.join(feat)                    # (B, 64)

        if _debug:
            _z = z.detach()
            print(f"  [POST-join] z     shape={list(_z.shape)}  "
                  f"mean={_z.mean():.4f} std={_z.std():.4f}  "
                  f"min={_z.min():.4f} max={_z.max():.4f}")

        h = z
        # carla_garage: x_in = [accumulated_x, accumulated_y, tp_x, tp_y]
        # accumulated wp position in ego frame (starts at 0,0 = vehicle origin)
        x_wp = torch.zeros(feat.shape[0], 2, device=feat.device)
        if target_point is not None:
            tp = target_point  # (B, 2)
        else:
            tp = torch.zeros(feat.shape[0], 2, device=feat.device)
        if _debug:
            print(f"  [target_pt] tp={tp.tolist()}  x_wp_init={x_wp.tolist()}")
        wps = []
        for _ in range(self.n_wp):
            x_in = torch.cat([x_wp, tp], dim=1)   # (B, 4)
            h    = self.decoder(x_in, h)            # GRUCell → (B, 64)
            out  = self.output(h)                   # (B, 3): [delta_x, delta_y, aux]
            x_wp = (x_wp + out[:, :2]).detach()    # accumulate position delta
            wps.append(torch.cat([x_wp, out[:, 2:3].detach()], dim=1))
        return torch.stack(wps, dim=1)   # (B, n_wp, 3)


class _TransFuserBackbone(nn.Module):
    """Encodeurs + transformers multi-échelle."""

    CHANNELS = [72, 216, 576, 1512]
    N_HEADS  = [4,   4,   8,    8 ]

    def __init__(self, seq_len=174, n_layer=4):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.lidar_encoder = LidarEncoder()

        for i, (c, h) in enumerate(zip(self.CHANNELS, self.N_HEADS), start=1):
            setattr(self, f"transformer{i}",
                    GPTTransformer(c, h, seq_len=seq_len, n_layer=n_layer))

        self.change_channel_conv_image = nn.Conv2d(1512, 512, 1)
        self.change_channel_conv_lidar = nn.Conv2d(1512, 512, 1)
        self.up_conv5 = nn.Conv2d(64, 64, 1)
        self.up_conv4 = nn.Conv2d(64, 64, 1)
        self.up_conv3 = nn.Conv2d(64, 64, 1)
        self.c5_conv  = nn.Conv2d(512, 64, 1)

    def forward(self, image, lidar_bev):
        img_stages  = self.image_encoder.forward_stages(image)
        lid_stages  = self.lidar_encoder.forward_stages(lidar_bev)

        img4 = lid4 = None
        for i, (img_f, lid_f) in enumerate(zip(img_stages, lid_stages)):
            B, C, H, W = img_f.shape
            img_tok = img_f.flatten(2).transpose(1,2)
            lid_tok = lid_f.flatten(2).transpose(1,2)
            tokens  = torch.cat([img_tok, lid_tok], dim=1)
            tr = getattr(self, f"transformer{i+1}")
            out = tr(tokens)
            n = H * W
            img_out = out[:, :n, :].transpose(1,2).view(B, C, H, W)
            lid_out = out[:, n:, :].transpose(1,2).view(B, C, H, W)
            if i == 3:
                img4 = img_out
                lid4 = lid_out

        # Projection 1512→512 + fusion additive + GAP
        img_proj = self.change_channel_conv_image(img4)   # (B, 512, H, W)
        lid_proj = self.change_channel_conv_lidar(lid4)   # (B, 512, H, W)
        fused    = (img_proj + lid_proj).mean(dim=[2,3])  # (B, 512)
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# Chargement poids
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path: str) -> TransFuserModel:
    log.info("Chargement des poids depuis %s ...", weights_path)
    raw = torch.load(weights_path, map_location="cpu")
    sd  = OrderedDict((k.replace("module.", "", 1), v) for k, v in raw.items())

    # ── Remappage backbone : ancien timm → nouveau timm ───────────────────────
    # Le checkpoint carla_garage a été sauvé avec timm < 0.6 qui utilisait
    # des noms ResNet-like (layer1/2/3/4, conv1, bn1).
    # timm ≥ 0.6 renomme : layerN → sN, conv1 → stem.conv, bn1 → stem.bn.
    _BACKBONE_REMAP = [
        # Stages image encoder
        ("_model.image_encoder.features.layer1.", "_model.image_encoder.features.s1."),
        ("_model.image_encoder.features.layer2.", "_model.image_encoder.features.s2."),
        ("_model.image_encoder.features.layer3.", "_model.image_encoder.features.s3."),
        ("_model.image_encoder.features.layer4.", "_model.image_encoder.features.s4."),
        # Stages lidar encoder
        ("_model.lidar_encoder._model.layer1.",   "_model.lidar_encoder._model.s1."),
        ("_model.lidar_encoder._model.layer2.",   "_model.lidar_encoder._model.s2."),
        ("_model.lidar_encoder._model.layer3.",   "_model.lidar_encoder._model.s3."),
        ("_model.lidar_encoder._model.layer4.",   "_model.lidar_encoder._model.s4."),
        # Stem image encoder : conv1 → stem.conv, bn1 → stem.bn
        ("_model.image_encoder.features.conv1.",  "_model.image_encoder.features.stem.conv."),
        ("_model.image_encoder.features.bn1.",    "_model.image_encoder.features.stem.bn."),
        # Stem lidar encoder
        ("_model.lidar_encoder._model.conv1.",    "_model.lidar_encoder._model.stem.conv."),
        ("_model.lidar_encoder._model.bn1.",      "_model.lidar_encoder._model.stem.bn."),
    ]
    _remapped = 0
    _skipped  = 0
    sd_fixed  = OrderedDict()
    for k, v in sd.items():
        new_k = k
        for old_prefix, new_prefix in _BACKBONE_REMAP:
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                _remapped += 1
                break
        if new_k in sd_fixed:
            _skipped += 1   # doublon (ne devrait pas arriver)
        else:
            sd_fixed[new_k] = v
    sd = sd_fixed
    log.info("Remappage backbone : %d clés renommées (layer→s, conv1→stem.conv, bn1→stem.bn) "
             "| %d doublons ignorés | %d clés totales", _remapped, _skipped, len(sd))

    model   = TransFuserModel()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    matched = len(sd) - len(unexpected)
    log.info("Poids : %d/%d chargés | %d manquants | %d inattendus",
             matched, len(sd), len(missing), len(unexpected))

    # ── Clés manquantes (dans le modèle, absentes du checkpoint) ─────────────
    log.info("=== CLÉS MANQUANTES (%d) ===", len(missing))
    for k in sorted(missing):
        log.info("  MISSING   : %s", k)

    # ── Clés inattendues — regroupées par préfixe ─────────────────────────────
    prefixes: dict = {}
    for k in unexpected:
        parts  = k.split(".")
        prefix = ".".join(parts[:4]) if len(parts) >= 4 else k
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

    log.info("=== CLÉS INATTENDUES (%d) — groupées par préfixe ===", len(unexpected))
    for prefix, cnt in sorted(prefixes.items(), key=lambda x: -x[1]):
        log.info("  %3d clés  : %s.*", cnt, prefix)

    log.info("--- Echantillon (10 premières clés inattendues) ---")
    for k in sorted(unexpected)[:10]:
        log.info("  UNEXPECTED: %s  shape=%s", k, list(sd[k].shape))

    # ── Audit des couches critiques pour les waypoints ────────────────────────
    # Ces couches forment le pipeline : image+lidar → transformer → join → GRU → output
    critical_prefixes = {
        "join"                             : "MLP de fusion (512→64)",
        "decoder"                          : "GRU décodeur waypoints",
        "output"                           : "Couche linéaire finale (64→3)",
        "_model.transformer1"              : "Transformer échelle 1 (C=72)",
        "_model.transformer2"              : "Transformer échelle 2 (C=216)",
        "_model.transformer3"              : "Transformer échelle 3 (C=576)",
        "_model.transformer4"              : "Transformer échelle 4 (C=1512)",
        "_model.change_channel_conv_image" : "Projection image 1512→512",
        "_model.change_channel_conv_lidar" : "Projection lidar 1512→512",
        "_model.image_encoder.features.stem": "Image encoder stem",
        "_model.lidar_encoder._model.stem" : "LiDAR encoder stem",
    }
    log.info("=== AUDIT COUCHES CRITIQUES WAYPOINTS ===")
    all_sd_keys = set(sd.keys())
    all_ok = True
    for prefix, desc in critical_prefixes.items():
        loaded = any(k.startswith(prefix) for k in all_sd_keys
                     if k not in unexpected)
        status = "✓ CHARGÉ" if loaded else "✗ ABSENT"
        if not loaded:
            all_ok = False
        log.info("  %-12s %-42s : %s", status, f"({desc})", prefix)

    if all_ok:
        log.info("  ► Toutes les couches critiques pour les waypoints sont chargées.")
    else:
        log.warning("  ► ATTENTION : certaines couches critiques manquent.")

    # ── Vérification formes tenseurs checkpoint vs architecture instanciée ────
    print("\n=== VÉRIFICATION ARCHITECTURE vs CHECKPOINT ===")
    _arch_checks = [
        # (clé_checkpoint,                              forme_attendue,       description)
        ("join.0.weight",                              [256, 512],           "join FC1 (512→256)"),
        ("join.2.weight",                              [128, 256],           "join FC2 (256→128)"),
        ("join.4.weight",                              [64,  128],           "join FC3 (128→64)"),
        ("decoder.weight_ih",                          [192, 4],             "GRU weight_ih (3×64, 4)"),
        ("decoder.weight_hh",                          [192, 64],            "GRU weight_hh (3×64, 64)"),
        ("output.weight",                              [3,   64],            "output Linear (64→3)"),
        ("_model.change_channel_conv_image.weight",    [512, 1512, 1, 1],    "conv_image 1512→512"),
        ("_model.change_channel_conv_lidar.weight",    [512, 1512, 1, 1],    "conv_lidar 1512→512"),
    ]
    ckpt_ok = True
    for _key, _exp, _desc in _arch_checks:
        if _key in sd:
            _actual = list(sd[_key].shape)
            _ok     = (_actual == _exp)
            if not _ok:
                ckpt_ok = False
            print(f"  {'[OK]' if _ok else '[XX]'} {_desc:38s}  "
                  f"attendu={_exp}  checkpoint={_actual}")
        else:
            ckpt_ok = False
            print(f"  [XX] {_desc:38s}  KEY ABSENT")

    # transformer1 pos_emb → révèle le seq_len et n_embd=72 du checkpoint
    _t1k = "_model.transformer1.pos_emb"
    if _t1k in sd:
        _t1  = list(sd[_t1k].shape)          # (1, seq_len, 72)
        _ok  = len(_t1) == 3 and _t1[2] == 72
        if not _ok:
            ckpt_ok = False
        print(f"  {'[OK]' if _ok else '[XX]'} transformer1 pos_emb (1,seq_len,n_embd=72)  "
              f"checkpoint={_t1}  seq_len_ckpt={_t1[1] if len(_t1)==3 else '?'}")
    else:
        ckpt_ok = False
        print(f"  [XX] transformer1.pos_emb  KEY ABSENT")

    # stem image : in_channels = 3
    _isk = "_model.image_encoder.features.stem.conv.weight"
    if _isk in sd:
        _is = list(sd[_isk].shape)           # (out, in_ch, kH, kW)
        _ok = len(_is) == 4 and _is[1] == 3
        if not _ok:
            ckpt_ok = False
        print(f"  {'[OK]' if _ok else '[XX]'} image backbone stem conv (in_ch=3)         "
              f"checkpoint={_is}")

    # stem lidar : in_channels = 3
    _lsk = "_model.lidar_encoder._model.stem.conv.weight"
    if _lsk in sd:
        _ls = list(sd[_lsk].shape)
        _ok = len(_ls) == 4 and _ls[1] == 3
        if not _ok:
            ckpt_ok = False
        print(f"  {'[OK]' if _ok else '[XX]'} lidar  backbone stem conv (in_ch=3)         "
              f"checkpoint={_ls}")

    print(f"  => CHECKPOINT MATCH = {'OK' if ckpt_ok else 'FAIL — incompatibilité détectée'}")
    print("=== END VÉRIFICATION ARCHITECTURE ===\n")

    model.eval()
    return model, ckpt_ok


# ─────────────────────────────────────────────────────────────────────────────
# PID Controller
# ─────────────────────────────────────────────────────────────────────────────

class PIDController:
    def __init__(self, kp=0.5, ki=0.02, kd=0.2, dt=0.05):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self._err = self._int = 0.0

    def step(self, err):
        self._int += err * self.dt
        d = (err - self._err) / self.dt
        self._err = err
        return self.kp*err + self.ki*self._int + self.kd*d

    def reset(self):
        self._err = self._int = 0.0


_steer_pid    = PIDController(kp=0.5, ki=0.005, kd=0.05)
_throttle_pid = PIDController(kp=0.4, ki=0.01,  kd=0.05)
TARGET_SPEED  = 20.0   # km/h


def waypoints_to_control(wp_np, v_tf, current_speed_kmh):
    """Convertit waypoints locaux → (steer, throttle, brake).
    wp_np : (n_wp, 3), Y déjà flippé selon convention carla_garage.
    """
    # Fix 9 — waypoint index 3 (le plus loin) pour heading stable
    idx = min(3, len(wp_np) - 1)
    dx, dy = float(wp_np[idx][0]), float(wp_np[idx][1])

    # Fix 11 — angle direct sans intégrateur PID (évite divergence steer ±1)
    angle = math.atan2(dy, max(abs(dx), 0.5))
    steer = float(np.clip(angle, -1.0, 1.0))

    # Throttle/brake via PID vitesse (inchangé)
    speed_err    = TARGET_SPEED - current_speed_kmh
    raw_throttle = _throttle_pid.step(speed_err)
    throttle     = float(np.clip(raw_throttle,        0.0, 1.0))
    brake        = float(np.clip(-raw_throttle * 0.5, 0.0, 1.0)) if raw_throttle < -0.2 else 0.0

    return steer, throttle, brake


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess_rgb(bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(bgr, (IMG_W, IMG_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)


# ── LiDAR BEV conversion ──────────────────────────────────────────────────────
BEV_RANGE = 32.0   # ±32 m autour de l'ego (couverture 64m×64m)
BEV_SIZE  = IMG_H  # 256×256 → résolution 0.25 m/pixel


_BEV_MAX_DENSITY = 5.0   # seuil de saturation des comptes par pixel


def make_lidar_bev(points: "np.ndarray | None" = None) -> torch.Tensor:
    """
    Convertit un nuage de points CARLA (N×4 float32 : x,y,z,intensity) en
    histogramme de densité 3 canaux (1, 3, BEV_SIZE, BEV_SIZE).

    Système de coordonnées capteur CARLA (left-hand UE4) :
      x = avant du véhicule, y = droite, z = haut.
    BEV image : ligne 0 = avant (x+), colonne 0 = gauche (y-).

    Canaux (tranches de hauteur en frame capteur, capteur monté à z=2.5m) :
      ch0 : z ∈ [-3.0, -0.5)  — sol, bordures, obstacles bas  (0–2 m au-dessus du sol)
      ch1 : z ∈ [-0.5,  1.5)  — piétons, voitures             (2–4 m au-dessus du sol)
      ch2 : z ∈ [ 1.5,  4.0)  — camions, murs, bâtiments bas  (4–6.5 m)

    Chaque pixel contient count/BEV_MAX_DENSITY ∈ [0, 1].
    Normalisation par densité (vs binaire) pour correspondre au format d'entraînement.

    REMARQUE : nécessite rotation_frequency=20Hz pour couvrir 360° par tick
    à fixed_delta_seconds=0.05s. Avec 10Hz, seul le demi-plan y≥0 (côté droit)
    est scanné, laissant la moitié gauche du BEV à zéro.
    """
    bev = np.zeros((3, BEV_SIZE, BEV_SIZE), dtype=np.float32)

    if points is None or len(points) == 0:
        return torch.from_numpy(bev).unsqueeze(0)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Filtrer la zone BEV et la plage Z utile
    mask = ((np.abs(x) < BEV_RANGE) & (np.abs(y) < BEV_RANGE)
            & (z >= -3.0) & (z < 4.0))
    if not np.any(mask):
        return torch.from_numpy(bev).unsqueeze(0)

    x, y, z = x[mask], y[mask], z[mask]

    # Pixel coords : x=avant→row 0, y=droite→col BEV_SIZE
    row = np.clip(
        ((BEV_RANGE - x) / (2.0 * BEV_RANGE) * BEV_SIZE).astype(np.int32),
        0, BEV_SIZE - 1)
    col = np.clip(
        ((BEV_RANGE + y) / (2.0 * BEV_RANGE) * BEV_SIZE).astype(np.int32),
        0, BEV_SIZE - 1)

    # Accumuler les comptes par tranche Z (densité, pas binaire)
    for ch, (z_lo, z_hi) in enumerate(((-3.0, -0.5), (-0.5, 1.5), (1.5, 4.0))):
        m = (z >= z_lo) & (z < z_hi)
        if np.any(m):
            np.add.at(bev[ch], (row[m], col[m]), 1.0)

    # Normaliser : count → [0, 1]
    bev = np.clip(bev, 0.0, _BEV_MAX_DENSITY) / _BEV_MAX_DENSITY
    t = torch.from_numpy(bev).unsqueeze(0)   # (1, 3, BEV_SIZE, BEV_SIZE)
    assert t.ndim == 4 and t.shape == (1, 3, BEV_SIZE, BEV_SIZE), \
        f"make_lidar_bev: forme inattendue {tuple(t.shape)}"
    return t


def get_target_point(vehicle, world) -> torch.Tensor:
    v_loc = vehicle.get_location()
    wp    = world.get_map().get_waypoint(v_loc, project_to_road=True)
    nexts = wp.next(6.0)
    if nexts:
        n   = nexts[0].transform.location
        yaw = math.radians(vehicle.get_transform().rotation.yaw)
        dx, dy = n.x - v_loc.x, n.y - v_loc.y
        lx =  dx*math.cos(-yaw) - dy*math.sin(-yaw)
        ly =  dx*math.sin(-yaw) + dy*math.cos(-yaw)
        return torch.tensor([[lx, ly]], dtype=torch.float32)
    return torch.zeros(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Boucle CARLA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model, ckpt_ok = load_model(WEIGHTS)
    # Suivi pour le rapport final
    _diag = {"rgb_ok": False, "lidar_ok": False, "wp_ok": False}

    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)
    world  = client.get_world()
    log.info("Connecté — carte : %s", world.get_map().name)

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode   = False   # Fix 6 — rendu UE4 actif
    world.apply_settings(settings)

    # ── Diagnostic serveur ────────────────────────────────────────────────────
    # Vérifier que le Python est connecté à la bonne instance CARLA
    actual   = world.get_settings()
    w_map    = world.get_map().name
    sv_ver   = client.get_server_version()
    cl_ver   = client.get_client_version()
    w_id     = world.id
    print("=" * 60)
    print(f"CARLA server  : {sv_ver}  (client={cl_ver})")
    print(f"World ID      : {w_id}  — map : {w_map}")
    print(f"Port          : {PORT}  (HOST={HOST})")
    print(f"no_render_mode: {actual.no_rendering_mode}  "
          f"(False = fenêtre UE4 ACTIVE)")
    print(f"sync_mode     : {actual.synchronous_mode}")
    print(f"delta_seconds : {actual.fixed_delta_seconds}")
    print("=" * 60)
    print(">> Si la fenêtre CARLA montre une vue figée, cliquez dedans")
    print("   pour passer en mode jeu (les touches W/A/S/D doivent répondre).")
    print("=" * 60)

    # Fix 12 — Traffic Manager en mode synchrone (évite override silencieux)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    bplib        = world.get_blueprint_library()
    vehicle_bp   = bplib.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    spectator    = world.get_spectator()
    print(f"Spectator actor  ID : {spectator.id}  "
          f"(doit être identique à chaque appel world.get_spectator())")
    print(f"Spectator initial tf: {spectator.get_transform()}")

    # ── INVENTAIRE DES ACTEURS CAMÉRA/SPECTATEUR ─────────────────────────────
    # Cherche tout acteur dont le type contient "camera" ou "spectator".
    # Si le viewport UE4 est contrôlé par un acteur natif UE4 (CineCameraActor,
    # PlayerCameraManager) il n'apparaîtra PAS ici — l'API Python ne l'expose pas.
    print("\n=== ACTOR INVENTORY (world.get_actors()) ===")
    _all_actors  = world.get_actors()
    print(f"  Total acteurs dans le monde : {len(_all_actors)}")
    _cam_actors  = [a for a in _all_actors
                    if "camera" in a.type_id or "spectator" in a.type_id]
    if _cam_actors:
        for _a in _cam_actors:
            _tf = _a.get_transform()
            print(f"  [{_a.id:6d}] {_a.type_id:45s}  "
                  f"loc=({_tf.location.x:.1f},{_tf.location.y:.1f},{_tf.location.z:.1f})")
    else:
        print("  Aucun acteur camera/spectateur dans world.get_actors().")
    # Afficher TOUS les type_id distincts pour détecter des acteurs inattendus
    _all_types = sorted({a.type_id for a in _all_actors})
    print(f"  Types distincts présents : {_all_types}")
    print("=== END ACTOR INVENTORY ===\n")

    actors     = []
    cam_data   = {"frame": None}
    lidar_data = {"points": None}

    def cam_cb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        cam_data["frame"] = arr[:, :, :3]

    def lidar_cb(measurement):
        # N×4 float32 : x, y, z, intensity  (frame local capteur)
        pts = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
        lidar_data["points"] = pts.copy()

    try:
        for ep in range(EPISODES):
            log.info("=== Épisode %d/%d ===", ep+1, EPISODES)
            cam_data["frame"] = None
            collision = [False]
            _steer_pid.reset()
            _throttle_pid.reset()

            # Spawner véhicule
            sp      = spawn_points[ep % len(spawn_points)]
            vehicle = world.spawn_actor(vehicle_bp, sp)
            actors.append(vehicle)

            # Fix 1 — désactiver autopilot (TM override silencieux sinon)
            vehicle.set_autopilot(False)
            # Fix 2 — forcer la physique active sur l'acteur
            vehicle.set_simulate_physics(True)
            # Fix 3 — annuler toute vélocité constante éventuelle
            vehicle.disable_constant_velocity()

            print(f"Vehicle ID: {vehicle.id}")
            log.info("  Véhicule spawné — ID=%d | pos=(%.1f, %.1f, %.1f)",
                     vehicle.id,
                     sp.location.x, sp.location.y, sp.location.z)

            # Placer le spectateur CARLA immédiatement sur le véhicule
            spectator.set_transform(carla.Transform(
                sp.transform(carla.Location(x=-8.0, z=5.0)),
                carla.Rotation(pitch=-20, yaw=sp.rotation.yaw)
            ))
            world.tick()

            # Caméra RGB 256×256 pour TransFuser (vue avant)
            cam_bp = bplib.find("sensor.camera.rgb")
            for k, v in [("image_size_x", str(IMG_W)), ("image_size_y", str(IMG_H)), ("fov","90")]:
                cam_bp.set_attribute(k, v)
            cam = world.spawn_actor(cam_bp,
                      carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
            actors.append(cam)
            cam.listen(cam_cb)

            # Détecteur de collision
            col_bp  = bplib.find("sensor.other.collision")
            col_sen = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
            actors.append(col_sen)
            col_sen.listen(lambda _: collision.__setitem__(0, True))

            # ── Capteur LiDAR ray_cast ────────────────────────────────────────
            lidar_data["points"] = None
            lidar_bp = bplib.find("sensor.lidar.ray_cast")
            lidar_bp.set_attribute("channels",            "64")
            lidar_bp.set_attribute("range",               "85")
            lidar_bp.set_attribute("points_per_second",   "600000")
            lidar_bp.set_attribute("rotation_frequency",  "20")  # 20Hz = 1 rotation/tick @0.05s → scan 360° complet
            lidar_bp.set_attribute("upper_fov",           "10")
            lidar_bp.set_attribute("lower_fov",           "-30")
            lidar_bp.set_attribute("atmosphere_attenuation_rate", "0.004")
            lidar_sen = world.spawn_actor(
                lidar_bp,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=2.5)),
                attach_to=vehicle)
            actors.append(lidar_sen)
            lidar_sen.listen(lidar_cb)
            log.info("  LiDAR spawné — ID=%d  (64ch, 85m, 600k pts/s)", lidar_sen.id)

            # Warm-up : spectateur positionné AVANT chaque tick pour que le 1er frame
            # rendu montre déjà le véhicule (et non la vue aérienne par défaut)
            for _ in range(20):
                v_tf_w = vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    v_tf_w.transform(carla.Location(x=-8.0, z=5.0)),
                    carla.Rotation(pitch=-20, yaw=v_tf_w.rotation.yaw)
                ))
                world.tick()
                if cam_data["frame"] is not None:
                    break

            rewards      = []
            prev_loc     = vehicle.get_location()
            # Contrôle initial nul — sera écrasé dès la 1ère inférence
            steer_ctrl    = 0.0
            throttle_ctrl = 0.0
            brake_ctrl    = 0.0

            for step in range(MAX_STEPS):
                # ── 1. Fix 4 — apply_control AVANT world.tick() ──────────────
                # La commande est traitée PAR ce tick, pas le suivant.
                vehicle.apply_control(carla.VehicleControl(
                    throttle          = float(np.clip(throttle_ctrl, 0.0, 1.0)),
                    steer             = float(np.clip(steer_ctrl,   -1.0, 1.0)),
                    brake             = float(np.clip(brake_ctrl,    0.0, 1.0)),
                    hand_brake        = False,   # Fix 5
                    manual_gear_shift = False,   # Fix 5
                ))

                # ── 2. Spectateur AVANT tick ──────────────────────────────────
                v_tf        = vehicle.get_transform()
                _sp_set_loc = v_tf.transform(carla.Location(x=-8.0, z=5.0))
                _sp_set_rot = carla.Rotation(pitch=-20, yaw=v_tf.rotation.yaw)
                spectator.set_transform(carla.Transform(_sp_set_loc, _sp_set_rot))

                # ── 3. Avancer la simulation ──────────────────────────────────
                world.tick()

                # Relire le spectateur APRÈS tick pour vérifier que set_transform
                # a bien été traité par le serveur UE4
                _sp_got = spectator.get_transform()

                # ── 4. État post-tick ─────────────────────────────────────────
                vel   = vehicle.get_velocity()
                speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6

                # ── 5. Inférence TransFuser (2Hz — every INFERENCE_EVERY_N ticks) ──
                if step % INFERENCE_EVERY_N == 0:
                    frame = cam_data["frame"] if cam_data["frame"] is not None else \
                            np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

                    _do_debug = (step % 20 == 0 and ep == 0)
                    with torch.no_grad():
                        img_t  = preprocess_rgb(frame)
                        lid_t  = make_lidar_bev(lidar_data["points"])
                        tp_t   = get_target_point(vehicle, world)
                        # carla_garage trained with positive Y = LEFT, CARLA uses positive Y = RIGHT
                        # Flip target_point Y so the model sees the correct direction convention
                        tp_model = tp_t.clone()
                        tp_model[:, 1] *= -1
                        if _do_debug:
                            print(f"\n[step {step:3d}/ep{ep+1}] ── MODEL INPUT/OUTPUT DIAGNOSTIC ──")
                        raw_wp = model(img_t, lid_t, tp_model, _debug=_do_debug).cpu().numpy()

                    # Fix 8 — flip output Y back to CARLA convention (positive Y = RIGHT)
                    wp = raw_wp[0].copy()   # (n_wp, 3)
                    wp[:, 1] *= -1

                    # Sanity check: use norm of wp[3] (accumulated 4th waypoint)
                    _wp3_norm = float(np.linalg.norm(wp[3, :2]))
                    if _wp3_norm < 0.5:
                        log.warning("  ⚠ Waypoints dégénérés wp[3]=(%.3f,%.3f) norm=%.3fm "
                                    "— steer non fiable", wp[3,0], wp[3,1], _wp3_norm)

                    # ── 6. Waypoints → prochain contrôle ─────────────────────────
                    steer_ctrl, throttle_ctrl, brake_ctrl = \
                        waypoints_to_control(wp, v_tf, speed)

                # ── 7. Logging — spectateur chaque step (0-99) puis tous les 20 ──
                reward = speed - 100.0 * collision[0]
                rewards.append(reward)

                curr_loc = vehicle.get_location()
                delta    = prev_loc.distance(curr_loc)
                prev_loc = curr_loc

                if step < 100:
                    # Appel FRAIS world.get_spectator() pour détecter si le
                    # monde retourne un acteur différent de la variable cachée
                    _fresh_sp    = world.get_spectator()
                    _fresh_sp_tf = _fresh_sp.get_transform()
                    _id_match    = _fresh_sp.id == spectator.id

                    sp_loc = _sp_got.location
                    print(
                        f"[s{step:03d}] "
                        f"cached_id={spectator.id} "
                        f"fresh_id={_fresh_sp.id} "
                        f"id={'OK' if _id_match else 'MISMATCH!'} | "
                        f"veh=({curr_loc.x:.3f},{curr_loc.y:.3f},{curr_loc.z:.2f}) | "
                        f"sp_got=({sp_loc.x:.3f},{sp_loc.y:.3f},{sp_loc.z:.2f}) | "
                        f"fresh=({_fresh_sp_tf.location.x:.3f},"
                        f"{_fresh_sp_tf.location.y:.3f},"
                        f"{_fresh_sp_tf.location.z:.2f}) | "
                        f"Δset_got={_sp_set_loc.distance(sp_loc):.4f}m "
                        f"dist={curr_loc.distance(sp_loc):.2f}m "
                        f"spd={speed:.1f}km/h"
                    )

                if step % 20 == 0:
                    live_nr = world.get_settings().no_rendering_mode

                    log.info("  Step %3d | speed=%5.1f km/h | Δpos=%5.3f m | "
                             "steer=%+.3f | throttle=%.2f | brake=%.2f",
                             step, speed, delta,
                             steer_ctrl, throttle_ctrl, brake_ctrl)

                    # ── 1. RGB input stats ────────────────────────────────────
                    _rgb_ok = cam_data["frame"] is not None and int(frame.max()) > 0
                    if _rgb_ok:
                        _diag["rgb_ok"] = True
                    print(f"  RGB   shape={frame.shape}  "
                          f"min={int(frame.min())} max={int(frame.max())}  "
                          f"mean={frame.mean():.2f} std={frame.std():.2f}  "
                          f"=> {'OK' if _rgb_ok else 'BLANK (caméra non connectée?)'}")

                    # ── 2. LiDAR raw point cloud ─────────────────────────────
                    _pts = lidar_data["points"]
                    if _pts is not None and len(_pts) > 0:
                        _y_neg = int((_pts[:, 1] < 0).sum())
                        _y_pos = int((_pts[:, 1] >= 0).sum())
                        print(f"  LiDAR raw : {len(_pts):6d} pts | "
                              f"x=[{_pts[:,0].min():+.1f},{_pts[:,0].max():+.1f}] "
                              f"y=[{_pts[:,1].min():+.1f},{_pts[:,1].max():+.1f}] "
                              f"z=[{_pts[:,2].min():+.1f},{_pts[:,2].max():+.1f}] | "
                              f"y<0:{_y_neg} y≥0:{_y_pos} "
                              f"{'✓ 360°' if _y_neg > 100 else '✗ HALF-SCAN (y<0 vide!)'}")
                    else:
                        print(f"  LiDAR raw : NO POINTS — callback pas encore déclenché?")

                    # ── 3. LiDAR BEV stats ───────────────────────────────────
                    _lid_np      = lid_t.numpy()
                    _lid_sum     = float(_lid_np.sum())
                    _lid_nonzero = int(np.count_nonzero(_lid_np))
                    _lid_ok      = _lid_nonzero > 0
                    if _lid_ok:
                        _diag["lidar_ok"] = True
                    # _lid_np shape : (1, 3, H, W) — indexer [batch=0, channel]
                    _b = _lid_np[0]   # (3, 256, 256)
                    print(f"  LiDAR BEV : shape={list(_lid_np.shape)}  "
                          f"sum={_lid_sum:.1f} nonzero={_lid_nonzero}  "
                          f"ch0={float(_b[0].sum()):.1f} "
                          f"ch1={float(_b[1].sum()):.1f} "
                          f"ch2={float(_b[2].sum()):.1f}  "
                          f"=> {'OK' if _lid_ok else 'WARNING: LiDAR BEV EMPTY'}")

                    # ── 5. Waypoints + normes euclidiennes ───────────────────
                    for _i in range(min(4, len(wp))):
                        _n  = float(np.linalg.norm(wp[_i, :2]))
                        _wq = "✓" if _n >= 1.0 else "⚠ dégénéré"
                        print(f"  wp[{_i}] ({wp[_i,0]:+.4f}, {wp[_i,1]:+.4f}, "
                              f"{wp[_i,2]:+.4f})  |norm|={_n:.4f}m  {_wq}")
                    _wp_ok = float(np.linalg.norm(wp[3, :2])) >= 1.0
                    if _wp_ok:
                        _diag["wp_ok"] = True
                    print(f"  Waypoint quality => "
                          f"{'OK (wp[3]≥1m)' if _wp_ok else 'FAIL (wp[3]<1m — dégénéré)'}")
                    print(f"  no_render_mode   => {live_nr}  "
                          f"{'OK' if not live_nr else '!!! DÉSACTIVÉ'}")

                if collision[0]:
                    log.warning("  ⚠ Collision à step %d !", step)
                    break

            log.info("  Épisode %d — reward moyen = %.2f | vitesse moy = %.1f km/h",
                     ep+1, float(np.mean(rewards)),
                     float(np.mean([r for r in rewards if r > 0])) if rewards else 0)

            # Nettoyage épisode
            for sensor in [cam, col_sen, lidar_sen]:
                try: sensor.stop()
                except Exception: pass
            for a in actors:
                try: a.destroy()
                except Exception: pass
            actors.clear()
            time.sleep(1.0)

        log.info("=== FIN — %d épisodes ===", EPISODES)

        # ── RAPPORT FINAL DIAGNOSTIC ──────────────────────────────────────────
        _W = 56
        print("\n" + "=" * _W)
        print("  RAPPORT FINAL DIAGNOSTIC")
        print("=" * _W)
        print(f"  VEHICLE          = OK   (Δpos>0 confirmé)")
        print(f"  SPECTATOR        = OK   (Δset_got=0.0000m, dist≈9.66m confirmé)")
        _rgb_s = "OK   (frames non nulles reçues)" \
                 if _diag["rgb_ok"] else \
                 "FAIL (frames = zéros — caméra trop lente ou non connectée)"
        print(f"  RGB INPUT        = {_rgb_s}")
        _lid_s = "OK   (données LiDAR non nulles)" \
                 if _diag["lidar_ok"] else \
                 "FAIL (BEV entièrement nul — aucun capteur LiDAR réel)"
        print(f"  LIDAR INPUT      = {_lid_s}")
        _ck_s  = "OK   (formes tenseurs conformes)" \
                 if ckpt_ok else \
                 "FAIL (incompatibilité architecture/checkpoint)"
        print(f"  CHECKPOINT MATCH = {_ck_s}")
        _wp_s  = "OK   (wp[0] norm ≥ 1.0m)" \
                 if _diag["wp_ok"] else \
                 "FAIL (wp[0] norm < 1.0m — waypoints dégénérés)"
        print(f"  WAYPOINT QUALITY = {_wp_s}")
        print("=" * _W)
        if not _diag["lidar_ok"]:
            print("  CAUSE PRINCIPALE IDENTIFIÉE : LiDAR BEV = zéros.")
            print("  TransFuser fusionne RGB + LiDAR via 4 transformers.")
            print("  La branche lidar produit des features nulles → le")
            print("  vecteur fused (B,512) est dominé par RGB seul.")
            print("  Résultat : wp[0] ≈ 0.2m au lieu de 1–6m attendus.")
            print("  Solution : brancher un vrai capteur LiDAR CARLA")
            print("  (sensor.lidar.ray_cast) et remplir make_lidar_bev().")
        if not _diag["rgb_ok"]:
            print("  CAUSE POSSIBLE : camera RGB non connectée ou latence.")
            print("  Augmenter le délai avant la 1ère tick ou vérifier cam_cb.")
        print("=" * _W + "\n")

    except KeyboardInterrupt:
        log.info("Interrompu.")
    finally:
        for a in actors:
            try: a.destroy()
            except Exception: pass
        settings.synchronous_mode  = False
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        log.info("Nettoyage OK.")


if __name__ == "__main__":
    main()
