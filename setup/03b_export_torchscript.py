"""
Export XFeat backbone and LighterGlue as TorchScript .pt models.

XFeat approach: export ONLY the convolutional backbone (fixed output shape),
then do NMS + descriptor sampling in C++ using CUDA kernels.
This avoids the torch.jit.trace dynamic-shape problem in XFeat's NMS function.

Backbone output:
  M1:  [1, 64, H/8, W/8]  — descriptor feature maps
  K1h: [1,  1,   H,   W]  — keypoint heatmap (after get_kpts_heatmap)
  H1:  [1,  1, H/8, W/8]  — reliability score map

C++ side (xfeat_extractor.cu) runs: ANMS on K1h, then grid_sample on M1.

LighterGlue: traced with the correct interface (data dict -> unwrapped via wrapper).

Run: python setup/03b_export_torchscript.py
Out: models/xfeat.pt        (backbone only)
     models/lighterglue.pt
"""

import sys, os, tempfile, subprocess
from pathlib import Path

ROOT   = Path(__file__).parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

TMP    = Path(tempfile.gettempdir())
AF_DIR = TMP / "accelerated_features"

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])

print("Installing deps...")
pip_install("kornia", "einops")

if not AF_DIR.exists():
    print("Cloning accelerated_features...")
    subprocess.check_call(["git", "clone", "--depth", "1",
        "https://github.com/verlab/accelerated_features.git", str(AF_DIR)])
    (AF_DIR / "setup.py").write_text(
        "from setuptools import setup, find_packages\n"
        "setup(name='accelerated_features', version='1.0', packages=['modules'])\n"
    )

pip_install("-e", str(AF_DIR), "--quiet")

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(AF_DIR))
from modules.xfeat      import XFeat
from modules.lighterglue import LighterGlue as LighterGlueModel

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# XFeat's preprocess_tensor does (H//32)*32, (W//32)*32 internally.
# KITTI seq 00: 1241x376 -> internal size 1216x352  (1241//32=38*32=1216, 376//32=11*32=352)
IMG_W_KITTI  = 1241
IMG_H_KITTI  = 376
IMG_W_TRACE  = (IMG_W_KITTI // 32) * 32   # 1216
IMG_H_TRACE  = (IMG_H_KITTI // 32) * 32   # 352

print(f"Device: {DEVICE}")
print(f"KITTI image size: {IMG_W_KITTI}x{IMG_H_KITTI}")
print(f"Backbone trace size (after preprocess_tensor): {IMG_W_TRACE}x{IMG_H_TRACE}")

# ── 1. XFeat Backbone TorchScript ────────────────────────────────────────────
print("\n=== Exporting XFeat backbone to TorchScript ===")

class XFeatBackbone(nn.Module):
    """
    Export ONLY the backbone + heatmap conversion.
    Input:  (1, 1, H, W) float32 in [0,1], H and W must be multiples of 32
            (use (H//32)*32 x (W//32)*32 — same as XFeat.preprocess_tensor)
    Output: (M1, K1h, H1)
       M1:  (1, 64, H/8, W/8)  descriptor feature map (L2-normalised)
       K1h: (1,  1,    H,   W) keypoint heatmap
       H1:  (1,  1, H/8, W/8)  reliability map
    """
    def __init__(self):
        super().__init__()
        xf = XFeat()
        self.net = xf.net

    def forward(self, x: torch.Tensor):
        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)
        # Convert K1 logits to heatmap (same as XFeat.get_kpts_heatmap)
        scores = F.softmax(K1 * 1.0, dim=1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return M1, heatmap, H1

backbone = XFeatBackbone().to(DEVICE).eval()

# Trace with the exact backbone input size (no NMS — fixed output shapes)
torch.manual_seed(42)
dummy_img = torch.rand(1, 1, IMG_H_TRACE, IMG_W_TRACE, device=DEVICE)

print(f"  Tracing backbone at {IMG_W_TRACE}x{IMG_H_TRACE}...")
with torch.no_grad():
    M1_t, K1h_t, H1_t = backbone(dummy_img)
print(f"  M1: {M1_t.shape}, K1h: {K1h_t.shape}, H1: {H1_t.shape}")

xfeat_out = str(MODELS / "xfeat.pt")
with torch.no_grad():
    traced = torch.jit.trace(backbone, dummy_img, strict=False)
traced.save(xfeat_out)
sz = os.path.getsize(xfeat_out) / 1e6
print(f"  Saved ({sz:.1f} MB): {xfeat_out}")

# Verify round-trip
loaded = torch.jit.load(xfeat_out).to(DEVICE).eval()
with torch.no_grad():
    M1_v, K1h_v, H1_v = loaded(dummy_img)
print(f"  Verified: M1={M1_v.shape}, K1h={K1h_v.shape}, H1={H1_v.shape}")

# ── 2. LighterGlue TorchScript ───────────────────────────────────────────────
print("\n=== Exporting LighterGlue to TorchScript ===")

class LighterGlueWrapper(nn.Module):
    """
    Wraps LighterGlue to accept flat tensors (no dict).
    Input:  kps0 (1,N,2), desc0 (1,N,64), kps1 (1,M,2), desc1 (1,M,64)
    Output: matches (N,) int64 (-1 = unmatched), scores (N,) float32
    """
    def __init__(self):
        super().__init__()
        self.lg = LighterGlueModel()  # loads pretrained weights automatically
        self.lg.eval()
        self.img_h = float(IMG_H_KITTI)
        self.img_w = float(IMG_W_KITTI)

    def forward(self, kps0: torch.Tensor, desc0: torch.Tensor,
                      kps1: torch.Tensor, desc1: torch.Tensor):
        # kps: (1,N,2) pixel coords;  desc: (1,N,64) L2-normalised
        img_size = torch.tensor([[self.img_h, self.img_w]],
                                 dtype=torch.float32, device=kps0.device)
        data = {
            'keypoints0':   kps0,
            'descriptors0': desc0,
            'image_size0':  img_size,
            'keypoints1':   kps1,
            'descriptors1': desc1,
            'image_size1':  img_size,
        }
        result = self.lg(data, min_conf=0.1)
        matches = result['matches0'].squeeze(0)   # (N,) int64
        scores  = result['scores0'].squeeze(0)    # (N,) float
        return matches, scores

lg_wrapper = LighterGlueWrapper().to(DEVICE).eval()

N = M = 512
dummy_kps0  = torch.rand(1, N, 2,  device=DEVICE) * float(IMG_W_KITTI)
dummy_desc0 = F.normalize(torch.rand(1, N, 64, device=DEVICE), dim=-1)
dummy_kps1  = torch.rand(1, M, 2,  device=DEVICE) * float(IMG_W_KITTI)
dummy_desc1 = F.normalize(torch.rand(1, M, 64, device=DEVICE), dim=-1)

print("  Running forward to verify interface...")
with torch.no_grad():
    try:
        matches_t, scores_t = lg_wrapper(dummy_kps0, dummy_desc0, dummy_kps1, dummy_desc1)
        print(f"  LG output: matches={matches_t.shape}, scores={scores_t.shape}")
        lg_ok = True
    except Exception as e:
        print(f"  LG forward failed: {e}")
        lg_ok = False

lg_out = str(MODELS / "lighterglue.pt")
if lg_ok:
    print(f"  Tracing -> {lg_out}")
    with torch.no_grad():
        lg_traced = torch.jit.trace(
            lg_wrapper,
            (dummy_kps0, dummy_desc0, dummy_kps1, dummy_desc1),
            strict=False
        )
    lg_traced.save(lg_out)
    sz = os.path.getsize(lg_out) / 1e6
    print(f"  Saved ({sz:.1f} MB): {lg_out}")

    lg_loaded = torch.jit.load(lg_out).to(DEVICE).eval()
    with torch.no_grad():
        m2, s2 = lg_loaded(dummy_kps0, dummy_desc0, dummy_kps1, dummy_desc1)
    print(f"  Verified: matches={m2.shape}, scores={s2.shape}")
else:
    print("  Skipping LighterGlue (relocalization will use L2 matching fallback)")

print("\n=== DONE ===")
print(f"  {xfeat_out}  -- backbone: M1 (64xH/8xW/8), K1h (heatmap), H1 (reliability)")
print(f"  {lg_out}")
