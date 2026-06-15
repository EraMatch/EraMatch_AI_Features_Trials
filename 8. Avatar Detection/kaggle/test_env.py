"""
Diagnostic: verifies GPU, datasets, imports, and model instantiation.
Run this on Kaggle (T4 GPU) before any training trial.
"""
import os, sys, json, time, random

print("=" * 60)
print("KAGGLE ENV TEST — AI Avatar Detection")
print("=" * 60)

# ─── 1. System info ───────────────────────────────────────────
print("\n[1] SYSTEM INFO")
print(f"Python: {sys.version}")
import platform
print(f"Platform: {platform.platform()}")
import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader"], capture_output=True, text=True)
if r.returncode == 0:
    print(f"nvidia-smi: {r.stdout.strip()}")

# ─── 2. GPU ───────────────────────────────────────────────────
print("\n[2] GPU CHECK")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    VRAM: {props.total_memory/1e9:.2f} GB")
        print(f"    CUDA capability: {props.major}.{props.minor}")

# ─── 3. Library versions ──────────────────────────────────────
print("\n[3] LIBRARY VERSIONS")
import timm;           print(f"timm:            {timm.__version__}")
import albumentations as A; print(f"albumentations:  {A.__version__}")
import sklearn;        print(f"scikit-learn:    {sklearn.__version__}")
import numpy as np;   print(f"numpy:           {np.__version__}")
import pandas as pd;  print(f"pandas:          {pd.__version__}")
import scipy;         print(f"scipy:           {scipy.__version__}")
from PIL import Image
import PIL;           print(f"Pillow:          {PIL.__version__}")

# ─── 4. Dataset paths + structure ─────────────────────────────
print("\n[4] DATASET PATHS")
INPUT = "/kaggle/input"

def tree(path, depth=0, max_depth=3, max_items=6):
    if depth > max_depth or not os.path.isdir(path):
        return
    for item in sorted(os.listdir(path))[:max_items]:
        full = os.path.join(path, item)
        indent = "  " * depth
        if os.path.isdir(full):
            n = len(os.listdir(full))
            print(f"{indent}{item}/ ({n} items)")
            tree(full, depth + 1, max_depth, max_items)
        else:
            sz = os.path.getsize(full) / 1024
            print(f"{indent}{item} ({sz:.0f} KB)")

for ds in sorted(os.listdir(INPUT)):
    ds_path = os.path.join(INPUT, ds)
    if not os.path.isdir(ds_path):
        continue
    print(f"\n  /kaggle/input/{ds}/")
    tree(ds_path, depth=1)

# ─── 5. Image counts per class ────────────────────────────────
print("\n[5] IMAGE COUNTS")
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp')

def count_walk(path):
    counts = {}
    for root, dirs, files in os.walk(path):
        imgs = [f for f in files if f.lower().endswith(IMG_EXTS)]
        if imgs:
            rel = os.path.relpath(root, path)
            counts[rel] = len(imgs)
    return counts

for ds in sorted(os.listdir(INPUT)):
    ds_path = os.path.join(INPUT, ds)
    if not os.path.isdir(ds_path):
        continue
    print(f"\n  {ds}:")
    counts = count_walk(ds_path)
    total = sum(counts.values())
    print(f"    Total images: {total}")
    for folder, n in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        tag = ""
        lower = folder.lower()
        if "real" in lower or "ffhq" in lower or "flickr" in lower:
            tag = " ← REAL"
        elif any(k in lower for k in ("fake", "ai", "gen", "flux", "sdxl", "synth")):
            tag = " ← FAKE"
        print(f"      {folder}: {n}{tag}")

# ─── 6. PIL image load test ───────────────────────────────────
print("\n[6] IMAGE LOADING TEST")
def find_images(base, n=6):
    found = []
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                found.append(os.path.join(root, f))
        if len(found) >= 200:
            break
    return random.sample(found, min(n, len(found)))

for ds in sorted(os.listdir(INPUT)):
    ds_path = os.path.join(INPUT, ds)
    if not os.path.isdir(ds_path):
        continue
    print(f"\n  {ds}:")
    for p in find_images(ds_path, 4):
        try:
            img = Image.open(p).convert("RGB")
            rel = os.path.relpath(p, ds_path)
            print(f"    OK   {rel} — {img.size} {img.mode}")
        except Exception as e:
            print(f"    FAIL {p}: {e}")

# ─── 7. Albumentations API compatibility ──────────────────────
print("\n[7] ALBUMENTATIONS API TEST")
try:
    from albumentations.pytorch import ToTensorV2
    transform = A.Compose([
        A.Resize(256, 256),
        A.RandomHorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(40, 90), p=0.5),  # NEW API
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    dummy = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    out = transform(image=dummy)["image"]
    print(f"  ImageCompression (new API): OK — output {tuple(out.shape)}")
except Exception as e:
    print(f"  New API failed: {e}")
    try:
        transform2 = A.Compose([
            A.Resize(256, 256),
            A.JpegCompression(quality_lower=40, quality_upper=90, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        dummy2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transform2(image=dummy2)
        print("  JpegCompression (old API): OK — NOTE: use old API in training scripts")
    except Exception as e2:
        print(f"  Old API also failed: {e2}")

# ─── 8. RandomResizedCrop API ─────────────────────────────────
print("\n  RandomResizedCrop API:")
try:
    crop = A.RandomResizedCrop(size=256, scale=(0.7, 1.0))
    dummy = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    crop(image=dummy)
    print("  size= API (new): OK")
except Exception as e:
    print(f"  size= API failed: {e}")
    try:
        crop2 = A.RandomResizedCrop(height=256, width=256, scale=(0.7, 1.0))
        crop2(image=dummy)
        print("  height=/width= API (old): OK — NOTE: use old API in training scripts")
    except Exception as e2:
        print(f"  Old API also failed: {e2}")

# ─── 9. Model instantiation ───────────────────────────────────
print("\n[8] MODEL INSTANTIATION + FORWARD PASS")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EfficientNet-B1 (Trial 1)
    m_eff = timm.create_model("efficientnet_b1", pretrained=False, num_classes=2).to(device)
    n_eff = sum(p.numel() for p in m_eff.parameters())

    # ConvNeXt-tiny (Trial 2)
    m_cnx = timm.create_model("convnext_tiny", pretrained=False, num_classes=2).to(device)
    n_cnx = sum(p.numel() for p in m_cnx.parameters())

    x = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        o_eff = m_eff(x)
        o_cnx = m_cnx(x)

    print(f"  EfficientNet-B1: {n_eff/1e6:.1f}M params → out {tuple(o_eff.shape)}")
    print(f"  ConvNeXt-tiny:   {n_cnx/1e6:.1f}M params → out {tuple(o_cnx.shape)}")
    print("  Models: OK")
except Exception as e:
    print(f"  Model ERROR: {e}")
    import traceback; traceback.print_exc()

# ─── 10. EfficientNet-B1 4-channel surgery (DCT branch) ───────
print("\n[9] 4-CHANNEL WEIGHT SURGERY (Trial 1 DCT)")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b1", pretrained=False, num_classes=2)

    # Replicate first conv for 4-channel input (same pattern as train_nb1_dct.py)
    old_conv = model.conv_stem
    new_conv = torch.nn.Conv2d(
        4, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:, :, :] = old_conv.weight[:, :1, :, :]  # copy R channel
    model.conv_stem = new_conv
    model = model.to(device)

    x4 = torch.randn(2, 4, 224, 224).to(device)
    with torch.no_grad():
        out4 = model(x4)
    print(f"  4-channel input → output {tuple(out4.shape)}: OK")
except Exception as e:
    print(f"  4-channel surgery ERROR: {e}")
    import traceback; traceback.print_exc()

# ─── 11. scipy DCT ────────────────────────────────────────────
print("\n[10] SCIPY DCT")
try:
    from scipy.fft import dctn, idctn
    gray = np.random.rand(256, 256).astype(np.float32)
    dct_map = dctn(gray, type=2, norm="ortho")
    recon  = idctn(dct_map, type=2, norm="ortho")
    err = float(np.abs(gray - recon).max())
    mag = np.log1p(np.abs(dct_map))
    print(f"  Roundtrip max error: {err:.2e} (expect <1e-5)")
    print(f"  DCT magnitude range: [{mag.min():.3f}, {mag.max():.3f}]")
    print("  scipy DCT: OK")
except Exception as e:
    print(f"  scipy DCT ERROR: {e}")

# ─── 12. DataLoader throughput ────────────────────────────────
print("\n[11] DATALOADER THROUGHPUT")
try:
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T

    class TinyDS(Dataset):
        def __init__(self, paths):
            self.paths = paths
            self.tfm = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            return self.tfm(Image.open(self.paths[i]).convert("RGB")), 0

    for ds in sorted(os.listdir(INPUT)):
        ds_path = os.path.join(INPUT, ds)
        if not os.path.isdir(ds_path):
            continue
        imgs = find_images(ds_path, 64)
        if not imgs:
            continue
        loader = DataLoader(TinyDS(imgs), batch_size=16, num_workers=2, pin_memory=True)
        t0 = time.time()
        for b, _ in loader:
            pass
        t1 = time.time()
        print(f"  {ds}: {len(imgs)} imgs / {t1-t0:.2f}s = {len(imgs)/(t1-t0):.0f} img/s")
except Exception as e:
    print(f"  DataLoader ERROR: {e}")
    import traceback; traceback.print_exc()

# ─── 13. GPU memory headroom ──────────────────────────────────
if torch.cuda.is_available():
    print("\n[12] GPU MEMORY HEADROOM (batch=32, 256×256)")
    try:
        device = torch.device("cuda")
        model = timm.create_model("efficientnet_b1", pretrained=False, num_classes=2).to(device)
        x = torch.randn(32, 3, 256, 256, device=device)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x)
        used  = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Peak: {used:.2f} GB / {total:.1f} GB total")
        print(f"  Headroom: {total - used:.2f} GB")
        if total - used > 4:
            print("  Verdict: sufficient for training (need ≥4 GB headroom)")
        else:
            print("  WARNING: low headroom — reduce batch size in training scripts")
        del model, x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  GPU memory ERROR: {e}")

# ─── Final summary ────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
summary = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
              if torch.cuda.is_available() else 0,
    "timm": timm.__version__,
    "albumentations": A.__version__,
    "scipy": scipy.__version__,
    "datasets": sorted([
        d for d in os.listdir(INPUT)
        if os.path.isdir(os.path.join(INPUT, d))
    ]),
}
print(json.dumps(summary, indent=2))
print("\nDONE — check output above for any FAIL or ERROR lines.")
