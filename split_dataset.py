# split_dataset.py
import shutil, random
from pathlib import Path

def split_class(src_dir: str, train_dir: str, val_dir: str, val_ratio=0.2):
    src = Path(src_dir)
    if not src.exists():
        print(f"SKIP (no existe): {src_dir}")
        return
    files = list(src.glob("*.jpg")) + list(src.glob("*.png")) + \
            list(src.glob("*.jpeg")) + list(src.glob("*.JPG"))
    if not files:
        print(f"SKIP (vacío): {src_dir}")
        return
    random.shuffle(files)
    
    n_val = int(len(files) * val_ratio)
    val_files   = files[:n_val]
    train_files = files[n_val:]
    
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    
    for f in train_files:
        shutil.copy(f, Path(train_dir) / f.name)
    for f in val_files:
        shutil.copy(f, Path(val_dir) / f.name)
    
    print(f"{src_dir}: {len(train_files)} train, {len(val_files)} val")

# Monedas
for denom in ["50", "100", "200", "500", "1000"]:
    split_class(
        f"mis_fotos/coins/{denom}",
        f"dataset/coins/train/{denom}",
        f"dataset/coins/val/{denom}"
    )