# download_backgrounds.py
import requests, os, random
from pathlib import Path

# Usa imágenes de picsum (fotos reales aleatorias)
output_dir = Path("dataset/shape_detector/train/background")
output_dir.mkdir(parents=True, exist_ok=True)

categories = [
    "texture", "table", "floor", "wood", "concrete", 
    "fabric", "paper", "hand", "desk", "wallet"
]

count = 0
for i in range(600):
    # picsum da imágenes reales variadas
    url = f"https://picsum.photos/seed/{random.randint(1,9999)}/400/400"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(output_dir / f"bg_{i:04d}.jpg", "wb") as f:
                f.write(r.content)
            count += 1
            if count % 50 == 0:
                print(f"Downloaded {count}")
    except:
        pass

print(f"Total: {count} backgrounds")