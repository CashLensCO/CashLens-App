"""
retrain_coins_v2.py — Detector de monedas colombianas (50, 100, 200, 500, 1000)

Pipeline de 2 etapas:
  1. Localización: detectar el círculo (moneda) y recortarlo (UNA VEZ por archivo).
  2. Clasificación: clasificar la moneda recortada con modelo fine-tuned.
"""

import sys
import os
import time
import random

# ─────────────────────────────────────────────────────────────
# GPU SETUP (antes de importar TF)
# ─────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def log(msg=""):
    """Print con flush inmediato para que siempre se vea en consola."""
    print(msg, flush=True)


log("🔧 Configurando TensorFlow...")

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    log(f"✅ GPU detectada: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    log(f"   Usando {len(gpus)} GPU(s)")
else:
    log("⚠ No se detectó GPU — entrenando en CPU (más lento)")
    log("  Si tienes NVIDIA, instala soporte GPU:")
    log("    pip install tensorflow[and-cuda]")
    log("  O:  conda install -c conda-forge cudatoolkit cudnn")

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import albumentations as A
import cv2

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
IMG_SIZE       = 224
BATCH_SIZE     = 32 if gpus else 16
EPOCHS_PHASE1  = 20
EPOCHS_PHASE2  = 80
CLASSES        = ["50", "100", "200", "500", "1000"]
NUM_CLASSES    = len(CLASSES)
OVERSAMPLE_TO  = 400
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("models", exist_ok=True)

log(f"📋 Config: IMG={IMG_SIZE}, BATCH={BATCH_SIZE}, CLASSES={CLASSES}")


# ─────────────────────────────────────────────────────────────
# LOCALIZACIÓN DE MONEDA (versión rápida: 3 intentos Hough)
# ─────────────────────────────────────────────────────────────
def normalize_lighting(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def find_coin_hough(gray, h, w):
    min_dim = min(h, w)
    min_r, max_r = int(min_dim * 0.04), int(min_dim * 0.55)
    for param2 in [30, 20, 40]:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min_dim // 3, param1=80, param2=param2,
            minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            best = max(circles[0], key=lambda c: c[2])
            return int(best[0]), int(best[1]), int(best[2])
    return None


def find_coin_contours(gray, h, w):
    min_dim = min(h, w)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for t in [thresh, cv2.bitwise_not(thresh)]:
        contours, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (min_dim * 0.03) ** 2 * 3.14:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = 4 * 3.14159 * area / (perim ** 2)
            if circ < 0.5:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            score = circ * r
            if score > best_score:
                best_score = score
                best = (int(cx), int(cy), int(r))
        if best:
            return best
    return None


def crop_coin(img_bgr, padding=0.20):
    h, w = img_bgr.shape[:2]

    # ── CLAVE: reducir imagen para detección rápida ──
    # HoughCircles en 2670px = 112s. En 640px = <1s.
    MAX_DET = 640
    scale = 1.0
    if max(h, w) > MAX_DET:
        scale = MAX_DET / max(h, w)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        small = img_bgr

    norm = normalize_lighting(small)
    gray = cv2.GaussianBlur(cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    sh, sw = small.shape[:2]

    result = find_coin_hough(gray, sh, sw)
    if result is None:
        result = find_coin_contours(gray, sh, sw)

    if result:
        # Escalar coordenadas de vuelta al tamaño original
        cx, cy, r = result
        cx = int(cx / scale)
        cy = int(cy / scale)
        r  = int(r / scale)

        rp = int(r * (1 + padding))
        x1, y1 = max(0, cx - rp), max(0, cy - rp)
        x2, y2 = min(w, cx + rp), min(h, cy + rp)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.shape[0] > 10 and crop.shape[1] > 10:
            return crop, True

    side = min(h, w)
    ys, xs = (h - side) // 2, (w - side) // 2
    return img_bgr[ys:ys + side, xs:xs + side], False


# ─────────────────────────────────────────────────────────────
# AUGMENTATION (compatible albumentations 1.x y 2.x)
# ─────────────────────────────────────────────────────────────
def _try(cls, new_kw, old_kw):
    try:
        return cls(**new_kw)
    except TypeError:
        try:
            return cls(**old_kw)
        except Exception:
            log(f"  ⚠ {cls.__name__} omitido")
            return None


log("🔧 Inicializando augmentation...")
_base = [
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.85, 1.15), translate_percent=(-0.08, 0.08), shear=(-8, 8), p=0.5),
    A.Perspective(scale=(0.02, 0.06), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45, p=0.9),
    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=35, val_shift_limit=45, p=0.8),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.RandomGamma(gamma_limit=(70, 140), p=0.5),
    A.RandomToneCurve(scale=0.15, p=0.3),
    A.CLAHE(clip_limit=3.5, p=0.4),
    A.Sharpen(alpha=(0.15, 0.45), lightness=(0.7, 1.3), p=0.4),
    A.Emboss(alpha=(0.1, 0.3), strength=(0.3, 0.7), p=0.15),
    A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 7)), A.MedianBlur(blur_limit=5)], p=0.35),
]
_extras = [
    _try(A.RandomSunFlare,
         dict(flare_roi=(0.1, 0.1, 0.9, 0.9), num_flare_circles_range=(1, 3), src_radius=60, p=0.2),
         dict(flare_roi=(0.1, 0.1, 0.9, 0.9), num_flare_circles_lower=1, num_flare_circles_upper=3, src_radius=60, p=0.2)),
    _try(A.RandomShadow,
         dict(shadow_roi=(0.0, 0.0, 1.0, 1.0), num_shadows_limit=(1, 2), shadow_dimension=4, p=0.3),
         dict(shadow_roi=(0.0, 0.0, 1.0, 1.0), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4, p=0.3)),
    _try(A.GaussNoise, dict(std_range=(0.02, 0.12), p=0.3), dict(var_limit=(5.0, 30.0), p=0.3)),
    _try(A.ImageCompression, dict(quality_range=(50, 95), p=0.3), dict(quality_lower=50, quality_upper=95, p=0.3)),
]
_all = _base + [t for t in _extras if t] + [A.Resize(IMG_SIZE, IMG_SIZE)]
aug_train = A.Compose(_all)
aug_val = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])
log(f"  ✅ {len(_all)} transforms activos\n")


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
def load_and_crop_files(split):
    base = Path(f"dataset/coins/{split}")
    crops_by_class = {}
    total_files = 0
    total_detected = 0

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = base / class_name
        if not class_dir.exists():
            log(f"  ⚠ No existe: {class_dir}")
            crops_by_class[label_idx] = []
            continue

        files = []
        for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG", "webp", "bmp"]:
            files.extend(class_dir.glob(f"*.{ext}"))

        log(f"  📂 ${class_name}: {len(files)} archivos, recortando...")
        crops, detected = [], 0
        t0 = time.time()

        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is None:
                log(f"    ❌ [{i+1}/{len(files)}] {f.name} — no se pudo leer")
                continue

            crop, found = crop_coin(img)
            del img  # liberar imagen original inmediatamente

            # Redimensionar YA para no llenar la RAM con crops de 2000px
            # Guardamos a 256px (un poco más que 224 para que augmentation tenga margen)
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            crops.append(crop)
            if found:
                detected += 1

            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            eta = avg * (len(files) - i - 1)
            icon = "🟢" if found else "🟡"
            log(f"    {icon} [{i+1}/{len(files)}] {f.name} "
                f"[{elapsed:.1f}s / ETA {eta:.0f}s]")

        log(f"  ✅ ${class_name}: {len(crops)} crops, "
            f"{detected}/{len(files)} monedas detectadas "
            f"({time.time()-t0:.1f}s)\n")

        crops_by_class[label_idx] = crops
        total_files += len(files)
        total_detected += detected

    log(f"  📊 Total: {total_files} archivos, {total_detected} monedas detectadas")
    return crops_by_class


def build_arrays(crops_by_class, split, oversample_to=None):
    is_train = (split == "train")
    transform = aug_train if is_train else aug_val
    images, labels = [], []

    for label_idx in range(NUM_CLASSES):
        crops = crops_by_class.get(label_idx, [])
        if not crops:
            log(f"  ⚠ Clase {CLASSES[label_idx]}: 0 imágenes!")
            continue

        if is_train and oversample_to and len(crops) < oversample_to:
            extended = []
            while len(extended) < oversample_to:
                extended.extend(crops)
            crops = extended[:oversample_to]

        log(f"  🔄 ${CLASSES[label_idx]}: augmentando {len(crops)} crops...")
        t0 = time.time()
        for crop_bgr in crops:
            img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            aug = transform(image=img_rgb)["image"]
            # EfficientNetB0 tiene su propia normalización interna
            # Espera [0, 255], NO [0, 1]. Dividir por 255 = doble normalización = no aprende.
            images.append(aug.astype(np.float32))
            labels.append(label_idx)
        log(f"    ✅ listo en {time.time()-t0:.1f}s")

    log(f"  🔀 Mezclando {len(images)} imágenes...")
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    return np.array(images), np.array(labels)


log("=" * 60)
log("PASO 1: RECORTANDO MONEDAS DE IMÁGENES ORIGINALES")
log("=" * 60)

log("\n── Train ──")
t0 = time.time()
train_crops = load_and_crop_files("train")
log(f"  ⏱ {time.time()-t0:.1f}s total\n")

log("── Val ──")
t0 = time.time()
val_crops = load_and_crop_files("val")
log(f"  ⏱ {time.time()-t0:.1f}s total\n")

log("=" * 60)
log("PASO 2: AUGMENTATION + OVERSAMPLING")
log("=" * 60)

t0 = time.time()
X_train, y_train = build_arrays(train_crops, "train", oversample_to=OVERSAMPLE_TO)
X_val, y_val     = build_arrays(val_crops, "val")
log(f"\n  ⏱ Augmentation: {time.time()-t0:.1f}s")

log(f"\n📊 Resumen:")
log(f"  Train: {X_train.shape}")
log(f"  Val:   {X_val.shape}")
for i, name in enumerate(CLASSES):
    log(f"  ${name}: train={np.sum(y_train == i)}, val={np.sum(y_val == i)}")

del train_crops, val_crops
log("")


# ─────────────────────────────────────────────────────────────
# MIXUP + GENERADOR
# ─────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=0.2):
    n = len(x)
    lam = np.maximum(l := np.random.beta(alpha, alpha, n).astype(np.float32), 1 - l)
    idx = np.random.permutation(n)
    xm = x * lam.reshape(-1, 1, 1, 1) + x[idx] * (1 - lam.reshape(-1, 1, 1, 1))
    yoh = tf.one_hot(y, NUM_CLASSES).numpy()
    ym = yoh * lam.reshape(-1, 1) + yoh[idx] * (1 - lam.reshape(-1, 1))
    return xm, ym


def data_gen(X, y, bs, mixup=True):
    n = len(X)
    idx = np.arange(n)
    while True:
        np.random.shuffle(idx)
        for s in range(0, n, bs):
            e = min(s + bs, n)
            bx, by = X[idx[s:e]], y[idx[s:e]]
            if mixup and len(bx) > 1:
                yield mixup_batch(bx, by)
            else:
                yield bx, tf.one_hot(by, NUM_CLASSES).numpy()


# ─────────────────────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────────────────────
log("=" * 60)
log("PASO 3: CONSTRUYENDO MODELO (EfficientNetB0)")
log("=" * 60)

base_model = keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)

log(f"  Parámetros: {model.count_params():,}")

spe = max(1, len(X_train) // BATCH_SIZE)
vs  = max(1, len(X_val) // BATCH_SIZE)


# ── FASE 1 ──
log("\n" + "=" * 60)
log("PASO 4a: FASE 1 — Entrenando cabeza (base congelada)")
log(f"  {EPOCHS_PHASE1} epochs, {spe} steps/epoch, batch={BATCH_SIZE}")
log("=" * 60 + "\n")

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    data_gen(X_train, y_train, BATCH_SIZE, mixup=True),
    steps_per_epoch=spe,
    validation_data=data_gen(X_val, y_val, BATCH_SIZE, mixup=False),
    validation_steps=vs,
    epochs=EPOCHS_PHASE1,
    callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy")],
    verbose=1,
)


# ── FASE 2 ──
log("\n" + "=" * 60)
log("PASO 4b: FASE 2 — Fine-tuning (descongelando capas)")
log("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

log(f"  Capas entrenables: {sum(1 for l in model.layers if l.trainable)}/{len(model.layers)}")
log(f"  {EPOCHS_PHASE2} epochs, LR=1e-5\n")

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    data_gen(X_train, y_train, BATCH_SIZE, mixup=True),
    steps_per_epoch=spe,
    validation_data=data_gen(X_val, y_val, BATCH_SIZE, mixup=False),
    validation_steps=vs,
    epochs=EPOCHS_PHASE2,
    callbacks=[
        keras.callbacks.ModelCheckpoint("models/coins_v2_best.keras",
                                        save_best_only=True, monitor="val_accuracy", verbose=1),
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_accuracy", verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ],
    verbose=1,
)


# ─────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("PASO 5: EVALUACIÓN")
log("=" * 60)

y_pred = np.argmax(model.predict(X_val, batch_size=BATCH_SIZE), axis=1)

try:
    from sklearn.metrics import classification_report, confusion_matrix
    log(classification_report(y_val, y_pred, target_names=CLASSES))
    log(str(confusion_matrix(y_val, y_pred)))
except ImportError:
    log(f"Accuracy: {np.mean(y_val == y_pred)*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# EXPORTAR
# ─────────────────────────────────────────────────────────────
log("\n🔧 Exportando TFLite...")

def representative_dataset():
    for i in range(min(300, len(X_val))):
        yield [X_val[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite = converter.convert()
with open("models/coins_v2.tflite", "wb") as f:
    f.write(tflite)
log(f"✅ models/coins_v2.tflite ({len(tflite)/1024:.1f} KB)")

with open("models/coins_v2_labels.txt", "w") as f:
    f.write("\n".join(CLASSES))
log("✅ models/coins_v2_labels.txt")

# inference script
with open("models/inference_coins.py", "w") as f:
    f.write(r'''"""inference_coins.py — python inference_coins.py imagen.jpg"""
import sys, cv2, numpy as np
from pathlib import Path
IMG_SIZE, CLASSES = 224, ["50","100","200","500","1000"]
def norm(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(3.0,(8,8)).apply(l); return cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR)
def crop(img,pad=.2):
    h,w=img.shape[:2]; sc=1.0
    if max(h,w)>640: sc=640/max(h,w); sm=cv2.resize(img,(int(w*sc),int(h*sc)))
    else: sm=img
    g=cv2.GaussianBlur(cv2.cvtColor(norm(sm),cv2.COLOR_BGR2GRAY),(5,5),0); sh,sw=sm.shape[:2]; md=min(sh,sw)
    for p2 in [30,20,40]:
        c=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1.2,md//3,param1=80,param2=p2,minRadius=int(md*.04),maxRadius=int(md*.55))
        if c is not None:
            b=max(c[0],key=lambda x:x[2]); cx,cy,r=int(b[0]/sc),int(b[1]/sc),int(b[2]/sc); rp=int(r*(1+pad))
            cr=img[max(0,cy-rp):min(h,cy+rp),max(0,cx-rp):min(w,cx+rp)]
            if cr.shape[0]>10 and cr.shape[1]>10: return cr
    s=min(h,w); return img[(h-s)//2:(h+s)//2,(w-s)//2:(w+s)//2]
def tta(interp,img,n=4):
    i,o=interp.get_input_details(),interp.get_output_details(); ps=[]
    for k in range(n):
        a=img.copy()
        if k>0: M=cv2.getRotationMatrix2D((a.shape[1]//2,a.shape[0]//2),k*90,1); a=cv2.warpAffine(a,M,(a.shape[1],a.shape[0]))
        c=crop(a); x=cv2.resize(cv2.cvtColor(c,cv2.COLOR_BGR2RGB),(IMG_SIZE,IMG_SIZE)).astype(np.float32)
        interp.set_tensor(i[0]["index"],np.expand_dims(x,0)); interp.invoke(); ps.append(interp.get_tensor(o[0]["index"])[0])
    return np.mean(ps,axis=0)
if __name__=="__main__":
    import tensorflow as tf; it=tf.lite.Interpreter("models/coins_v2.tflite"); it.allocate_tensors()
    for p in sys.argv[1:]:
        p=Path(p); fs=list(p.glob("*.*")) if p.is_dir() else [p]
        for f in fs:
            img=cv2.imread(str(f))
            if img is None: continue
            pr=tta(it,img); ix=np.argmax(pr); print(f"{f.name}: ${CLASSES[ix]} ({pr[ix]*100:.1f}%)")
''')
log("✅ models/inference_coins.py")
log("\n🎉 ¡LISTO!")