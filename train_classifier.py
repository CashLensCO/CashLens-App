# train_classifier.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import albumentations as A
import cv2
import sys

# ── Config ────────────────────────────────────────────────
MODE = sys.argv[1] if len(sys.argv) > 1 else "coins"  # "coins" o "bills"

if MODE == "coins":
    IMG_SIZE   = 192
    CLASSES    = ["100", "1000", "200", "50", "500"]   # alfabético
    DATA_DIR   = "dataset/coins"
    MODEL_NAME = "coins_v1"
else:
    IMG_SIZE   = 224
    CLASSES    = ["10000", "100000", "20000", "2000", "50000", "5000"]  # alfabético
    DATA_DIR   = "dataset/bills"
    MODEL_NAME = "bills_v1"

BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-4
# ─────────────────────────────────────────────────────────

aug_train = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=45, p=0.7),  # tus fotos ya tienen rotaciones pero más no hace daño
    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    A.GaussNoise(var_limit=(10, 60), p=0.3),
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.3),
    A.Perspective(scale=(0.03, 0.10), p=0.5),
    A.CoarseDropout(max_holes=6, max_height=30, max_width=30, p=0.25),
    A.CLAHE(clip_limit=3.0, p=0.3),  # mejora contraste local — ayuda mucho con billetes
    A.Resize(IMG_SIZE, IMG_SIZE),
])

aug_val = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])


def load_dataset(split="train"):
    base = Path(f"{DATA_DIR}/{split}")
    images, labels = [], []
    
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = base / class_name
        if not class_dir.exists():
            print(f"WARNING: {class_dir} no existe")
            continue
        
        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
                list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.JPG"))
        print(f"  {class_name}: {len(files)} imgs")
        
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transform = aug_train if split == "train" else aug_val
            img = transform(image=img)["image"].astype(np.float32) / 255.0
            images.append(img)
            labels.append(label_idx)
    
    return np.array(images), np.array(labels)


print(f"\n=== Entrenando: {MODE.upper()} ===")
print("Cargando datos...")
X_train, y_train = load_dataset("train")
X_val,   y_val   = load_dataset("val")
print(f"Train: {X_train.shape} | Val: {X_val.shape}")

# ── Modelo ───────────────────────────────────────────────
# Para billetes usamos EfficientNetLite que captura más detalle fino
# Para monedas MobileNetV2 es suficiente (forma circular más simple)

if MODE == "bills":
    base_model = keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
else:
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

base_model.trainable = False

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        f"{MODEL_NAME}_best.keras",
        save_best_only=True, monitor="val_accuracy", verbose=1
    ),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
]

# Fase 1
print("\n=== FASE 1: cabeza ===")
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=20, batch_size=BATCH_SIZE, callbacks=callbacks)

# Fase 2: fine-tune
print("\n=== FASE 2: fine-tune ===")
base_model.trainable = True
freeze_until = 80 if MODE == "coins" else 100
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(LR / 10),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

# ── Exportar TFLite INT8 ─────────────────────────────────
def representative_dataset():
    for i in range(min(300, len(X_val))):
        yield [X_val[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
out_path = f"{MODEL_NAME}.tflite"
with open(out_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ {out_path} guardado ({len(tflite_model)/1024:.1f} KB)")

with open(f"{MODEL_NAME}_labels.txt", "w") as f:
    f.write("\n".join(CLASSES))
print(f"✅ {MODEL_NAME}_labels.txt guardado")