# train_shape_detector.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import albumentations as A
import cv2

# ── Config ──────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 40
LR          = 1e-4
MODEL_NAME  = "shape_detector"
CLASSES     = ["background", "bill", "coin"]  # orden alfabético = orden Keras
# ────────────────────────────────────────────────────────

# Augmentations con Albumentations
aug = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # simula desenfoque de cámara
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.RandomShadow(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2),
    A.Perspective(scale=(0.02, 0.08), p=0.4),  # simula ángulos de cámara
    A.Resize(IMG_SIZE, IMG_SIZE),
])

aug_val = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])


def load_dataset(split="train"):
    base = Path(f"dataset/shape_detector/{split}")
    images, labels = [], []
    
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = base / class_name
        if not class_dir.exists():
            print(f"WARNING: {class_dir} no existe")
            continue
        
        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
                list(class_dir.glob("*.jpeg"))
        print(f"  {class_name}: {len(files)} imágenes")
        
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            transform = aug if split == "train" else aug_val
            augmented = transform(image=img)
            img = augmented["image"].astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(label_idx)
    
    return np.array(images), np.array(labels)


print("Cargando train...")
X_train, y_train = load_dataset("train")
print("Cargando val...")
X_val, y_val = load_dataset("val")

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# ── Modelo: MobileNetV2 con transfer learning ────────────
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # primero congelado

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        f"{MODEL_NAME}_best.keras",
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1
    ),
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
]

# Fase 1: solo la cabeza
print("\n=== FASE 1: Entrenando cabeza ===")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

# Fase 2: fine-tuning capas superiores del backbone
print("\n=== FASE 2: Fine-tuning ===")
base_model.trainable = True

# Congelar las primeras 100 capas, entrenar el resto
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(LR / 10),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

print("✅ Shape detector entrenado")

# ── Exportar a TFLite con cuantización INT8 ──────────────
def representative_dataset():
    for i in range(min(200, len(X_val))):
        yield [X_val[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open("shape_detector.tflite", "wb") as f:
    f.write(tflite_model)

print(f"✅ shape_detector.tflite guardado ({len(tflite_model)/1024:.1f} KB)")

# Guardar labels
with open("shape_detector_labels.txt", "w") as f:
    f.write("\n".join(CLASSES))