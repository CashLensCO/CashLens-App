# export_shape_detector.py
import tensorflow as tf
import numpy as np

# Cargar el mejor modelo guardado
model = tf.keras.models.load_model("shape_detector_best.keras")

# Dataset representativo para cuantización
# Carga unas pocas imágenes de val
import cv2
from pathlib import Path

def representative_dataset():
    for class_name in ["background", "bill", "coin"]:
        class_dir = Path(f"dataset/shape_detector/val/{class_name}")
        files = list(class_dir.glob("*.jpg"))[:30]
        for f in files:
            img = cv2.imread(str(f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open("shape_detector.tflite", "wb") as f:
    f.write(tflite_model)

print(f"✅ shape_detector.tflite ({len(tflite_model)/1024:.1f} KB)")

with open("shape_detector_labels.txt", "w") as f:
    f.write("background\nbill\ncoin")

print("✅ shape_detector_labels.txt")