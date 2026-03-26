# confusion_coins.py
import tensorflow as tf
import numpy as np
from pathlib import Path
import albumentations as A
import cv2
from sklearn.metrics import confusion_matrix, classification_report

CLASSES = ["100", "1000", "200", "50", "500"]
IMG_SIZE = 224

aug_val = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

images, labels = [], []
for label_idx, class_name in enumerate(CLASSES):
    class_dir = Path(f"dataset/coins/val/{class_name}")
    files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + \
            list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
    for f in files:
        img = cv2.imread(str(f))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = aug_val(image=img)["image"].astype(np.float32) / 255.0
        images.append(img)
        labels.append(label_idx)

X = np.array(images)
y = np.array(labels)

interpreter = tf.lite.Interpreter(model_path="models/coins_v1.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

preds = []
for img in X:
    interpreter.set_tensor(input_details[0]['index'], img[np.newaxis])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    preds.append(np.argmax(output))

print(classification_report(y, preds, target_names=CLASSES))
print("\nMatriz de confusión:")
print(confusion_matrix(y, preds))