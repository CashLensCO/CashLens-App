# training/predict.py
import argparse
import numpy as np
import tensorflow as tf

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Ruta a final.keras o best.keras")
    p.add_argument("--labels", type=str, required=True, help="Ruta a labels.txt")
    p.add_argument("--image", type=str, required=True, help="Ruta a imagen jpg/png")
    p.add_argument("--image_size", type=int, default=192)
    return p.parse_args()

def main():
    args = parse_args()

    # Carga directa (ya no requiere custom_objects)
    model = tf.keras.models.load_model(args.model)

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    img = tf.keras.utils.load_img(args.image, target_size=(args.image_size, args.image_size))
    x = tf.keras.utils.img_to_array(img)          # float32 [0..255]
    x = np.expand_dims(x, axis=0)                 # (1, H, W, 3)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    pred_label = labels[idx]
    conf = float(probs[idx])

    top3 = np.argsort(probs)[::-1][:3]
    print(f"Predicción: {pred_label}  (conf={conf:.3f})")
    print("Top-3:")
    for i in top3:
        i = int(i)
        print(f"  {labels[i]}: {float(probs[i]):.3f}")

if __name__ == "__main__":
    main()
