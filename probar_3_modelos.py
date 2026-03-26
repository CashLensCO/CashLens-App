import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import tensorflow as tf


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"No existe el archivo de labels: {labels_path}")

    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"El archivo de labels esta vacio: {labels_path}")

    return labels


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def format_topk(scores: np.ndarray, labels: list[str], k: int = 3) -> str:
    k = min(k, len(scores), len(labels))
    top_indices = np.argsort(scores)[::-1][:k]
    return ", ".join(f"{labels[idx]}: {scores[idx] * 100:.2f}%" for idx in top_indices)


# ─────────────────────────────────────────────────────────────
# LOCALIZACIÓN DE MONEDA (mismo que en entrenamiento)
# ─────────────────────────────────────────────────────────────
def normalize_lighting(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def crop_coin(img_bgr: np.ndarray, padding: float = 0.20) -> np.ndarray:
    """
    Localiza la moneda en la imagen y la recorta.
    Usa downscale a 640px para detección rápida, luego recorta de la original.
    """
    h, w = img_bgr.shape[:2]

    # Downscale para detección rápida
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
    min_dim = min(sh, sw)

    # Hough circles
    for param2 in [30, 20, 40]:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min_dim // 3, param1=80, param2=param2,
            minRadius=int(min_dim * 0.04), maxRadius=int(min_dim * 0.55),
        )
        if circles is not None:
            best = max(circles[0], key=lambda c: c[2])
            cx = int(best[0] / scale)
            cy = int(best[1] / scale)
            r = int(best[2] / scale)

            rp = int(r * (1 + padding))
            x1, y1 = max(0, cx - rp), max(0, cy - rp)
            x2, y2 = min(w, cx + rp), min(h, cy + rp)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.shape[0] > 10 and crop.shape[1] > 10:
                return crop

    # Fallback: recorte cuadrado central
    side = min(h, w)
    ys, xs = (h - side) // 2, (w - side) // 2
    return img_bgr[ys:ys + side, xs:xs + side]


class TFLiteClassifier:
    def __init__(self, model_path: Path, labels_path: Path, normalize_input: bool = True):
        """
        normalize_input: si True, divide por 255 (modelos normales).
                         si False, pasa [0,255] directo (EfficientNet).
        """
        if not model_path.exists():
            raise FileNotFoundError(f"No existe el modelo: {model_path}")

        self.model_path = model_path
        self.labels = load_labels(labels_path)
        self.normalize_input = normalize_input

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_info = self.interpreter.get_input_details()[0]
        self.output_info = self.interpreter.get_output_details()[0]

        input_shape = self.input_info["shape"]
        if len(input_shape) != 4:
            raise ValueError(f"Shape de entrada no soportado en {model_path}: {input_shape}")

        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])

    def _prepare_input(self, image_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image_rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_AREA,
        )

        tensor = resized.astype(np.float32)

        # Solo normalizar si el modelo lo requiere.
        # EfficientNetB0 tiene normalización interna y espera [0, 255].
        if self.normalize_input:
            tensor = tensor / 255.0

        tensor = np.expand_dims(tensor, axis=0)

        input_dtype = self.input_info["dtype"]
        if input_dtype != np.float32:
            scale, zero_point = self.input_info.get("quantization", (0.0, 0))
            if scale and scale > 0:
                tensor = tensor / scale + zero_point

            if np.issubdtype(input_dtype, np.integer):
                dtype_info = np.iinfo(input_dtype)
                tensor = np.clip(tensor, dtype_info.min, dtype_info.max)

            tensor = tensor.astype(input_dtype)

        return tensor

    def _prepare_output(self, raw_output: np.ndarray) -> np.ndarray:
        scores = np.squeeze(raw_output)
        scores = np.asarray(scores)

        output_dtype = self.output_info["dtype"]
        if np.issubdtype(output_dtype, np.integer):
            scale, zero_point = self.output_info.get("quantization", (0.0, 0))
            if scale and scale > 0:
                scores = (scores.astype(np.float32) - zero_point) * scale
            else:
                scores = scores.astype(np.float32)
        else:
            scores = scores.astype(np.float32)

        if scores.ndim != 1:
            scores = scores.reshape(-1)

        if np.min(scores) < 0.0 or np.max(scores) > 1.0 or not np.isclose(np.sum(scores), 1.0, atol=1e-3):
            scores = softmax(scores)

        return scores

    def predict(self, image_rgb: np.ndarray) -> tuple[str, float, np.ndarray]:
        input_tensor = self._prepare_input(image_rgb)

        self.interpreter.set_tensor(self.input_info["index"], input_tensor)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_info["index"])

        scores = self._prepare_output(raw_output)
        pred_index = int(np.argmax(scores))

        if pred_index >= len(self.labels):
            pred_label = f"class_{pred_index}"
        else:
            pred_label = self.labels[pred_index]

        confidence = float(scores[pred_index])
        return pred_label, confidence, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba 3 modelos TFLite: shape detector + clasificador de monedas/billetes."
    )

    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Ruta de la imagen. Si no se indica, se abrira un selector de archivos.",
    )

    parser.add_argument("--shape-model", type=str, default="models/shape_detector.tflite")
    parser.add_argument("--shape-labels", type=str, default="models/shape_detector_labels.txt")

    parser.add_argument("--coin-model", type=str, default="models/coins_v2.tflite")
    parser.add_argument("--coin-labels", type=str, default="models/coins_v2_labels.txt")

    parser.add_argument("--bill-model", type=str, default="models/bills_v1.tflite")
    parser.add_argument("--bill-labels", type=str, default="models/bills_v1_labels.txt")

    return parser.parse_args()


def get_image_path(cli_path: str | None) -> Path:
    if cli_path:
        selected = cli_path
    else:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Selecciona una imagen para evaluar",
            filetypes=[
                ("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        root.destroy()

    selected = selected.strip().strip('"').strip("'")
    if not selected:
        raise ValueError("No se selecciono ninguna imagen.")

    image_path = Path(selected)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    return image_path


def main() -> None:
    args = parse_args()

    image_path = get_image_path(args.image)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Shape detector (usa /255, modelo normal) ──
    shape_classifier = TFLiteClassifier(
        model_path=Path(args.shape_model),
        labels_path=Path(args.shape_labels),
        normalize_input=True,  # shape detector sí necesita /255
    )

    shape_label, shape_confidence, shape_scores = shape_classifier.predict(image_rgb)
    shape_label_lower = shape_label.lower()

    print("\n=== Resultado del Shape Detector ===")
    print(f"Imagen: {image_path}")
    print(f"Prediccion shape: {shape_label} ({shape_confidence * 100:.2f}%)")
    print(f"Top shape: {format_topk(shape_scores, shape_classifier.labels, k=3)}")

    if shape_label_lower == "coin":
        # ── Recortar moneda antes de clasificar ──
        coin_crop_bgr = crop_coin(image_bgr)
        coin_crop_rgb = cv2.cvtColor(coin_crop_bgr, cv2.COLOR_BGR2RGB)

        currency_classifier = TFLiteClassifier(
            model_path=Path(args.coin_model),
            labels_path=Path(args.coin_labels),
            normalize_input=False,  # EfficientNetB0 espera [0, 255]
        )
        currency_type = "MONEDA"

        # Clasificar el crop, no la imagen completa
        currency_label, currency_confidence, currency_scores = currency_classifier.predict(coin_crop_rgb)

    elif shape_label_lower == "bill":
        currency_classifier = TFLiteClassifier(
            model_path=Path(args.bill_model),
            labels_path=Path(args.bill_labels),
            normalize_input=True,  # ajustar si el modelo de billetes también cambia
        )
        currency_type = "BILLETE"

        currency_label, currency_confidence, currency_scores = currency_classifier.predict(image_rgb)

    else:
        print("\nNo se detecto moneda o billete (posible background).")
        return

    print("\n=== Resultado de Clasificacion ===")
    print(f"Tipo detectado: {currency_type}")
    print(f"Denominacion: {currency_label} ({currency_confidence * 100:.2f}%)")
    print(f"Top denominaciones: {format_topk(currency_scores, currency_classifier.labels, k=3)}")


if __name__ == "__main__":
    main()