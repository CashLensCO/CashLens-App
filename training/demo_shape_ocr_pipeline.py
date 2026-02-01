import argparse
import re
import cv2
import numpy as np
import tensorflow as tf
import pytesseract

TESSERACT_CMD_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------
# Labels
# -----------------------------
def load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


# -----------------------------
# Coin model preprocessing
# Debe coincidir con tu entrenamiento (Rescaling -> [-1, 1])
# -----------------------------
def preprocess_for_coin_model(bgr_img: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(bgr_img, (size, size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32)
    x = x / 127.5 - 1.0
    return np.expand_dims(x, axis=0)


# -----------------------------
# Gaussian blur helper
# -----------------------------
def gaussian_blur(bgr_img: np.ndarray, ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    k = (ksize, ksize)
    return cv2.GaussianBlur(bgr_img, k, sigmaX=sigma)


# ------------------------------------------------------------
# ✅ SHAPE DETECTION: edge-based + fallback
# ------------------------------------------------------------
def detect_main_shape(bgr_img: np.ndarray) -> tuple[str, tuple[int, int, int, int] | None, np.ndarray]:
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)

    # Canny automático
    v = np.median(gray)
    low = int(max(0, 0.66 * v))
    high = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, low, high)

    # Más agresivo para unir bordes (billetes a veces quedan partidos)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=3
    )

    H, W = edges.shape[:2]
    frame_area = float(H * W)

    # 1) Intento normal: contorno externo grande
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    if contours:
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 800:
                continue

            x, y, w, h = cv2.boundingRect(c)
            bbox_area = float(w * h)
            if bbox_area / frame_area > 0.98:
                continue

            score = bbox_area  # para billetes es mejor puntuar por bbox
            if best is None or score > best[0]:
                best = (score, c, (x, y, w, h), area)

    if best is not None:
        _, cnt, bbox, area = best
        x, y, w, h = bbox
        peri = cv2.arcLength(cnt, True)
        circularity = (4.0 * np.pi * area) / (peri * peri + 1e-6)
        ar = w / (h + 1e-6)

        if circularity > 0.55 and 0.75 <= ar <= 1.33:
            return "circle", bbox, edges

        if ar >= 1.20 or ar <= 0.83:
            return "rectangle", bbox, edges

    # 2) Fallback: bbox del componente conectado más grande
    num, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    if num <= 1:
        return "unknown", None, edges

    best_i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[best_i, cv2.CC_STAT_LEFT])
    y = int(stats[best_i, cv2.CC_STAT_TOP])
    w = int(stats[best_i, cv2.CC_STAT_WIDTH])
    h = int(stats[best_i, cv2.CC_STAT_HEIGHT])
    comp_area = float(stats[best_i, cv2.CC_STAT_AREA])

    if (w * h) / frame_area < 0.03:
        return "unknown", None, edges
    if (w * h) / frame_area > 0.98:
        return "unknown", None, edges

    ar = w / (h + 1e-6)

    if ar >= 1.20 or ar <= 0.83:
        return "rectangle", (x, y, w, h), edges

    if 0.75 <= ar <= 1.33 and comp_area / frame_area > 0.01:
        return "circle", (x, y, w, h), edges

    return "unknown", (x, y, w, h), edges


# ------------------------------------------------------------
# ✅ ROI del número del billete (esquina sup-izq) + padding
# ------------------------------------------------------------
def crop_bill_number_corner(bgr_bill_roi: np.ndarray) -> np.ndarray:
    H, W = bgr_bill_roi.shape[:2]

    # Recorte centrado en el "100/50/20/10/5/2" (arriba-izq)
    x0, y0 = 0, 0
    x1 = int(0.34 * W)
    y1 = int(0.30 * H)

    corner = bgr_bill_roi[y0:y1, x0:x1].copy()

    # ✅ padding (clave para que Tesseract no "se asfixie" en el borde)
    pad = int(0.06 * max(corner.shape[:2]))
    corner = cv2.copyMakeBorder(corner, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE)
    return corner


# ------------------------------------------------------------
# ✅ OCR preprocessing robusto para números grandes
# (reduce ruido + agranda + borde blanco)
# ------------------------------------------------------------
def preprocess_for_ocr(bgr_roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)

    # Agrandar antes de binarizar (muy importante para números grandes)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # Suavizado para quitar "grano"
    gray = cv2.medianBlur(gray, 5)

    # Contraste local
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Asegurar negro sobre blanco
    if np.mean(th == 255) < 0.5:
        th = cv2.bitwise_not(th)

    # Limpiar y unir
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                          iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                          iterations=2)

    # ✅ borde blanco extra (tesseract mejora un montón)
    th = cv2.copyMakeBorder(th, 20, 20, 20, 20, borderType=cv2.BORDER_CONSTANT, value=255)
    return th


# ------------------------------------------------------------
# ✅ OCR runner: prueba varios PSM y toma el mejor (más dígitos)
# ------------------------------------------------------------
def run_ocr_on_roi(roi_bin: np.ndarray) -> tuple[str, dict]:
    # Probar varios PSM; el mejor depende del recorte
    psms = [8]  # 8=palabra, 7=línea, 10=char, 6=bloque
    best_text = ""
    best_digits = -1
    attempts = {}

    for psm in psms:
        config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789"
        txt = pytesseract.image_to_string(roi_bin, config=config) or ""
        digits = len(re.findall(r"\d", txt))
        attempts[f"psm_{psm}"] = txt

        if digits > best_digits:
            best_digits = digits
            best_text = txt

    return best_text, attempts


# ------------------------------------------------------------
# ✅ Extraer SOLO 2/5/10/20/50/100 con normalización
# ------------------------------------------------------------
def extract_bill_value_from_big_number(ocr_text: str) -> tuple[int | None, dict]:
    cleaned = (ocr_text or "").upper()
    cleaned = cleaned.replace(" ", "").replace("\n", "").replace("\t", "")

    tokens = re.findall(r"\d{1,4}", cleaned)  # hasta 4 por si aparece 1000
    found_ints = []
    for t in tokens:
        try:
            found_ints.append(int(t))
        except ValueError:
            pass

    # Normalizaciones típicas
    normalized = []
    for n in found_ints:
        if n == 110 or n == 101:   # 100 mal leído
            normalized.append(100)
        elif n == 1000:            # 100 con 0 extra
            normalized.append(100)
        else:
            normalized.append(n)

    # Si se partieron ceros (ej: lee "1" y "00" raro), intenta reforzar
    if 1 in normalized:
        normalized.append(10)
        normalized.append(100)

    priority = [100, 50, 20, 10, 5, 2]
    chosen_big = None
    for p in priority:
        if p in normalized:
            chosen_big = p
            break

    mapping = {2: 2000, 5: 5000, 10: 10000, 20: 20000, 50: 50000, 100: 100000}
    value = mapping.get(chosen_big) if chosen_big is not None else None

    debug = {
        "raw_text": ocr_text,
        "cleaned": cleaned,
        "tokens": tokens,
        "found_ints": found_ints,
        "normalized": normalized,
        "chosen_big": chosen_big,
        "mapped_value": value
    }
    return value, debug


# -----------------------------
# Main demo pipeline
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Ruta a imagen de prueba")
    parser.add_argument("--coin_model", required=True, help="Ruta a final.keras o best.keras de coins")
    parser.add_argument("--coin_labels", required=True, help="Ruta a labels.txt de coins")
    parser.add_argument("--image_size", type=int, default=192, help="Input size del modelo de coins")
    parser.add_argument("--gauss_ksize", type=int, default=5, help="Kernel gaussiano impar 3/5/7")
    parser.add_argument("--gauss_sigma", type=float, default=1.2, help="Sigma gaussiano")
    parser.add_argument("--show_debug", action="store_true", help="Muestra ventanas OpenCV para depurar")
    args = parser.parse_args()

    if TESSERACT_CMD_WINDOWS:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_WINDOWS

    coin_model = tf.keras.models.load_model(args.coin_model)
    coin_labels = load_labels(args.coin_labels)

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError("No se pudo cargar la imagen. Revisa la ruta.")

    img_blur = gaussian_blur(img, ksize=args.gauss_ksize, sigma=args.gauss_sigma)

    shape, bbox, dbg_edges = detect_main_shape(img_blur)
    print(f"[RESULT] Forma detectada: {shape}")

    if args.show_debug:
        cv2.imshow("debug_edges", dbg_edges)

    if bbox is None:
        print("[RESULT] No se detectó objeto principal. Prueba con mejor encuadre/fondo.")
        if args.show_debug:
            cv2.imshow("input", img)
            cv2.waitKey(0)
        return

    x, y, w, h = bbox

    # padding para no cortar el objeto
    pad = int(0.04 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_blur.shape[1], x + w + pad)
    y1 = min(img_blur.shape[0], y + h + pad)

    roi = img_blur[y0:y1, x0:x1].copy()

    if shape == "circle":
        x_in = preprocess_for_coin_model(roi, args.image_size)
        probs = coin_model.predict(x_in, verbose=0)[0]
        idx = int(np.argmax(probs))
        pred = coin_labels[idx]
        conf = float(probs[idx])
        print(f"[RESULT] Moneda predicha: {pred} | Confianza: {conf:.3f}")

        print("Top-3:")
        for i in np.argsort(probs)[::-1][:3]:
            i = int(i)
            print(f"  {coin_labels[i]}: {float(probs[i]):.3f}")

        if args.show_debug:
            vis = img.copy()
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.imshow("roi_circle", roi)
            cv2.imshow("vis", vis)

    elif shape == "rectangle":
        corner = crop_bill_number_corner(roi)
        roi_ocr = preprocess_for_ocr(corner)

        text, attempts = run_ocr_on_roi(roi_ocr)
        value, dbg = extract_bill_value_from_big_number(text)

        print("[OCR] Mejor texto:")
        print(repr(text))
        print("[OCR] Tokens:", dbg["tokens"])
        print("[OCR] Enteros detectados:", dbg["found_ints"])
        print("[OCR] Normalizados:", dbg["normalized"])
        print("[OCR] Número grande elegido:", dbg["chosen_big"])

        if value is None:
            print("[RESULT] Billete detectado, pero NO se encontró 2/5/10/20/50/100.")
        else:
            print(f"[RESULT] Denominación inferida: {value}")

        if args.show_debug:
            vis = img.copy()
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.imshow("roi_rectangle_full", roi)
            cv2.imshow("roi_rectangle_corner", corner)
            cv2.imshow("roi_ocr_preprocessed", roi_ocr)
            cv2.imshow("vis", vis)

            # opcional: ver qué sacó cada psm
            for k, v in attempts.items():
                print(f"[DBG OCR {k}] -> {repr(v)}")

    else:
        print("[RESULT] Forma desconocida. No se ejecuta ni coins ni OCR.")
        if args.show_debug:
            vis = img.copy()
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.imshow("roi_unknown", roi)
            cv2.imshow("vis", vis)

    if args.show_debug:
        cv2.imshow("input_blur", img_blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
