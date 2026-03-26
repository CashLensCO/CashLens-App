# Prueba de 3 modelos (shape -> moneda/billete)

Este proyecto incluye un script para probar los 3 modelos TFLite en cascada:

1. Detecta la forma con `shape_detector.tflite` (`background`, `bill`, `coin`).
2. Si detecta `coin`, clasifica la denominacion con `coins_v1.tflite`.
3. Si detecta `bill`, clasifica la denominacion con `bills_v1.tflite`.

El script creado es: `probar_3_modelos.py`.

## Requisitos

- Python 3.10+
- Paquetes:

```bash
pip install tensorflow opencv-python numpy
```

## Estructura esperada de modelos

Por defecto, el script usa estos archivos dentro de `models/`:

- `models/shape_detector.tflite`
- `models/shape_detector_labels.txt`
- `models/coins_v1.tflite`
- `models/coins_v1_labels.txt`
- `models/bills_v1.tflite`
- `models/bills_v1_labels.txt`

## Uso

### Opcion 1: pasando ruta por argumento

```bash
python probar_3_modelos.py --image "mis_fotos/coins/500/tu_imagen.jpg"
```

### Opcion 2: sin argumento (te pide la ruta por consola)

```bash
python probar_3_modelos.py
```

Luego escribe la ruta cuando aparezca:

```text
Ruta de la imagen a evaluar: mis_fotos/bills/50000/tu_imagen.jpg
```

## Salida esperada

El script imprime:

- Prediccion del shape detector (con confianza y top 3).
- Si es moneda o billete, la denominacion estimada (con confianza y top 3).
- Si es `background`, informa que no detecto moneda/billete.

Ejemplo de salida:

```text
=== Resultado del Shape Detector ===
Imagen: mis_fotos/coins/500/ejemplo.jpg
Prediccion shape: coin (99.10%)
Top shape: coin: 99.10%, bill: 0.70%, background: 0.20%

=== Resultado de Clasificacion ===
Tipo detectado: MONEDA
Denominacion: 500 (97.45%)
Top denominaciones: 500: 97.45%, 1000: 1.80%, 100: 0.75%
```

## Cambiar rutas de modelos (opcional)

Si quieres usar otros archivos/rutas, puedes sobrescribirlos:

```bash
python probar_3_modelos.py --image "ruta/a/imagen.jpg" --shape-model "ruta/shape_detector.tflite" --shape-labels "ruta/shape_labels.txt" --coin-model "ruta/coins.tflite" --coin-labels "ruta/coins_labels.txt" --bill-model "ruta/bills.tflite" --bill-labels "ruta/bills_labels.txt"
```
