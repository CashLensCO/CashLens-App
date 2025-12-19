# training/train_quick.py
import argparse
import pathlib
import tensorflow as tf

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=192)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_split", type=float, default=0.2)
    return p.parse_args()

def main():
    args = parse_args()
    data_dir = pathlib.Path(args.data_dir).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_dir: {data_dir}")
    print(f"[INFO] out_dir:  {out_dir}")

    seed = 42
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        validation_split=args.val_split,
        subset="training",
        seed=seed,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        validation_split=args.val_split,
        subset="validation",
        seed=seed,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names
    print("[INFO] classes:", class_names)

    # Guarda labels.txt en el mismo orden que usa el dataset/modelo
    labels_path = out_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")
    print(f"[OK] labels -> {labels_path}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(500).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    ], name="aug")

    # Normalización MobileNet-style: [0..255] -> [-1..1]
    # Evita Lambda(preprocess_input) para que el modelo cargue sin custom_objects.
    preprocess = tf.keras.layers.Rescaling(1./127.5, offset=-1, name="preprocess")

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(args.image_size, args.image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))
    x = aug(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.keras"),
            monitor="val_accuracy",
            save_best_only=True
        ),
    ]

    print("[INFO] training...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    final_path = out_dir / "final.keras"
    model.save(final_path)
    print(f"[OK] model -> {final_path}")

    # Export TFLite (para usar después en móvil)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = out_dir / "coins_v1.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[OK] tflite -> {tflite_path}")

if __name__ == "__main__":
    main()
