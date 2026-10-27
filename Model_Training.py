import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files without opening a window
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PLOTS = ROOT / "outputs" / "plots"

EPOCHS = 20
BATCH_SIZE = 64
SEED = 42


def train_model():
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # 1. Load data (X.npy / y.npy come from Data_Preprocessing.py)
    try:
        X = np.load(DATA / "X.npy")
        y = np.load(DATA / "y.npy")
    except FileNotFoundError as e:
        print("Missing file:", e, "\nRun Data_Preprocessing.py first.")
        return

    # 2. Check the data
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    num_features = X.shape[1]
    num_classes = int(y.max()) + 1
    print(f"Samples: {len(X)}, features: {num_features}, classes: {num_classes}")

    # 3. Split into training and validation (80/20, keeps class proportions)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 4. Class weights (the data is mostly normal traffic, so rare attacks get more weight)
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(weights))

    # 5. Build the model
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 6. Train (stop early when validation loss stops improving)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    # 7. Save the model
    MODELS.mkdir(exist_ok=True)
    model.save(MODELS / "nids_model.keras")

    # 8. Plot accuracy and loss
    PLOTS.mkdir(parents=True, exist_ok=True)
    for metric in ["accuracy", "loss"]:
        plt.figure()
        plt.plot(history.history[metric], label="Training")
        plt.plot(history.history["val_" + metric], label="Validation")
        plt.title(f"Training and validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(PLOTS / f"{metric}.png")
        plt.close()

    # 9. Final results
    print("\nFinal training accuracy:  ", round(history.history["accuracy"][-1], 4))
    print("Final training loss:      ", round(history.history["loss"][-1], 4))
    print("Final validation accuracy:", round(history.history["val_accuracy"][-1], 4))
    print("Final validation loss:    ", round(history.history["val_loss"][-1], 4))
    print("Best validation accuracy: ", round(max(history.history["val_accuracy"]), 4))
    print("Model saved to", MODELS / "nids_model.keras")


if __name__ == "__main__":
    train_model()