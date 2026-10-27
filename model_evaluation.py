import json
import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # set before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # hide INFO/WARNING logs

import matplotlib

matplotlib.use("Agg")  # save plots to files without opening a window
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Paths (this file lives in src/, so the project root is one level up)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = ROOT / "models" / "nids_model.keras"
X_TEST_PATH = DATA / "X_test.npy"
Y_TEST_PATH = DATA / "y_test.npy"
PLOTS = ROOT / "outputs" / "plots"
REPORTS = ROOT / "outputs" / "reports"

# # Integer labels 0..4 in this exact order (must match the preprocessing script)
# CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def evaluate_model():
    try:
        # 1. Check that all required files exist
        for path in [MODEL_PATH, X_TEST_PATH, Y_TEST_PATH, PREPROCESSOR_PATH]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")
        CLASS_NAMES = list(joblib.load(PREPROCESSOR_PATH)["classes"])

        # 2. Load the trained model and the official test arrays
        model = load_model(MODEL_PATH)
        X_test = np.load(X_TEST_PATH)
        y_test = np.load(Y_TEST_PATH)
        print(f"Model loaded from: {MODEL_PATH}")
        print(f"Test samples: {len(X_test)}, features: {X_test.shape[-1]}")

        # 3. Validate the test data
        if X_test.ndim != 2:
            raise ValueError(f"X_test must be 2-D (samples, features), got shape {X_test.shape}")
        y_test = np.asarray(y_test).squeeze()
        if y_test.ndim != 1:
            raise ValueError(f"y_test must be 1-D, got shape {y_test.shape}")
        if len(X_test) != len(y_test):
            raise ValueError(f"X_test has {len(X_test)} rows but y_test has {len(y_test)}")
        if len(X_test) == 0:
            raise ValueError("The test dataset is empty")
        if not np.isfinite(X_test).all():
            raise ValueError("X_test contains NaN or infinite values")
        if not np.isfinite(y_test).all():
            raise ValueError("y_test contains NaN or infinite values")
        if not np.all(y_test == np.round(y_test)):
            raise ValueError("y_test must contain integer class labels")
        y_test = y_test.astype(int)

        num_classes = len(CLASS_NAMES)
        if y_test.min() < 0 or y_test.max() > num_classes - 1:
            raise ValueError(
                f"Labels must be in 0..{num_classes - 1}, "
                f"found range {y_test.min()}..{y_test.max()}"
            )

        # 4. Check that the data matches the model's input and output sizes
        expected_features = model.input_shape[-1]
        if X_test.shape[1] != expected_features:
            raise ValueError(
                f"Model expects {expected_features} features but X_test has {X_test.shape[1]}"
            )
        model_classes = model.output_shape[-1]
        if model_classes != num_classes:
            raise ValueError(
                f"Model outputs {model_classes} classes but {num_classes} class names "
                f"were given. Check that training and preprocessing used the same labels."
            )

        # 5. Predict probabilities, then convert to class labels
        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        # 6. Calculate metrics (labels=range(...) keeps all 5 classes in every output)
        labels = list(range(num_classes))
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0),
            "num_test_samples": len(y_test),
        }
        metrics = {k: float(v) if k != "num_test_samples" else v for k, v in metrics.items()}

        report_text = classification_report(
            y_test, y_pred, labels=labels, target_names=CLASS_NAMES,
            digits=4, zero_division=0,
        )
        report_dict = classification_report(
            y_test, y_pred, labels=labels, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # 7. Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (unseen test set)")
        print("=" * 60)
        print(f"Accuracy           : {metrics['accuracy']:.4f}")
        print(f"Precision (macro)  : {metrics['precision_macro']:.4f}")
        print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (macro)     : {metrics['recall_macro']:.4f}")
        print(f"Recall (weighted)  : {metrics['recall_weighted']:.4f}")
        print(f"F1-score (macro)   : {metrics['f1_macro']:.4f}")
        print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
        print("\nClassification Report:\n")
        print(report_text)
        print("Confusion Matrix (rows = actual, columns = predicted):\n")
        print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

        # 8. Create output folders
        PLOTS.mkdir(parents=True, exist_ok=True)
        REPORTS.mkdir(parents=True, exist_ok=True)

        # 9. Save the confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        )
        plt.title("Confusion Matrix - NIDS Test Set")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.tight_layout()
        plt.savefig(PLOTS / "confusion_matrix.png", dpi=150)
        plt.close()

        # 10. Save metrics (JSON) and classification report (CSV)
        with open(REPORTS / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        pd.DataFrame(report_dict).transpose().to_csv(REPORTS / "classification_report.csv")

        print("\nSaved files:")
        print(" -", PLOTS / "confusion_matrix.png")
        print(" -", REPORTS / "evaluation_metrics.json")
        print(" -", REPORTS / "classification_report.csv")

    except FileNotFoundError as e:
        print(f"[File Error] {e}\nRun the preprocessing and training scripts first.")
    except ValueError as e:
        print(f"[Data/Model Error] {e}")
    except (OSError, RuntimeError, TypeError, AttributeError, KeyError, IndexError) as e:
        print(f"[Unexpected Error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    evaluate_model()