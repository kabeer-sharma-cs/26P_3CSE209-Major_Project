from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def preprocess():
    df = pd.read_csv(DATA / "train_cleaned.csv")

    # Remove infinity, drop rows with no label
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Label"])

    # Labels -> numbers, sorted so the order is always the same
    classes = sorted(df["Label"].unique())
    y = df["Label"].map({name: i for i, name in enumerate(classes)}).to_numpy()
    X = df.drop(columns="Label")

    # Split FIRST: 80% train/validation, 20% locked away as the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fill missing values and scale using TRAIN statistics only
    median = X_train.median()
    X_train = X_train.fillna(median)
    X_test = X_test.fillna(median)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save
    MODELS.mkdir(exist_ok=True)
    np.save(DATA / "X.npy", X_train)
    np.save(DATA / "y.npy", y_train)
    np.save(DATA / "X_test.npy", X_test)
    np.save(DATA / "y_test.npy", y_test)
    joblib.dump({"scaler": scaler, "classes": classes, "median": median},
                MODELS / "preprocessor.joblib")

    print("Train/val:", X_train.shape, "| Test:", X_test.shape)
    print("Classes:", dict(enumerate(classes)))


if __name__ == "__main__":
    preprocess()