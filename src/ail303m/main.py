from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FEATURE_COLS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "type",
]


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv", sep=";")
    test = pd.read_csv(DATA_DIR / "test.csv", sep=";")
    return train, test


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    for df in (train, test):
        df["type"] = (df["type"] == "white").astype(int)

    X = train[FEATURE_COLS].values.astype(np.float32)
    y = train["quality"].values
    X_test = test[FEATURE_COLS].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    return X, y_enc, X_test, label_encoder, test["id"]


def build_model(input_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    keras.utils.set_random_seed(42)

    train, test = load_data()
    X, y, X_test, label_encoder, test_ids = preprocess(train, test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X.shape[1], len(label_encoder.classes_))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    val_pred = model.predict(X_val).argmax(axis=1)
    print(f"Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")

    test_pred = model.predict(X_test).argmax(axis=1)
    test_quality = label_encoder.inverse_transform(test_pred)

    submission = pd.DataFrame({"id": test_ids, "quality": test_quality})
    out_path = DATA_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
