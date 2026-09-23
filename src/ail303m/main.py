import os
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/"
FEATURES = [
    "alcohol",
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "density",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "pH",
    "sulphates",
    # "quality",
    # "type",
    # Derived
    "is_white",
]


def get_incremental_path(path_str: str) -> str:
    path = Path(path_str)

    if not path.exists():
        return str(path)

    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return str(new_path)
        counter += 1


def featuring(df: pd.DataFrame) -> pd.DataFrame:
    df["is_white"] = (df["type"] == "white").astype(int)
    return df


def build_model(features: list[str]):
    inputs = keras.Input(shape=(len(FEATURES),))

    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.Dense(16, activation="relu")(x)

    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="wife")
    return model


def main():
    train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), sep=";")
    train = featuring(train)

    X = train[FEATURES].values
    y = train["quality"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_model(FEATURES)

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
    ]
    _history = model.fit(
        X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), batch_size=64, epochs=32, callbacks=callbacks
    )

    test_scores = model.evaluate(X_val, y_val, verbose=2)
    print("Val loss:", test_scores[0])
    print("Val accuracy:", test_scores[1])

    model.save(get_incremental_path("models/wife.keras"))

    # Inference
    infer = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), sep=";")
    infer = featuring(infer)
    X = infer[FEATURES]
    y = model(X)

    df = pd.DataFrame({"id": infer["id"], "quality": np.round(y[..., 0]).astype(int)})
    df.to_csv(get_incremental_path("submit/wife.csv"), index=False)


if __name__ == "__main__":
    main()
