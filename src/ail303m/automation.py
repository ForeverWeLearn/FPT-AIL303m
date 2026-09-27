import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ail303m.utils import get_incremental_path

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
    "quality",
    "type",
    "is_white",
    "total_acidity",
    "volatile_to_fixed_ratio",
    "citric_to_total_acid_ratio",
    "bound_sulfur_dioxide",
    "free_so2_proportion",
    "sugar_to_alcohol_ratio",
    "sugar_to_acid_ratio",
    "alcohol_total_acid_product",
    "free_so2_per_pH",
]
FEATURES_SET: dict[str, list[str]] = {
    "raw": [
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
        "is_white",
    ],
    "extras": [
        "alcohol",
        "citric acid",
        "density",
        "is_white",
        "total_acidity",
        "volatile_to_fixed_ratio",
        "citric_to_total_acid_ratio",
        "bound_sulfur_dioxide",
        "free_so2_proportion",
        "sugar_to_alcohol_ratio",
        "sugar_to_acid_ratio",
        "alcohol_total_acid_product",
        "free_so2_per_pH",
    ],
}
report = pd.DataFrame(columns=["model", "loss", "accuracy", "features_set"])
scaler = StandardScaler()


def featuring(df: pd.DataFrame) -> pd.DataFrame:
    if "quality" in df.columns:
        df["quality"] -= 3

    df["is_white"] = (df["type"] == "white").astype(np.float32)

    df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
    df["volatile_to_fixed_ratio"] = df["volatile acidity"] / (
        df["fixed acidity"] + 1e-5
    )
    df["citric_to_total_acid_ratio"] = df["citric acid"] / (df["total_acidity"] + 1e-5)

    df["bound_sulfur_dioxide"] = df["total sulfur dioxide"] - df["free sulfur dioxide"]
    df["free_so2_proportion"] = df["free sulfur dioxide"] / (
        df["total sulfur dioxide"] + 1e-5
    )

    df["sugar_to_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
    df["sugar_to_acid_ratio"] = df["residual sugar"] / (df["total_acidity"] + 1e-5)

    df["alcohol_total_acid_product"] = df["alcohol"] * df["total_acidity"]
    df["free_so2_per_pH"] = df["free sulfur dioxide"] / (df["pH"] + 1e-5)

    return df


def get_train_val(df: pd.DataFrame, features: list[str]) -> tuple:
    X = df[features]
    y = df["quality"].values.astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


def build_model_lg(num_features: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(num_features,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dense(7, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_model_md(num_features: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(num_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(7, activation="softmax"),
        ]
    )
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def build_model_sm(num_features: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(num_features,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(12, activation="relu"),
            layers.Dense(7, activation="softmax"),
        ]
    )
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def train(
    model: keras.Model, X_train, X_val, y_train, y_val, epochs: int, batch_size: int
):
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
    )


def val(model: keras.Model, X_val, y_val) -> tuple[float, float]:
    scores = model.evaluate(X_val, y_val, verbose="2")
    print("Val loss:", scores[0])
    print("Val accuracy:", scores[1])
    return scores[0], scores[1]


def submit(model: keras.Model, X, ids, file_identity: str):
    y = np.argmax(model(X), axis=-1)
    y += 3
    df = pd.DataFrame({"id": ids, "quality": y})
    df.to_csv(get_incremental_path(f"submit/wife-{file_identity}.csv"), index=False)


def run_with_features_set(features_set: str):
    features = FEATURES_SET[features_set]

    train_df = pd.read_csv("data/train.csv", sep=";")
    infer_df = pd.read_csv("data/test.csv", sep=";")

    train_df = featuring(train_df)
    infer_df = featuring(infer_df)

    X_train, X_val, y_train, y_val = get_train_val(train_df, features)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_infer = scaler.fit_transform(infer_df[features])

    num_features = len(features)
    how_slow = 1.0
    models: dict[str, tuple[keras.Model, int, int]] = {
        "l": (build_model_lg(num_features), int(300 * how_slow), 128),
        "m": (build_model_md(num_features), int(200 * how_slow), 64),
        "s": (build_model_sm(num_features), int(100 * how_slow), 64),
    }

    for name, model in models.items():
        train(model[0], X_train, X_val, y_train, y_val, model[1], model[2])

        loss, accuracy = val(model[0], X_val, y_val)
        report.loc[len(report)] = [name, loss, accuracy, features_set]

        submit(model[0], X_infer, infer_df["id"], f"{name}-{features_set}")


def main():
    for features_set in FEATURES_SET:
        run_with_features_set(features_set)

    print(report)


if __name__ == "__main__":
    main()
