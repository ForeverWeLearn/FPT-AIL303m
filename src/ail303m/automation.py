import keras
import numpy as np
import pandas as pd
from keras import layers, ops
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
]


def featuring(df: pd.DataFrame) -> pd.DataFrame:
    df["is_white"] = (df["type"] == "white").astype(np.float32)
    return df


def get_train_val(df: pd.DataFrame, features: list[str]) -> tuple:
    X = df[features]
    y = df["quality"].values.astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, y_train, y_val


def build_elasticnet_model(num_features: int):
    inputs = keras.Input(shape=(num_features,), name="input_features")
    outputs = layers.Dense(
        units=1,
        activation=None,
        kernel_regularizer=keras.regularizers.L1L2(l1=1e-3, l2=1e-2),
        name="linear_output",
    )(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ElasticNet_Baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mean_squared_error",
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def build_resnet_mlp(num_features):
    inputs = keras.Input(shape=(num_features,), name="input_features")
    x = layers.Dense(128)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    res1 = layers.Dense(128)(x)
    res1 = layers.BatchNormalization()(res1)
    res1 = layers.Activation("relu")(res1)
    res1 = layers.Dropout(0.2)(res1)
    res1 = layers.Dense(128)(res1)

    x = layers.add([x, res1])
    x = layers.Activation("relu")(x)

    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(1, activation=None, name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Tabular_ResNet")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="huber",  # Huber loss is robust to outlier ratings (3 or 9)
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


class SoftDecisionTreeBlock(layers.Layer):
    """Neural approximation of gradient tree splitting."""

    def __init__(self, num_leafs=16, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_leafs = num_leafs
        self.dense = layers.Dense(hidden_dim, activation="swish")
        self.routing = layers.Dense(num_leafs, activation="softmax")
        self.leaf_values = self.add_weight(
            shape=(num_leafs, 1), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        x = self.dense(inputs)
        weights = self.routing(x)  # (batch_size, num_leafs)
        out = ops.matmul(weights, self.leaf_values)  # (batch_size, 1)
        return out


def build_neural_tree_model(num_features):
    inputs = keras.Input(shape=(num_features,), name="input_features")

    # Construct an ensemble of 4 parallel soft trees
    tree_outputs = []
    for i in range(4):
        tree_out = SoftDecisionTreeBlock(num_leafs=32, hidden_dim=64, name=f"tree_{i}")(
            inputs
        )
        tree_outputs.append(tree_out)

    # Average the soft tree predictions (Forest Output)
    merged = layers.concatenate(tree_outputs, axis=-1)
    outputs = ops.mean(merged, axis=-1, keepdims=True)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Neural_Decision_Forest")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-3),
        loss="mean_squared_error",
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
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


def val(model: keras.Model, X_val, y_val):
    scores = model.evaluate(X_val, y_val, verbose=2)
    print("Val loss:", scores[0])
    print("Val accuracy:", scores[1])


def submit(model: keras.Model, X, ids):
    y = model(X)
    df = pd.DataFrame({"id": ids, "quality": np.round(y[..., 0]).astype(int)})
    # df.to_csv(get_incremental_path("submit/wife.csv"), index=False)


def run_with_features(features: list[str]):
    train_df = pd.read_csv("data/train.csv", sep=";")
    infer_df = pd.read_csv("data/test.csv", sep=";")

    train_df = featuring(train_df)
    infer_df = featuring(infer_df)

    X_train, X_val, y_train, y_val = get_train_val(train_df, features)
    X_infer = infer_df[features]

    num_features = len(features)
    models: list[keras.Model] = [
        build_elasticnet_model(num_features),
        build_resnet_mlp(num_features),
        build_neural_tree_model(num_features),
    ]

    for model in models:
        train(model, X_train, X_val, y_train, y_val, 10, 64)
        val(model, X_val, y_val)
        submit(model, X_infer, infer_df["id"])


def main():
    features = [
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
    ]
    run_with_features(features)


if __name__ == "__main__":
    main()
