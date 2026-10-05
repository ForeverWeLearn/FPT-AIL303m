from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

BASE_FEATURES = [
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
]

EXTRA_FEATURES = [
    "is_white",
    "acidity_total",
    "sulfur_ratio",
    "alcohol_acidity",
    "alcohol_density",
    "sugar_alcohol",
    "chlorides_sulphates",
    "density_ph",
    "volatile_fixed",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_white"] = (df["type"] == "white").astype(np.float32)
    df["acidity_total"] = (
        df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    )
    df["sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
    df["alcohol_acidity"] = df["alcohol"] / (df["acidity_total"] + 1e-6)
    df["alcohol_density"] = df["alcohol"] / (df["density"] + 1e-6)
    df["sugar_alcohol"] = df["residual sugar"] / (df["alcohol"] + 1e-6)
    df["chlorides_sulphates"] = df["chlorides"] * df["sulphates"]
    df["density_ph"] = df["density"] * df["pH"]
    df["volatile_fixed"] = df["volatile acidity"] / (df["fixed acidity"] + 1e-6)
    return df


def build_classifier(input_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_regressor(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber", metrics=["mae"])
    return model


def main():
    keras.utils.set_random_seed(42)

    train = add_features(pd.read_csv(DATA_DIR / "train.csv", sep=";"))
    test = add_features(pd.read_csv(DATA_DIR / "test.csv", sep=";"))

    cols = BASE_FEATURES + EXTRA_FEATURES
    X = train[cols].values.astype(np.float32)
    X_test = test[cols].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train["quality"].values)
    y_raw = train["quality"].values.astype(np.float32)
    classes = label_encoder.classes_  # sorted quality values
    num_classes = len(classes)

    callbacks = lambda: [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_probs = np.zeros((len(X), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(X_test), num_classes), dtype=np.float32)
    val_reg = np.zeros(len(X), dtype=np.float32)
    test_reg = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n===== Fold {fold} =====")
        # Classifier
        clf = build_classifier(X.shape[1], num_classes)
        clf.fit(
            X[tr_idx],
            y[tr_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=200,
            batch_size=64,
            callbacks=callbacks(),
            verbose=0,
        )
        val_probs[val_idx] = clf.predict(X[val_idx], verbose=0)
        test_probs += clf.predict(X_test, verbose=0) / kf.n_splits

        # Regressor
        reg = build_regressor(X.shape[1])
        reg.fit(
            X[tr_idx],
            y_raw[tr_idx],
            validation_data=(X[val_idx], y_raw[val_idx]),
            epochs=200,
            batch_size=64,
            callbacks=callbacks(),
            verbose=0,
        )
        val_reg[val_idx] = reg.predict(X[val_idx], verbose=0).ravel()
        test_reg += reg.predict(X_test, verbose=0).ravel() / kf.n_splits

    # Approach A: classifier argmax (expected-value refinement also tried)
    clf_argmax = classes[val_probs.argmax(axis=1)]
    clf_expect = np.rint(val_probs @ classes).astype(int)
    reg_pred = np.clip(np.rint(val_reg), classes.min(), classes.max()).astype(int)

    acc_clf = accuracy_score(train["quality"], clf_argmax)
    acc_exp = accuracy_score(train["quality"], clf_expect)
    acc_reg = accuracy_score(train["quality"], reg_pred)
    print(f"Val acc classifier argmax : {acc_clf:.4f}")
    print(f"Val acc classifier expect : {acc_exp:.4f}")
    print(f"Val acc regressor         : {acc_reg:.4f}")

    # Blend: average of classifier expected value and regressor prediction
    test_clf_expect = test_probs @ classes
    best_blend = None
    best_blend_acc = -1
    for w in np.linspace(0, 1, 11):
        blended = np.clip(
            np.rint(w * (val_probs @ classes) + (1 - w) * val_reg),
            classes.min(),
            classes.max(),
        ).astype(int)
        acc = accuracy_score(train["quality"], blended)
        if acc > best_blend_acc:
            best_blend_acc, best_blend = acc, w
    print(f"Val acc blend (w={best_blend:.1f} on classifier): {best_blend_acc:.4f}")

    candidates = {
        "clf_argmax": (acc_clf, classes[test_probs.argmax(axis=1)]),
        "clf_expect": (
            acc_exp,
            np.clip(np.rint(test_clf_expect), classes.min(), classes.max()).astype(int),
        ),
        "regressor": (
            acc_reg,
            np.clip(np.rint(test_reg), classes.min(), classes.max()).astype(int),
        ),
        "blend": (
            best_blend_acc,
            np.clip(
                np.rint(best_blend * test_clf_expect + (1 - best_blend) * test_reg),
                classes.min(),
                classes.max(),
            ).astype(int),
        ),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0])
    print(f"Using '{best_name}' strategy (val acc {candidates[best_name][0]:.4f})")
    test_final = candidates[best_name][1]

    out_path = DATA_DIR / "submission.csv"
    with open(out_path, "w") as f:
        f.writelines(["id, quality"])
        for test_id, q in zip(test["id"], test_final):
            f.writelines([f"{int(test_id)}, {int(q)}"])
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
