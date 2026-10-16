from pathlib import Path

from ail303m.utils import get_incremental_path
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
)
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

SEED = 42
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
HUBER_DELTA = 0.5
N_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 64
# "balanced": inverse-frequency weights, "sqrt": tempered, "none": no weighting
CLASS_WEIGHT_MODE = "none"


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


def build_class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if CLASS_WEIGHT_MODE == "balanced":
        weights = len(y) / (num_classes * counts)
    elif CLASS_WEIGHT_MODE == "sqrt":
        weights = np.sqrt(len(y) / (num_classes * counts))
    else:
        weights = np.ones(num_classes)
    return {int(i): float(w) for i, w in enumerate(weights)}


def correct_prior(probs: np.ndarray, prior: np.ndarray) -> np.ndarray:
    corrected = probs * prior[np.newaxis, :]
    return corrected / np.clip(corrected.sum(axis=1, keepdims=True), 1e-12, None)


def make_optimizer() -> keras.optimizers.Optimizer:
    return keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0,
    )


def make_callbacks(monitor: str) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=15, restore_best_weights=True, mode="min"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=5, min_lr=1e-6, mode="min"
        ),
    ]


def make_expected_mae(class_values: np.ndarray):
    values = keras.ops.convert_to_tensor(class_values, dtype="float32")

    def expected_mae(y_true, y_pred):
        true_value = keras.ops.take(values, keras.ops.cast(y_true, "int32"))
        expected = keras.ops.sum(y_pred * values, axis=-1)
        return keras.ops.mean(keras.ops.abs(expected - true_value))

    expected_mae.__name__ = "expected_mae"
    return expected_mae


def make_ordinal_expected_mae(class_values: np.ndarray):
    values = keras.ops.convert_to_tensor(class_values, dtype="float32")
    steps = values[1:] - values[:-1]

    def expected_mae(y_true, y_pred):
        true_value = keras.ops.take(values, keras.ops.cast(y_true, "int32"))
        expected = values[0] + keras.ops.sum(
            keras.ops.sigmoid(y_pred) * steps, axis=-1
        )
        return keras.ops.mean(keras.ops.abs(expected - true_value))

    expected_mae.__name__ = "expected_mae"
    return expected_mae


def make_sparse_ce(num_classes: int):
    def sparse_ce(y_true, y_pred):
        y_idx = keras.ops.cast(y_true, "int32")
        log_probs = keras.ops.log(keras.ops.clip(y_pred, 1e-7, 1.0))
        target = keras.ops.one_hot(y_idx, num_classes)
        target = target * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / num_classes
        return -keras.ops.sum(target * log_probs, axis=-1)

    sparse_ce.__name__ = "sparse_categorical_crossentropy"
    return sparse_ce


def build_classifier(input_dim: int, num_classes: int, class_values: np.ndarray) -> keras.Model:
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
        optimizer=make_optimizer(),
        loss=make_sparse_ce(num_classes),
        metrics=["accuracy", make_expected_mae(class_values)],
    )
    return model


def build_ordinal_classifier(
    input_dim: int,
    num_classes: int,
    class_values: np.ndarray,
    class_weights: dict[int, float],
) -> keras.Model:
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
            keras.layers.Dense(num_classes - 1),
        ]
    )
    thresholds = keras.ops.arange(num_classes - 1)
    weights = keras.ops.convert_to_tensor(
        np.array([class_weights[i] for i in range(num_classes)], dtype="float32")
    )

    def ordinal_loss(y_true, y_pred):
        y_idx = keras.ops.cast(y_true, "int32")
        target = keras.ops.cast(
            keras.ops.reshape(thresholds, (1, -1))
            < keras.ops.reshape(y_idx, (-1, 1)),
            y_pred.dtype,
        )
        target = target * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        bce = keras.ops.binary_crossentropy(target, y_pred, from_logits=True)
        sample_weight = keras.ops.take(weights, y_idx)
        return keras.ops.mean(keras.ops.mean(bce, axis=-1) * sample_weight)

    model.compile(
        optimizer=make_optimizer(),
        loss=ordinal_loss,
        metrics=[make_ordinal_expected_mae(class_values)],
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
    model.compile(
        optimizer=make_optimizer(),
        loss=keras.losses.Huber(delta=HUBER_DELTA),
        metrics=["mae"],
    )
    return model


def ordinal_to_pmf(logits: np.ndarray) -> np.ndarray:
    exceed = 1.0 / (1.0 + np.exp(-logits))
    pmf = np.concatenate(
        [1.0 - exceed[:, :1], exceed[:, :-1] - exceed[:, 1:], exceed[:, -1:]],
        axis=1,
    )
    pmf = np.clip(pmf, 1e-8, None)
    return pmf / pmf.sum(axis=1, keepdims=True)


def report_history(model: keras.Model, name: str, monitor: str) -> None:
    values = model.history.history[monitor]
    best = int(np.argmin(values))
    print(f"  {name}: best {monitor}={values[best]:.4f} @ epoch {best + 1}")


def describe(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name:<16} acc={acc:.4f}  bal_acc={bal:.4f}  macro_f1={f1:.4f}  mae={mae:.4f}")
    return acc


def main():
    keras.utils.set_random_seed(SEED)

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
    classes = label_encoder.classes_
    class_values = classes.astype(np.float32)
    num_classes = len(classes)
    y_true = train["quality"].values

    prior = np.bincount(y, minlength=num_classes) / len(y)
    class_weight = build_class_weights(y, num_classes)
    weighted = CLASS_WEIGHT_MODE != "none"
    print(
        f"Class weights ({CLASS_WEIGHT_MODE}):",
        np.round([class_weight[i] for i in range(num_classes)], 2),
    )

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    val_probs = np.zeros((len(X), num_classes), dtype=np.float32)
    val_ord_logits = np.zeros((len(X), num_classes - 1), dtype=np.float32)
    val_reg = np.zeros(len(X), dtype=np.float32)
    test_probs = np.zeros((len(X_test), num_classes), dtype=np.float32)
    test_ord_logits = np.zeros((len(X_test), num_classes - 1), dtype=np.float32)
    test_reg = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n===== Fold {fold} =====")

        clf = build_classifier(X.shape[1], num_classes, class_values)
        clf.fit(
            X[tr_idx],
            y[tr_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight if weighted else None,
            callbacks=make_callbacks("val_expected_mae"),
            verbose=0,
        )
        report_history(clf, "softmax", "val_expected_mae")
        val_probs[val_idx] = clf.predict(X[val_idx], verbose=0)
        test_probs += clf.predict(X_test, verbose=0) / N_FOLDS

        ordinal = build_ordinal_classifier(
            X.shape[1], num_classes, class_values, class_weight
        )
        ordinal.fit(
            X[tr_idx],
            y[tr_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=make_callbacks("val_expected_mae"),
            verbose=0,
        )
        report_history(ordinal, "ordinal", "val_expected_mae")
        val_ord_logits[val_idx] = ordinal.predict(X[val_idx], verbose=0)
        test_ord_logits += ordinal.predict(X_test, verbose=0) / N_FOLDS

        reg = build_regressor(X.shape[1])
        reg.fit(
            X[tr_idx],
            y_raw[tr_idx],
            validation_data=(X[val_idx], y_raw[val_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=make_callbacks("val_mae"),
            verbose=0,
        )
        report_history(reg, "regressor", "val_mae")
        val_reg[val_idx] = reg.predict(X[val_idx], verbose=0).ravel()
        test_reg += reg.predict(X_test, verbose=0).ravel() / N_FOLDS

    if weighted:
        print(
            "\nUncorrected softmax argmax OOF acc: "
            f"{accuracy_score(y_true, classes[val_probs.argmax(axis=1)]):.4f}"
        )
        val_probs = correct_prior(val_probs, prior)
        test_probs = correct_prior(test_probs, prior)
    val_ord = ordinal_to_pmf(val_ord_logits)
    test_ord = ordinal_to_pmf(test_ord_logits)
    if weighted:
        val_ord = correct_prior(val_ord, prior)
        test_ord = correct_prior(test_ord, prior)

    def as_int(scores: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(scores), classes.min(), classes.max()).astype(int)

    val_scores = {
        "clf_expect": val_probs @ classes,
        "ord_expect": val_ord @ classes,
        "regressor": val_reg,
    }
    test_scores = {
        "clf_expect": test_probs @ classes,
        "ord_expect": test_ord @ classes,
        "regressor": test_reg,
    }

    candidates = {
        "clf_argmax": classes[val_probs.argmax(axis=1)].astype(int),
        "ord_argmax": classes[val_ord.argmax(axis=1)].astype(int),
    }
    test_preds = {
        "clf_argmax": classes[test_probs.argmax(axis=1)].astype(int),
        "ord_argmax": classes[test_ord.argmax(axis=1)].astype(int),
    }
    for name in val_scores:
        candidates[name] = as_int(val_scores[name])
        test_preds[name] = as_int(test_scores[name])

    for base_name in ("clf_expect", "ord_expect"):
        best_acc, best_w = -1.0, 0.5
        for w in np.linspace(0, 1, 21):
            blended = as_int(
                w * val_scores[base_name] + (1 - w) * val_scores["regressor"]
            )
            acc = accuracy_score(y_true, blended)
            if acc > best_acc:
                best_acc, best_w = acc, w
        blend_name = base_name.replace("_expect", "_blend")
        candidates[blend_name] = as_int(
            best_w * val_scores[base_name] + (1 - best_w) * val_scores["regressor"]
        )
        test_preds[blend_name] = as_int(
            best_w * test_scores[base_name] + (1 - best_w) * test_scores["regressor"]
        )
        print(f"Blend {base_name}+regressor w={best_w:.2f} -> OOF acc {best_acc:.4f}")

    print("\nCandidate OOF metrics:")
    scores = {
        name: describe(name, y_true, pred) for name, pred in candidates.items()
    }
    best_name = max(scores, key=lambda k: scores[k])
    print(f"\nSelected '{best_name}' (OOF acc {scores[best_name]:.4f})")
    print(classification_report(y_true, candidates[best_name], zero_division=0, digits=3))

    out_path = get_incremental_path(str(DATA_DIR / "submission.csv"))
    pd.DataFrame(
        {"id": test["id"].astype(int), "quality": test_preds[best_name]}
    ).to_csv(out_path, index=False)
    print(f"Saved '{best_name}' predictions to {out_path}")


if __name__ == "__main__":
    main()
