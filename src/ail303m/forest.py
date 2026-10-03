import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import PredictionErrorDisplay
from sklearn.preprocessing import StandardScaler

from ail303m.utils import get_incremental_path

from .automation import featuring, get_train_val

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
    # "total_acidity",
    # "volatile_to_fixed_ratio",
    # "citric_to_total_acid_ratio",
    # "bound_sulfur_dioxide",
    # "free_so2_proportion",
    # "sugar_to_alcohol_ratio",
    # "sugar_to_acid_ratio",
    # "alcohol_total_acid_product",
    # "free_so2_per_pH",
]
scaler = StandardScaler()


def modeling(df):
    train_df = featuring(df)

    X, X_val, y, y_val = get_train_val(train_df, features)

    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)

    # 2. Swap LinearRegression for a tree-based ensemble model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    print(f"Validation R2 Score: {model.score(X_val, y_val)}")

    fig, ax = plt.subplots(figsize=(6, 6))
    PredictionErrorDisplay.from_estimator(
        model,
        X_val,
        y_val,
        kind="actual_vs_predicted",
        ax=ax,
        scatter_kwargs={"alpha": 0.5},
    )
    plt.title("Actual vs. Predicted Values")
    plt.show()

    return model


def predict(df, model):  # Removed strict type hinting for LinearRegression
    test_df = featuring(df)

    X = scaler.transform(test_df[features])
    y = model.predict(X)

    # 4. Optional: Since wine quality is typically an integer score (e.g., 3 to 9),
    # rounding regression predictions often yields better final classifications.
    y_rounded = np.round(y).astype(int)
    print(y_rounded[:20])

    return y_rounded


def main():
    df = pd.read_csv("data/train.csv", sep=";")

    model = modeling(df)

    test_df = pd.read_csv("data/test.csv", sep=";")
    p = predict(test_df, model)

    df = pd.DataFrame({"id": test_df["id"], "quality": p})
    df.to_csv(get_incremental_path("submit/wife-forest.csv"), index=False)

if __name__ == "__main__":
    main()
