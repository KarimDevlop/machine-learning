"""Shared data and helpers for the car-price lesson notebooks."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data.csv"
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False)

for column in df.select_dtypes(include=["object", "string"]).columns:
    df[column] = df[column].str.lower().str.replace(" ", "_", regex=False)

price_logs = np.log1p(df["msrp"])

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test
idx = np.random.RandomState(2).permutation(n)

df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
df_val = df.iloc[idx[n_train : n_train + n_val]].reset_index(drop=True)
df_test = df.iloc[idx[n_train + n_val :]].reset_index(drop=True)

y_train = np.log1p(df_train["msrp"].to_numpy())
y_val = np.log1p(df_val["msrp"].to_numpy())
y_test = np.log1p(df_test["msrp"].to_numpy())

df_train = df_train.drop(columns=["msrp"])
df_val = df_val.drop(columns=["msrp"])
df_test = df_test.drop(columns=["msrp"])

base = [
    "engine_hp",
    "engine_cylinders",
    "highway_mpg",
    "city_mpg",
    "popularity",
]

categorical_variables = [
    "make",
    "engine_fuel_type",
    "transmission_type",
    "driven_wheels",
    "market_category",
    "vehicle_size",
    "vehicle_style",
]
categories = {
    column: list(df_train[column].value_counts().head().index)
    for column in categorical_variables
}
makes = categories["make"]
features = base + ["age"]
features += [f"num_doors_{doors}" for doors in (2, 3, 4)]
features += [
    f"{column}_{value}"
    for column, values in categories.items()
    for value in values
]


def prepare_X(data: pd.DataFrame) -> np.ndarray:
    """Build the lesson's full feature matrix, filling missing values with zero."""
    prepared = data.copy()
    prepared["age"] = 2017 - prepared["year"]

    for doors in (2, 3, 4):
        prepared[f"num_doors_{doors}"] = (
            prepared["number_of_doors"] == doors
        ).astype(int)

    for column, values in categories.items():
        for value in values:
            prepared[f"{column}_{value}"] = (prepared[column] == value).astype(int)

    return prepared[features].fillna(0).to_numpy()


def train_linear_regression(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit linear regression with the normal equation."""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    return weights[0], weights[1:]


def train_linear_regression_reg(
    X: np.ndarray, y: np.ndarray, r: float = 0.001
) -> tuple[float, np.ndarray]:
    """Fit regularized linear regression with the normal equation."""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    gram = X_with_bias.T @ X_with_bias
    gram += r * np.eye(gram.shape[0])
    weights = np.linalg.inv(gram) @ X_with_bias.T @ y
    return weights[0], weights[1:]


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the root mean squared error."""
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))


X_train = prepare_X(df_train)
X_val = prepare_X(df_val)
X_test = prepare_X(df_test)
