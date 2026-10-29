from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car_details_v3.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "car_price_model.joblib"

FEATURE_COLUMNS = [
    "Brand",
    "Car_Age",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats",
]


def parse_numeric(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", text)
    if not match:
        return np.nan
    return float(match.group(0))


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    required = {"name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "seats"}
    missing = required.difference(cleaned.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned["selling_price"] = pd.to_numeric(cleaned["selling_price"], errors="coerce")
    cleaned["km_driven"] = pd.to_numeric(cleaned["km_driven"], errors="coerce")
    cleaned["mileage"] = cleaned["mileage"].map(parse_numeric)
    cleaned["engine"] = cleaned["engine"].map(parse_numeric)
    cleaned["max_power"] = cleaned["max_power"].map(parse_numeric)
    cleaned["seats"] = pd.to_numeric(cleaned["seats"], errors="coerce")
    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")

    cleaned["Brand"] = cleaned["name"].fillna("").astype(str).str.split().str[0].str.title()
    cleaned["Car_Age"] = (pd.Timestamp.now().year - cleaned["year"]).clip(lower=0)

    cleaned = cleaned[
        [
            "Brand",
            "Car_Age",
            "km_driven",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "mileage",
            "engine",
            "max_power",
            "seats",
            "selling_price",
        ]
    ].copy()

    cleaned = cleaned.dropna(subset=[
        "Brand",
        "Car_Age",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "mileage",
        "engine",
        "max_power",
        "seats",
        "selling_price",
    ]).copy()

    cleaned["fuel"] = cleaned["fuel"].astype(str).str.strip()
    cleaned["seller_type"] = cleaned["seller_type"].astype(str).str.strip()
    cleaned["transmission"] = cleaned["transmission"].astype(str).str.strip()
    cleaned["owner"] = cleaned["owner"].astype(str).str.strip()

    return cleaned


def build_pipeline() -> Pipeline:
    categorical_cols = ["Brand", "fuel", "seller_type", "transmission", "owner"]
    numeric_cols = ["Car_Age", "km_driven", "mileage", "engine", "max_power", "seats"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        ],
        remainder="passthrough",
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=16,
        min_samples_leaf=3,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_and_save_model(force_retrain: bool = False) -> Pipeline:
    if MODEL_PATH.exists() and not force_retrain:
        return load_model()

    df = clean_dataset(pd.read_csv(DATA_PATH))
    X = df[FEATURE_COLUMNS]
    y = df["selling_price"]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)


def get_reference_options() -> Dict[str, Any]:
    raw_df = pd.read_csv(DATA_PATH)
    year_values = pd.to_numeric(raw_df["year"], errors="coerce").dropna()

    return {
        "brand": sorted(clean_dataset(raw_df)["Brand"].unique().tolist()),
        "fuel": sorted(clean_dataset(raw_df)["fuel"].unique().tolist()),
        "seller_type": sorted(clean_dataset(raw_df)["seller_type"].unique().tolist()),
        "transmission": sorted(clean_dataset(raw_df)["transmission"].unique().tolist()),
        "owner": sorted(clean_dataset(raw_df)["owner"].unique().tolist()),
        "year_min": int(year_values.min()) if not year_values.empty else 2000,
        "year_max": int(pd.Timestamp.now().year),
    }


def prepare_user_input(raw_input: Dict[str, Any]) -> pd.DataFrame:
    row = {
        "Brand": raw_input["brand"],
        "Car_Age": max(0, int(raw_input["car_age"])),
        "km_driven": float(raw_input["km_driven"]),
        "fuel": raw_input["fuel"],
        "seller_type": raw_input["seller_type"],
        "transmission": raw_input["transmission"],
        "owner": raw_input["owner"],
        "mileage": float(raw_input["mileage"]),
        "engine": float(raw_input["engine"]),
        "max_power": float(raw_input["max_power"]),
        "seats": float(raw_input["seats"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_price(raw_input: Dict[str, Any]) -> float:
    model = load_model()
    row = prepare_user_input(raw_input)
    prediction = float(model.predict(row)[0])
    return max(prediction, 0.0)


def predict_price_range(raw_input: Dict[str, Any]) -> Tuple[float, float]:
    prediction = predict_price(raw_input)
    variation = max(prediction * 0.12, 15000.0)
    return prediction - variation, prediction + variation
