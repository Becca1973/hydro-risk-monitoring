from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Hydro Risk Monitoring API")
STATIONS_PATH = Path("data/metadata/hidro_stations.csv")

HIDRO_DIR = Path("data/preprocessed/hidro")
WEATHER_DIR = Path("data/preprocessed/weather")

WATER_MODEL_PATH = Path("models/hidro_global/model_hidro_global.keras")
WATER_PIPELINE_PATH = Path("models/hidro_global/pipeline_hidro_global.pkl")

RISK_MODEL_PATH = Path("models/risk/risk_classifier.keras")
RISK_PREPROCESSING_PATH = Path("models/risk/risk_classifier_preprocessing.pkl")
RISK_MAPPING_PATH = Path(
    "reports/modeling/risk_classifier_used_station_mapping.csv")

WATER_FEATURE_COLS = [
    "vodostaj",
    "pretok",
    "temp_vode",
    "station_code",
    "hour",
    "day",
    "month",
    "dayofweek",
]

COORDINATES_PATH = Path("data/metadata/station_coordinates.csv")

WINDOW_SIZE = 24

water_model = None
water_pipeline = None
risk_model = None
risk_preprocessing = None
risk_mapping = None


@app.on_event("startup")
def load_models():
    global water_model, water_pipeline, risk_model, risk_preprocessing, risk_mapping

    print("=== MODEL STARTUP CHECK ===")
    print("WATER_MODEL_PATH:", WATER_MODEL_PATH,
          "exists:", WATER_MODEL_PATH.exists())
    print("WATER_PIPELINE_PATH:", WATER_PIPELINE_PATH,
          "exists:", WATER_PIPELINE_PATH.exists())
    print("RISK_MODEL_PATH:", RISK_MODEL_PATH,
          "exists:", RISK_MODEL_PATH.exists())
    print("RISK_PREPROCESSING_PATH:", RISK_PREPROCESSING_PATH,
          "exists:", RISK_PREPROCESSING_PATH.exists())
    print("RISK_MAPPING_PATH:", RISK_MAPPING_PATH,
          "exists:", RISK_MAPPING_PATH.exists())

    if WATER_MODEL_PATH.exists():
        try:
            water_model = tf.keras.models.load_model(WATER_MODEL_PATH)
            print("water model loaded")
        except Exception as e:
            print("water model error:", e)

    if WATER_PIPELINE_PATH.exists():
        try:
            with open(WATER_PIPELINE_PATH, "rb") as f:
                water_pipeline = pickle.load(f)
            print("water pipeline loaded")
        except Exception as e:
            print("water pipeline error:", e)

    if RISK_MODEL_PATH.exists():
        try:
            risk_model = tf.keras.models.load_model(RISK_MODEL_PATH)
            print("risk model loaded")
        except Exception as e:
            print("risk model error:", e)

    if RISK_PREPROCESSING_PATH.exists():
        try:
            with open(RISK_PREPROCESSING_PATH, "rb") as f:
                risk_preprocessing = pickle.load(f)
            print("risk preprocessing loaded")
        except Exception as e:
            print("risk preprocessing error:", e)

    if RISK_MAPPING_PATH.exists():
        try:
            risk_mapping = pd.read_csv(RISK_MAPPING_PATH)
            print("risk mapping loaded")
        except Exception as e:
            print("risk mapping error:", e)

    print("=== MODEL STARTUP CHECK END ===")


@app.get("/")
def root():
    return {"message": "Hydro Risk Monitoring API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stations")
def stations():
    df = pd.read_csv(STATIONS_PATH)
    return {
        "stations": sorted(df["station"].tolist())
    }


@app.get("/models")
def models():
    return {
        "water_level_model": {
            "name": "hidro_global_water_level_forecaster",
            "loaded": water_model is not None,
            "type": "LSTM regression",
        },
        "risk_model": {
            "name": "hidro_risk_classifier",
            "loaded": risk_model is not None,
            "type": "Dense neural network classification",
        },
    }


def find_hidro_file(station: str) -> Path:
    path = HIDRO_DIR / f"{station}.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Station {station} not found.")
    return path


def normalize_name(name: str) -> str:
    return (
        str(name)
        .upper()
        .replace("Č", "C")
        .replace("Š", "S")
        .replace("Ž", "Z")
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_weather_file(weather_station: str) -> Path:
    weather_station_norm = normalize_name(weather_station)

    for path in WEATHER_DIR.glob("*.csv"):
        if normalize_name(path.stem) == weather_station_norm:
            return path

    raise HTTPException(
        status_code=404,
        detail=f"Weather station {weather_station} not found.",
    )


@app.get("/predict/water-level/{station}")
def predict_water_level(station: str):
    if water_model is None or water_pipeline is None:
        raise HTTPException(
            status_code=500, detail="Water level model is not loaded.")

    hidro_path = find_hidro_file(station)
    df = pd.read_csv(hidro_path)

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum", "vodostaj"])
    df = df.sort_values("datum")

    if len(df) < WINDOW_SIZE:
        raise HTTPException(
            status_code=400, detail="Not enough data for prediction.")

    station_code = station.split("_")[0]

    df["station_code"] = pd.to_numeric(station_code, errors="coerce")
    df["hour"] = df["datum"].dt.hour
    df["day"] = df["datum"].dt.day
    df["month"] = df["datum"].dt.month
    df["dayofweek"] = df["datum"].dt.dayofweek

    for col in WATER_FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    recent_df = df[WATER_FEATURE_COLS].tail(WINDOW_SIZE)

    processed = water_pipeline.transform(recent_df)
    X = np.array([processed], dtype=float)

    prediction_scaled = float(water_model.predict(X, verbose=0).flatten()[0])

    scaler = water_pipeline.named_steps["scaler"]
    target_idx = WATER_FEATURE_COLS.index("vodostaj")

    dummy = np.zeros((1, len(WATER_FEATURE_COLS)))
    dummy[0, target_idx] = prediction_scaled

    prediction_original = scaler.inverse_transform(dummy)[0, target_idx]

    latest_row = df.iloc[-1]

    return {
        "station": station,
        "latest_timestamp": str(latest_row["datum"]),
        "latest_water_level": float(latest_row["vodostaj"]),
        "predicted_water_level": round(float(prediction_original), 2),
        "unit": "cm",
        "model": "hidro_global_water_level_forecaster",
    }


@app.get("/predict/risk/{station}")
def predict_risk(station: str):
    if risk_model is None or risk_preprocessing is None:
        raise HTTPException(
            status_code=500, detail="Risk model is not loaded.")

    if risk_mapping is None:
        raise HTTPException(
            status_code=500, detail="Risk mapping is not available.")

    row = risk_mapping[risk_mapping["hidro_station"] == station]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail="Risk prediction is not available for this station.",
        )

    weather_station = row.iloc[0]["matched_weather_station"]

    hidro_path = find_hidro_file(station)
    weather_path = find_weather_file(weather_station)

    hidro_df = pd.read_csv(hidro_path)
    weather_df = pd.read_csv(weather_path)

    hidro_df["datum"] = pd.to_datetime(hidro_df["datum"], errors="coerce")
    hidro_df = hidro_df.dropna(subset=["datum", "vodostaj"])
    hidro_df = hidro_df.sort_values("datum")

    weather_df["timestamp_local"] = (
        weather_df["timestamp_local"]
        .astype(str)
        .str.replace(" CEST", "", regex=False)
        .str.replace(" CET", "", regex=False)
    )

    weather_df["datum"] = pd.to_datetime(
        weather_df["timestamp_local"],
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )

    weather_df = weather_df.sort_values("datum")

    latest_hidro = hidro_df.tail(1).copy()

    station_code = station.split("_")[0]
    latest_hidro["station_code"] = pd.to_numeric(station_code, errors="coerce")

    merged = pd.merge_asof(
        latest_hidro,
        weather_df,
        on="datum",
        direction="nearest",
        tolerance=pd.Timedelta("90min"),
    )

    if merged.empty or pd.isna(merged.iloc[0].get("temperature")):
        raise HTTPException(
            status_code=400,
            detail="No recent matching weather data found.",
        )

    feature_cols = risk_preprocessing["feature_cols"]

    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    X = merged[feature_cols]

    imputer = risk_preprocessing["imputer"]
    scaler = risk_preprocessing["scaler"]
    label_encoder = risk_preprocessing["label_encoder"]

    X_processed = imputer.transform(X)
    X_processed = scaler.transform(X_processed)

    probabilities = risk_model.predict(X_processed, verbose=0)
    prediction_idx = int(np.argmax(probabilities, axis=1)[0])
    risk_level = label_encoder.inverse_transform([prediction_idx])[0]

    confidence = float(np.max(probabilities))

    return {
        "station": station,
        "weather_station": weather_station,
        "risk_level": risk_level,
        "confidence": round(confidence, 3),
        "model": "hidro_risk_classifier",
    }


@app.get("/predict/full/{station}")
def predict_full(station: str):
    water = predict_water_level(station)

    try:
        risk = predict_risk(station)
    except HTTPException:
        risk = {
            "station": station,
            "risk_level": "N/A",
            "confidence": None,
            "message": "Risk prediction is not available for this station.",
        }

    return {
        "station": station,
        "water_level_prediction": water,
        "risk_prediction": risk,
    }


@app.get("/history/{station}")
def station_history(station: str, limit: int = 168):
    hidro_path = find_hidro_file(station)

    df = pd.read_csv(hidro_path)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum", "vodostaj"])
    df = df.sort_values("datum").tail(limit)

    history = df[["datum", "vodostaj"]].copy()
    history["datum"] = history["datum"].astype(str)

    return {
        "station": station,
        "records": history.to_dict(orient="records"),
    }


@app.get("/stations/map")
def stations_map():
    if not COORDINATES_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Station coordinates file not found.",
        )

    coordinates = pd.read_csv(COORDINATES_PATH)

    if STATIONS_PATH.exists():
        stations_df = pd.read_csv(STATIONS_PATH)
        available_stations = set(stations_df["station"].dropna().tolist())
    else:
        available_stations = set(
            [path.stem for path in HIDRO_DIR.glob("*.csv")])

    coordinates = coordinates[
        coordinates["station"].isin(available_stations)
    ].copy()

    return {
        "stations": coordinates.to_dict(orient="records")
    }
