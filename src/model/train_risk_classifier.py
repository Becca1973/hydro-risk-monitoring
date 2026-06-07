from pathlib import Path
import os
import pickle
import random

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


HIDRO_TO_WEATHER = {
    # Ptuj / Drava
    "2110_Ptuj": "PTUJ",
    "2150_Borl": "PTUJ",
    "2160_Zavrc": "PTUJ",
    "2719_Podlehnik": "PTUJ",
    "2754_Trzec": "PTUJ",

    # Pomurje
    "1070_Petanjci": "GORICKO",
    "1100_Cankova": "GORICKO",
    "1140_Pristava": "GORICKO",
    "1165_Nuskova": "GORICKO",
    "1220_Polana": "GORICKO",
    "1260_Centiba": "Lendava",
    "1300_Martjanci": "GORICKO",
    "1312_Kobilje": "Lendava",
    "1335_Sredisce": "Lendava",
    "1355_Hodos": "GORICKO",

    # Celje / Savinja
    "6020_Solcava": "RADEGUNDA",
    "6060_Nazarje": "RADEGUNDA",
    "6068_Letus": "CELJE",
    "6120_Medlog": "CELJE",
    "6200_Lasko": "CELJE",
    "6220_Luce": "RADEGUNDA",
    "6240_Krase": "CELJE",
    "6280_Velenje": "VELENJE",
    "6300_Sostanj": "VELENJE",
    "6340_Recica": "CELJE",
    "6415_Gaberke": "VELENJE",
    "6630_Levec": "CELJE",
    "6691_Crnolica": "CELJE",
    "6720_Celje": "CELJE",
    "6770_Polze": "CELJE",

    # Ljubljana / Gorenjska / Sava
    "3420_Radovljica": "LESCE",
    "3465_Okroglo": "KRANJ",
    "3530_Medno": "KRANJ",
    "3570_Sentjakob": "LITIJA",
    "3660_Litija": "LITIJA",
    "3725_Hrastnik": "HRASTNIK",
    "3850_Catez": "Krsko",
    "4025_Ovsise": "LESCE",
    "4050_Preska": "KRANJ",
    "4095_Lajb": "KRANJ",
    "4120_Kokra": "JEZERSKO",
    "4155_Kranj": "KRANJ",
    "4200_Suha": "KRANJ",
    "4209_Medvode": "KRANJ",
    "4222_Ziri": "TOPOL",
    "4230_Zminec": "KRANJ",
    "4270_Zelezniki": "RATITOVEC",
    "4298_Vester": "KRANJ",

    # Kamnik / okolica
    "4400_Kamnik": "KRVAVEC",
    "4430_Vir": "KRVAVEC",
    "4445_Bisce": "KRVAVEC",
    "4480_Nevlje": "KRVAVEC",
    "4515_Vir": "KRVAVEC",
    "4520_Podrecje": "KRVAVEC",
    "4570_Topole": "KRVAVEC",
    "4575_Loka": "KRVAVEC",

    # Notranjska / Ljubljanica
    "5030_Vrhnika": "VRHNIKA",
    "5040_Kamin": "VRHNIKA",
    "5078_Moste": "VRHNIKA",
    "5240_Verd": "VRHNIKA",
    "5270_Bistra": "VRHNIKA",
    "5330_Borovnica": "VRHNIKA",
    "5440_Ig": "VRHNIKA",
    "5479_Bokalce": "VRHNIKA",
    "5500_Dvor": "VRHNIKA",
    "5540_Razori": "VRHNIKA",
    "5770_Cerknica": "POSTOJNA",
    "5800_Prestranek": "POSTOJNA",
    "5880_Hasberg": "POSTOJNA",
    "5910_Malni": "POSTOJNA",
    "5940_Logatec": "LOGATEC",

    # Dolenjska / Bela krajina
    "4820_Petrina": "KOCEVJE",
    "4828_Sodevci": "KOCEVJE",
    "4860_Metlika": "METLIKA",
    "4960_Livold": "KOCEVJE",
    "4969_Gradac": "METLIKA",
    "4986_Dolenjce": "METLIKA",
    "7029_Podbukovje": "TREBNJE",
    "7060_Soteska": "TREBNJE",
    "7160_Podbocje": "Krsko",
    "7200_Mlacevo": "TREBNJE",
    "7220_Rasica": "TREBNJE",
    "7230_Gradicek": "TREBNJE",
    "7245_Fuzina": "TREBNJE",
    "7340_Precna": "TREBNJE",
    "7350_Stopice": "TREBNJE",
    "7380_Skocjan": "SKOCJAN",
    "7440_Sodrazica": "KOCEVJE",
    "7488_Prigorica": "KOCEVJE",
    "7498_Blate": "KOCEVJE",

    # Primorska / Vipava / Soča / Obala
    "8031_Krsovec": "BOVEC",
    "8080_Kobarid": "BOVEC",
    "8180_Solkan": "BILJE",
    "8270_Zaga": "BOVEC",
    "8332_Tolmin": "KRN",
    "8350_Podroteja": "OTLICA",
    "8351_Podroteja": "OTLICA",
    "8450_Hotesk": "OTLICA",
    "8453_Podroteja": "OTLICA",
    "8454_Cerkno": "DAVCA",
    "8561_Vipava": "PODNANOS",
    "8565_Dolenje": "PODNANOS",
    "8591_Zalosce": "BILJE",
    "8601_Miren": "BILJE",
    "8610_Podnanos": "PODNANOS",
    "8630_Ajdovscina": "PODNANOS",
    "8640_Branik": "BILJE",
    "8670_Bezovljak": "BILJE",
    "8680_Neblo": "BILJE",
    "8700_Neblo": "BILJE",
    "8710_Potoki": "BILJE",

    # Kras / Obala
    "9015_Trpcane": "POSTOJNA",
    "9030_Trnovo": "POSTOJNA",
    "9077_Skocjan": "SKOCJAN",
    "9108_Zarecica": "POSTOJNA",
    "9210_Kubed": "KUBED",
    "9240_Dekani": "KOPER",
    "9275_Salara": "KOPER",
    "9280_Pisine": "KOPER",
    "9300_Podkastel": "KOPER",
}


def normalize_name(name: str) -> str:
    return (
        name.upper()
        .replace("Č", "C")
        .replace("Š", "S")
        .replace("Ž", "Z")
        .replace(" ", "_")
        .replace("-", "_")
    )


def load_weather_data(weather_dir: Path) -> pd.DataFrame:
    frames = []

    for csv_path in sorted(weather_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)

        if "timestamp_local" not in df.columns or "station" not in df.columns:
            continue

        df["timestamp_local"] = (
            df["timestamp_local"]
            .astype(str)
            .str.replace(" CEST", "", regex=False)
            .str.replace(" CET", "", regex=False)
        )

        df["datum"] = pd.to_datetime(
            df["timestamp_local"],
            format="%d.%m.%Y %H:%M",
            errors="coerce",
        )

        df["weather_station_norm"] = df["station"].astype(
            str).apply(normalize_name)

        frames.append(df)

    if not frames:
        raise ValueError("No valid weather data found.")

    return pd.concat(frames, ignore_index=True)


def create_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    q75 = df.groupby("hidro_station")["vodostaj"].transform(
        lambda x: x.quantile(0.75)
    )
    q90 = df.groupby("hidro_station")["vodostaj"].transform(
        lambda x: x.quantile(0.90)
    )

    conditions = [
        df["vodostaj"] < q75,
        (df["vodostaj"] >= q75) & (df["vodostaj"] < q90),
        df["vodostaj"] >= q90,
    ]

    df["risk_level"] = np.select(
        conditions,
        ["LOW", "MEDIUM", "HIGH"],
        default="LOW",
    )

    return df


def resolve_weather_station(hidro_station: str) -> str:
    if hidro_station in HIDRO_TO_WEATHER:
        return normalize_name(HIDRO_TO_WEATHER[hidro_station])

    station_label = hidro_station.split(
        "_", 1)[1] if "_" in hidro_station else hidro_station
    return normalize_name(station_label)


def load_and_merge_data(
    hidro_dir: Path,
    weather_dir: Path,
    params: dict,
) -> pd.DataFrame:
    weather_df = load_weather_data(weather_dir)

    merged_frames = []
    skipped = []

    hidro_feature_cols = params["feature_cols"]
    required_hidro_cols = ["datum"] + hidro_feature_cols

    for hidro_path in sorted(hidro_dir.glob("*.csv")):
        station_name = hidro_path.stem
        weather_station_norm = resolve_weather_station(station_name)

        hidro_df = pd.read_csv(hidro_path)

        missing_cols = [
            col for col in required_hidro_cols
            if col not in hidro_df.columns
        ]

        if missing_cols:
            skipped.append(
                {
                    "station": station_name,
                    "weather_station": weather_station_norm,
                    "reason": f"missing columns: {missing_cols}",
                }
            )
            continue

        hidro_df = hidro_df[required_hidro_cols].copy()
        hidro_df["datum"] = pd.to_datetime(hidro_df["datum"], errors="coerce")
        hidro_df = hidro_df.dropna(subset=["datum", "vodostaj"])
        hidro_df = hidro_df.sort_values("datum")

        for col in hidro_feature_cols:
            hidro_df[col] = pd.to_numeric(hidro_df[col], errors="coerce")

        if len(hidro_df) < params["min_rows"]:
            skipped.append(
                {
                    "station": station_name,
                    "weather_station": weather_station_norm,
                    "reason": "not enough hidro rows",
                }
            )
            continue

        matching_weather = weather_df[
            weather_df["weather_station_norm"] == weather_station_norm
        ].copy()

        if matching_weather.empty:
            skipped.append(
                {
                    "station": station_name,
                    "weather_station": weather_station_norm,
                    "reason": "no matching weather station",
                }
            )
            continue

        matching_weather = matching_weather.sort_values("datum")

        hidro_df["hidro_station"] = station_name
        station_code = station_name.split("_")[0]
        hidro_df["station_code"] = pd.to_numeric(station_code, errors="coerce")

        # Merge hidro measurement with nearest weather measurement within 90 minutes.
        merged = pd.merge_asof(
            hidro_df,
            matching_weather,
            on="datum",
            direction="nearest",
            tolerance=pd.Timedelta("90min"),
        )

        merged = merged.dropna(subset=["temperature"])

        if len(merged) < params["min_rows"]:
            skipped.append(
                {
                    "station": station_name,
                    "weather_station": weather_station_norm,
                    "reason": "not enough merged rows",
                }
            )
            continue

        merged["matched_weather_station"] = weather_station_norm
        merged_frames.append(merged)

    Path("reports/modeling").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(skipped).to_csv(
        "reports/modeling/risk_classifier_skipped_stations.csv",
        index=False,
    )

    if not merged_frames:
        raise ValueError("No hidro/weather station pairs could be merged.")

    full_df = pd.concat(merged_frames, ignore_index=True)
    full_df = create_risk_level(full_df)

    used_mapping = (
        full_df[["hidro_station", "matched_weather_station"]]
        .drop_duplicates()
        .sort_values("hidro_station")
    )
    used_mapping.to_csv(
        "reports/modeling/risk_classifier_used_station_mapping.csv",
        index=False,
    )

    return full_df


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]
    random_state = train_params.get("random_state", 42)

    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    hidro_dir = Path(train_params["hidro_input_dir"])
    weather_dir = Path(train_params["weather_input_dir"])
    output_dir = Path(train_params.get(
        "risk_models_output_dir", "models/risk"))
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("hidro-risk-classification")

    print("Loading and merging hidro + weather data...")
    df = load_and_merge_data(hidro_dir, weather_dir, train_params)

    print(f"Merged rows: {len(df)}")
    print(f"Stations used: {df['hidro_station'].nunique()}")
    print(df["risk_level"].value_counts())

    print("Used stations:")
    print(sorted(df["hidro_station"].unique()))

    feature_cols = [
        "vodostaj",
        "pretok",
        "temp_vode",
        "station_code",
        "temperature",
        "dew_point",
        "rel_humidity",
        "precipitation",
        "precipitation_1h",
        "precipitation_12h",
        "wind_speed",
        "wind_direction",
        "snow",
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["risk_level"])
    df = df.sort_values("datum")

    test_size = min(
        train_params["test_size"] * max(df["hidro_station"].nunique(), 1),
        int(len(df) * 0.2),
    )

    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    X_train = train_df[feature_cols]
    y_train = train_df["risk_level"]

    X_test = test_df[feature_cols]
    y_test = test_df["risk_level"]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()

    X_train_processed = imputer.fit_transform(X_train)
    X_train_processed = scaler.fit_transform(X_train_processed)

    X_test_processed = imputer.transform(X_test)
    X_test_processed = scaler.transform(X_test_processed)

    num_classes = len(label_encoder.classes_)

    model = build_model(
        input_dim=X_train_processed.shape[1],
        num_classes=num_classes,
    )

    with mlflow.start_run(run_name="hidro_weather_risk_neural_classifier"):
        mlflow.log_param("model_type", "dense_neural_network_classifier")
        mlflow.log_param("task", "risk_level_classification")
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_param("num_rows", len(df))
        mlflow.log_param("num_stations", df["hidro_station"].nunique())
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_param("risk_labels", ",".join(label_encoder.classes_))

        print("Training neural risk classifier...")

        model.fit(
            X_train_processed,
            y_train_encoded,
            validation_split=0.2,
            epochs=30,
            batch_size=32,
            verbose=1,
        )

        probabilities = model.predict(X_test_processed)
        predictions_encoded = np.argmax(probabilities, axis=1)
        predictions = label_encoder.inverse_transform(predictions_encoded)

        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average="macro")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")
        print(classification_report(y_test, predictions))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        model_path = output_dir / "risk_classifier.keras"
        preprocessing_path = output_dir / "risk_classifier_preprocessing.pkl"

        model.save(model_path)

        with open(preprocessing_path, "wb") as f:
            pickle.dump(
                {
                    "imputer": imputer,
                    "scaler": scaler,
                    "label_encoder": label_encoder,
                    "feature_cols": feature_cols,
                },
                f,
            )

        print(f"Saved neural risk classifier to {model_path}")
        print(f"Saved preprocessing to {preprocessing_path}")

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(preprocessing_path),
                            artifact_path="preprocessing")

        mlflow.keras.log_model(
            model,
            artifact_path="keras_model",
            registered_model_name="hidro_risk_classifier",
        )

        metrics = {
            "model": "hidro_risk_classifier",
            "status": "trained",
            "model_type": "dense_neural_network_classifier",
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "num_rows": len(df),
            "num_stations": df["hidro_station"].nunique(),
            "test_size": len(test_df),
            "model_path": str(model_path),
            "preprocessing_path": str(preprocessing_path),
        }

        metrics_path = Path("reports/modeling/risk_classifier_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        mlflow.log_artifact(str(metrics_path), artifact_path="reports")

        print(f"Saved risk classifier metrics to {metrics_path}")


if __name__ == "__main__":
    main()
