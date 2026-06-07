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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def build_model(input_shape):
    # Build LSTM model for global water level forecasting.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


def load_all_hidro_data(input_dir: Path, params: dict) -> pd.DataFrame:
    # Load all hidro CSV files and merge them into one dataset.
    feature_cols = params["feature_cols"]
    target_col = params["target_col"]
    required_cols = ["datum"] + feature_cols

    all_frames = []
    skipped = []

    for csv_path in sorted(input_dir.glob("*.csv")):
        station_name = csv_path.stem
        df = pd.read_csv(csv_path)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            skipped.append(
                {
                    "station": station_name,
                    "reason": f"missing columns: {missing_cols}",
                }
            )
            continue

        df = df[required_cols].copy()
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
        df = df.dropna(subset=["datum", target_col])
        df = df.sort_values("datum")

        empty_feature_cols = [
            col for col in feature_cols
            if df[col].notna().sum() == 0
        ]

        if empty_feature_cols:
            skipped.append(
                {
                    "station": station_name,
                    "reason": f"empty feature columns: {empty_feature_cols}",
                }
            )
            continue

        if len(df) < params["min_rows"]:
            skipped.append(
                {
                    "station": station_name,
                    "reason": "not enough rows",
                }
            )
            continue

        # Extract numeric station code from filename, e.g. 2110_Ptuj -> 2110.
        station_code = station_name.split("_")[0]

        df["station_name"] = station_name
        df["station_code"] = pd.to_numeric(station_code, errors="coerce")

        # Add time-based features from timestamp.
        df["hour"] = df["datum"].dt.hour
        df["day"] = df["datum"].dt.day
        df["month"] = df["datum"].dt.month
        df["dayofweek"] = df["datum"].dt.dayofweek

        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        all_frames.append(df)

    Path("reports/modeling").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(skipped).to_csv(
        "reports/modeling/hidro_global_skipped_stations.csv",
        index=False,
    )

    if not all_frames:
        raise ValueError("No valid hidro datasets found for global training.")

    return pd.concat(all_frames, ignore_index=True)


def create_windows_by_station(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
    test_size: int,
):
    # Create sliding windows separately for each station.
    # This prevents mixing the last row of one station with the first row of another.
    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []
    X_full_all = []
    y_full_all = []

    target_idx = feature_cols.index(target_col)

    for station_name, group in df.groupby("station_name"):
        group = group.sort_values("datum")
        data = group[feature_cols].to_numpy(dtype=float)

        X_station = []
        y_station = []

        for i in range(window_size, len(data)):
            X_station.append(data[i - window_size:i])
            y_station.append(data[i, target_idx])

        X_station = np.array(X_station)
        y_station = np.array(y_station)

        if len(X_station) <= test_size:
            print(
                f"Skipping {station_name} in global model, "
                "not enough samples after windowing."
            )
            continue

        X_full_all.append(X_station)
        y_full_all.append(y_station)

        X_train_all.append(X_station[:-test_size])
        y_train_all.append(y_station[:-test_size])

        X_test_all.append(X_station[-test_size:])
        y_test_all.append(y_station[-test_size:])

    if not X_train_all:
        raise ValueError("No stations have enough samples for global model.")

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)

    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)

    X_full = np.concatenate(X_full_all, axis=0)
    y_full = np.concatenate(y_full_all, axis=0)

    return X_train, X_test, y_train, y_test, X_full, y_full


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]

    random_state = train_params.get("random_state", 42)

    # Set random seeds for more reproducible training.
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    input_dir = Path(train_params["hidro_input_dir"])

    output_dir = Path(
        train_params.get("global_models_output_dir", "models/hidro_global")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a separate MLflow experiment for the global model.
    mlflow.set_experiment("hidro-global-water-level-forecasting")

    feature_cols = train_params["feature_cols"] + [
        "station_code",
        "hour",
        "day",
        "month",
        "dayofweek",
    ]

    target_col = train_params["target_col"]

    print("Loading all hidro datasets...")
    df = load_all_hidro_data(input_dir, train_params)

    print(f"Loaded rows: {len(df)}")
    print(f"Stations used: {df['station_name'].nunique()}")

    # Impute missing values and normalize all numeric features.
    preprocessing_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    processed_values = preprocessing_pipeline.fit_transform(df[feature_cols])

    processed_df = pd.DataFrame(processed_values, columns=feature_cols)
    processed_df["station_name"] = df["station_name"].values
    processed_df["datum"] = df["datum"].values

    # Convert time series into supervised learning samples.
    X_train, X_test, y_train, y_test, X_full, y_full = create_windows_by_station(
        df=processed_df,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=train_params["window_size"],
        test_size=train_params["test_size"],
    )

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # Start one MLflow run for the global model.
    with mlflow.start_run(run_name="hidro_lstm_global"):
        mlflow.log_param("model_type", "global_lstm")
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_param("window_size", train_params["window_size"])
        mlflow.log_param("test_size", train_params["test_size"])
        mlflow.log_param("epochs", train_params["epochs"])
        mlflow.log_param("batch_size", train_params["batch_size"])
        mlflow.log_param("min_rows", train_params["min_rows"])
        mlflow.log_param("num_rows", len(df))
        mlflow.log_param("num_stations", df["station_name"].nunique())
        mlflow.log_param("num_train_samples", len(X_train))
        mlflow.log_param("num_test_samples", len(X_test))

        # Train evaluation model on train split.
        model = build_model((X_train.shape[1], X_train.shape[2]))

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=train_params["epochs"],
            batch_size=train_params["batch_size"],
            callbacks=[early_stopping],
            verbose=1,
        )

        predictions = model.predict(X_test).flatten()

        test_mae = mean_absolute_error(y_test, predictions)
        test_mse = mean_squared_error(y_test, predictions)
        test_rmse = np.sqrt(test_mse)

        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")

        # Train final production candidate on the full dataset.
        final_model = build_model((X_full.shape[1], X_full.shape[2]))

        final_model.fit(
            X_full,
            y_full,
            validation_split=0.2,
            epochs=train_params["epochs"],
            batch_size=train_params["batch_size"],
            callbacks=[early_stopping],
            verbose=1,
        )

        full_predictions = final_model.predict(X_full).flatten()

        full_mae = mean_absolute_error(y_full, full_predictions)
        full_mse = mean_squared_error(y_full, full_predictions)
        full_rmse = np.sqrt(full_mse)

        print(f"Full dataset MAE: {full_mae:.4f}")
        print(f"Full dataset MSE: {full_mse:.4f}")
        print(f"Full dataset RMSE: {full_rmse:.4f}")

        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("full_mae", full_mae)
        mlflow.log_metric("full_mse", full_mse)
        mlflow.log_metric("full_rmse", full_rmse)

        model_path = output_dir / "model_hidro_global.keras"
        pipeline_path = output_dir / "pipeline_hidro_global.pkl"

        # Save model and preprocessing pipeline locally.
        final_model.save(model_path)

        with open(pipeline_path, "wb") as f:
            pickle.dump(preprocessing_pipeline, f)

        print(f"Saved global model to {model_path}")
        print(f"Saved global pipeline to {pipeline_path}")

        # Log saved files as MLflow artifacts.
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(pipeline_path), artifact_path="pipeline")

        # Programmatic MLflow Model Registry registration.
        mlflow.keras.log_model(
            final_model,
            artifact_path="keras_model",
            registered_model_name="hidro_global_water_level_forecaster",
        )

        metrics = {
            "model": "hidro_global_water_level_forecaster",
            "status": "trained",
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "full_mae": full_mae,
            "full_mse": full_mse,
            "full_rmse": full_rmse,
            "model_path": str(model_path),
            "pipeline_path": str(pipeline_path),
            "num_rows": len(df),
            "num_stations": df["station_name"].nunique(),
            "num_train_samples": len(X_train),
            "num_test_samples": len(X_test),
        }

        metrics_dir = Path("reports/modeling")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = metrics_dir / "hidro_global_model_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        # Log metrics CSV as report artifact.
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")

        print(f"Saved global metrics to {metrics_path}")


if __name__ == "__main__":
    main()
