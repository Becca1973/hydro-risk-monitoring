from pathlib import Path
import pickle
import random
import os

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.model.preprocess import DatePreprocessor, SlidingWindowTransformer


def build_model(input_shape):
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


def train_station(csv_path: Path, params: dict):
    print(f"\nTraining model for {csv_path.name}")

    station_name = csv_path.stem

    df = pd.read_csv(csv_path)

    feature_cols = params["feature_cols"]
    target_col = params["target_col"]
    window_size = params["window_size"]
    test_size = params["test_size"]

    required_cols = ["datum"] + feature_cols

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Skipping {csv_path.name}, missing columns: {missing_cols}")
        return {
            "station": station_name,
            "status": "skipped",
            "reason": f"missing columns: {missing_cols}",
        }

    df = df[required_cols].copy()
    df = df.dropna(subset=[target_col])

    empty_feature_cols = [
        col for col in feature_cols
        if df[col].notna().sum() == 0
    ]

    if empty_feature_cols:
        print(
            f"Skipping {csv_path.name}, empty feature columns: {empty_feature_cols}"
        )
        return {
            "station": station_name,
            "status": "skipped",
            "reason": f"empty feature columns: {empty_feature_cols}",
        }

    if len(df) < params["min_rows"]:
        print(f"Skipping {csv_path.name}, not enough rows.")
        return {
            "station": station_name,
            "status": "skipped",
            "reason": "not enough rows",
        }

    preprocessing_pipeline = Pipeline(
        steps=[
            ("date_preprocessor", DatePreprocessor(date_col="datum")),
            (
                "column_transformer",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", MinMaxScaler()),
                                ]
                            ),
                            feature_cols + ["hour", "day",
                                            "month", "dayofweek"],
                        )
                    ]
                ),
            ),
        ]
    )

    processed_data = preprocessing_pipeline.fit_transform(df)

    processed_columns = feature_cols + ["hour", "day", "month", "dayofweek"]
    processed_df = pd.DataFrame(processed_data, columns=processed_columns)

    sliding_window = SlidingWindowTransformer(
        window_size=window_size,
        target_col=target_col,
        feature_cols=processed_columns,
    )

    X, y = sliding_window.transform(processed_df)

    if len(X) <= test_size:
        print(f"Skipping {csv_path.name}, not enough samples after windowing.")
        return {
            "station": station_name,
            "status": "skipped",
            "reason": "not enough samples after windowing",
        }

    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # MLflow tracking: vsak model/postaja ima svoj run
    with mlflow.start_run(run_name=f"hidro_lstm_{station_name}"):

        # MLflow tracking: logiranje parametrov
        mlflow.log_param("station", station_name)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("epochs", params["epochs"])
        mlflow.log_param("batch_size", params["batch_size"])
        mlflow.log_param("min_rows", params["min_rows"])
        mlflow.log_param("num_rows", len(df))
        mlflow.log_param("num_samples", len(X))

        model = build_model((X_train.shape[1], X_train.shape[2]))

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
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

        # Train final model on the full dataset, similarly to professor's example
        final_model = build_model((X.shape[1], X.shape[2]))

        final_model.fit(
            X,
            y,
            validation_split=0.2,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stopping],
            verbose=1,
        )

        full_predictions = final_model.predict(X).flatten()

        full_mae = mean_absolute_error(y, full_predictions)
        full_mse = mean_squared_error(y, full_predictions)
        full_rmse = np.sqrt(full_mse)

        print(f"Full dataset MAE: {full_mae:.4f}")
        print(f"Full dataset MSE: {full_mse:.4f}")
        print(f"Full dataset RMSE: {full_rmse:.4f}")

        # MLflow tracking: logiranje metrik
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("full_mae", full_mae)
        mlflow.log_metric("full_mse", full_mse)
        mlflow.log_metric("full_rmse", full_rmse)

        output_dir = Path(params["models_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"model_{station_name}.keras"
        pipeline_path = output_dir / f"pipeline_{station_name}.pkl"

        final_model.save(model_path)

        with open(pipeline_path, "wb") as f:
            pickle.dump(preprocessing_pipeline, f)

        print(f"Saved model to {model_path}")
        print(f"Saved pipeline to {pipeline_path}")

        # MLflow tracking: logiranje artefaktov
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(pipeline_path), artifact_path="pipeline")

        return {
            "station": station_name,
            "status": "trained",
            "reason": "",
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "full_mae": full_mae,
            "full_mse": full_mse,
            "full_rmse": full_rmse,
            "model_path": str(model_path),
            "pipeline_path": str(pipeline_path),
            "num_rows": len(df),
            "num_samples": len(X),
        }


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]

    random_state = train_params.get("random_state", 42)

    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # MLflow tracking: ime eksperimenta

    # Local station-by-station training is kept for development purposes.
    # It is disabled for remote MLflow tracking to avoid creating too many runs.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("hidro-water-level-forecasting-local")

    input_dir = Path(train_params["hidro_input_dir"])

    results = []

    for csv_path in sorted(input_dir.glob("*.csv")):
        result = train_station(csv_path, train_params)
        if result is not None:
            results.append(result)

    metrics_dir = Path("reports/modeling")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "hidro_model_metrics.csv"

    pd.DataFrame(results).to_csv(metrics_path, index=False)

    # MLflow tracking: shrani skupni CSV z metrikami
    mlflow.log_artifact(str(metrics_path), artifact_path="reports")

    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
