from pathlib import Path
import shutil

import pandas as pd


TOLERANCE = pd.Timedelta("90min")
MAX_MATCHED_ROWS_PER_STATION = 500

HIDRO_SRC = Path("data/preprocessed/hidro")
WEATHER_SRC = Path("data/preprocessed/weather")

HIDRO_DST = Path("data/production/hidro")
WEATHER_DST = Path("data/production/weather")

REPORTS_DST = Path("reports/modeling")
MAPPING_PATH = REPORTS_DST / "risk_classifier_used_station_mapping.csv"


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


def parse_hidro_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["datum"], errors="coerce")
    return df.dropna(subset=["_dt"]).sort_values("_dt")


def parse_weather_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_dt"] = (
        df["timestamp_local"]
        .astype(str)
        .str.replace(" CEST", "", regex=False)
        .str.replace(" CET", "", regex=False)
    )
    df["_dt"] = pd.to_datetime(
        df["_dt"],
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )
    return df.dropna(subset=["_dt"]).sort_values("_dt")


def find_weather_file(weather_station: str) -> Path | None:
    weather_norm = normalize_name(weather_station)

    for path in WEATHER_SRC.glob("*.csv"):
        if normalize_name(path.stem) == weather_norm:
            return path

    return None


def clean_output_dirs():
    for path in [HIDRO_DST, WEATHER_DST]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def main():
    clean_output_dirs()

    mapping = pd.read_csv(MAPPING_PATH)

    diagnostics = []

    for _, row in mapping.iterrows():
        hidro_station = row["hidro_station"]
        weather_station = row["matched_weather_station"]

        hidro_path = HIDRO_SRC / f"{hidro_station}.csv"
        weather_path = find_weather_file(weather_station)

        if not hidro_path.exists():
            diagnostics.append({
                "hidro_station": hidro_station,
                "weather_station": weather_station,
                "status": "skipped",
                "reason": "hidro file missing",
            })
            continue

        if weather_path is None or not weather_path.exists():
            diagnostics.append({
                "hidro_station": hidro_station,
                "weather_station": weather_station,
                "status": "skipped",
                "reason": "weather file missing",
            })
            continue

        hidro_df = parse_hidro_datetime(pd.read_csv(hidro_path))
        weather_df = parse_weather_datetime(pd.read_csv(weather_path))

        if hidro_df.empty or weather_df.empty:
            diagnostics.append({
                "hidro_station": hidro_station,
                "weather_station": weather_station,
                "status": "skipped",
                "reason": "empty parsed data",
            })
            continue

        # Najdi hidro vrstice, ki imajo vremensko meritev znotraj 90 minut.
        probe = pd.merge_asof(
            hidro_df[["_dt"]].sort_values("_dt"),
            weather_df[["_dt"]].sort_values("_dt").rename(
                columns={"_dt": "_weather_dt"}),
            left_on="_dt",
            right_on="_weather_dt",
            direction="nearest",
            tolerance=TOLERANCE,
        )

        matched_hidro_times = (
            probe.dropna(subset=["_weather_dt"])["_dt"]
            .sort_values()
            .tail(MAX_MATCHED_ROWS_PER_STATION)
        )

        if matched_hidro_times.empty:
            nearest_diff_min = None
            try:
                latest_hidro = hidro_df["_dt"].max()
                weather_df["_diff_min"] = (
                    weather_df["_dt"] - latest_hidro
                ).abs().dt.total_seconds() / 60
                nearest_diff_min = float(weather_df["_diff_min"].min())
            except Exception:
                pass

            diagnostics.append({
                "hidro_station": hidro_station,
                "weather_station": weather_station,
                "status": "skipped",
                "reason": "no matched timestamps within tolerance",
                "nearest_diff_min": nearest_diff_min,
                "hidro_min": hidro_df["_dt"].min(),
                "hidro_max": hidro_df["_dt"].max(),
                "weather_min": weather_df["_dt"].min(),
                "weather_max": weather_df["_dt"].max(),
            })
            continue

        start_time = matched_hidro_times.min()
        end_time = matched_hidro_times.max()

        hidro_prod = hidro_df[
            (hidro_df["_dt"] >= start_time) &
            (hidro_df["_dt"] <= end_time)
        ].drop(columns=["_dt"])

        weather_prod = weather_df[
            (weather_df["_dt"] >= start_time - TOLERANCE) &
            (weather_df["_dt"] <= end_time + TOLERANCE)
        ].drop(columns=["_dt"], errors="ignore")

        hidro_prod.to_csv(HIDRO_DST / hidro_path.name, index=False)
        weather_prod.to_csv(WEATHER_DST / weather_path.name, index=False)

        diagnostics.append({
            "hidro_station": hidro_station,
            "weather_station": weather_station,
            "status": "ok",
            "hidro_rows": len(hidro_prod),
            "weather_rows": len(weather_prod),
            "start_time": start_time,
            "end_time": end_time,
        })

    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_path = REPORTS_DST / "production_data_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_path, index=False)

    print("Production data created.")
    print("Diagnostics:")
    print(diagnostics_df["status"].value_counts(dropna=False))
    print(f"Saved diagnostics to {diagnostics_path}")


if __name__ == "__main__":
    main()
