import os
import yaml


def extract_hydro_stations(hydro_dir):
    stations = set()
    for fname in os.listdir(hydro_dir):
        if fname.endswith(".csv"):
            parts = fname.split("_")
            if len(parts) == 2:
                station = parts[1].replace(".csv", "")
                stations.add(station)
    return sorted(stations)


def extract_weather_stations(weather_dir):
    stations = set()
    for fname in os.listdir(weather_dir):
        if fname.endswith(".csv"):
            station = fname.replace(".csv", "")
            stations.add(station)
    return sorted(stations)


def save_to_params(hydro_stations, weather_stations, out_path="params.yaml"):
    params = {
        "hydro_stations": hydro_stations,
        "weather_stations": weather_stations
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(params, f, allow_unicode=True)


if __name__ == "__main__":
    hydro_dir = "data/preprocessed/hidro"
    weather_dir = "data/preprocessed/weather"
    hydro_stations = extract_hydro_stations(hydro_dir)
    weather_stations = extract_weather_stations(weather_dir)
    save_to_params(hydro_stations, weather_stations)
    print("✅ Params datoteka uspešno ustvarjena.")
