import os
import re
import pandas as pd
import numpy as np
from lxml import etree as ET
import yaml


def preprocess_weather_data():
    params = yaml.safe_load(open("params.yaml"))
    weather_stations = params["stations"]["weather_stations"]

    # Pot do XML datoteke
    input_path = "data/raw/weather/weather_data.xml"

    # Naloži in razčleni XML
    with open(input_path, "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    output_dir = "data/preprocessed/weather"
    os.makedirs(output_dir, exist_ok=True)

    for meting in root.findall("metData"):
        # Ključne vrednosti
        station_name_raw = meting.findtext(
            "domain_shortTitle", default="neznano")

        # Preskoči, če postaja ni med dovoljenimi
        if station_name_raw not in weather_stations:
            continue

        station_name = re.sub(r"[^\wšđžčćŠĐŽČĆ]+", "_",
                              station_name_raw, flags=re.UNICODE)
        timestamp = meting.findtext("valid_UTC", default="")
        timestamp_local = meting.findtext("valid", default="")

        podatki = {
            "timestamp_utc": timestamp,
            "timestamp_local": timestamp_local,
            "temperature": meting.findtext("t", default=""),
            "dew_point": meting.findtext("td", default=""),
            "rel_humidity": meting.findtext("rh", default=""),
            "precipitation": meting.findtext("rr_val", default=""),
            "precipitation_1h": meting.findtext("tp_1h_acc", default=""),
            "precipitation_12h": meting.findtext("tp_12h_acc", default=""),
            "wind_speed": meting.findtext("ff_val", default=""),
            "wind_direction": meting.findtext("ddavg_val", default=""),
            "snow": meting.findtext("snow", default=""),
            "station": station_name_raw,
        }

        df_new = pd.DataFrame([podatki])

        # Ime datoteke glede na postajo
        filename = f"{station_name}.csv"
        filepath = os.path.join(output_dir, filename)

        # Če obstaja CSV, združi
        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath, encoding="utf-8")
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new

        if pd.isna(timestamp) or timestamp == "":
            print(
                f"⚠️ Opozorilo: v {filename} obstaja vrstica brez UTC časovnega žiga!")

        # Odstrani podvojene po UTC datumu
        df = df.drop_duplicates(subset=["timestamp_utc"])
        df = df.sort_values(by="timestamp_utc")

        # Zamenjaj prazne vrednosti z NaN
        df = df.replace("", np.nan)
        df = df.infer_objects(copy=False)
        # Shrani CSV
        df.to_csv(filepath, index=False, encoding="utf-8")

        print(f"[✓] Shranjeno: {filename} ({len(df)} skupnih vrstic)")

    print("✅ Vsi vremenski podatki so bili uspešno obdelani in shranjeni.")


if __name__ == "__main__":
    preprocess_weather_data()
