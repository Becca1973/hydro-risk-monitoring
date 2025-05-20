import requests
from datetime import datetime
import os
import yaml


def fetch_weather_data():

    params = yaml.safe_load(open("params.yaml"))["fetch"]
    try:
        # URL za pridobivanje XML podatkov o vremenu
        url = params["weather_url"]

        # Pridobimo podatke
        response = requests.get(url)
        response.raise_for_status()

        # Pot do datoteke za shranjevanje
        file_path = "data/raw/weather/weather_data.xml"
        # Ustvari mape, če še ne obstajajo
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            file.write(response.content)

        # Izpis sporočila o uspehu
        print(
            f"Fetching successful. Data saved to {file_path} at {datetime.now()}")

    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")


if __name__ == "__main__":
    fetch_weather_data()
