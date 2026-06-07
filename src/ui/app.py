import requests
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Hydro Risk Monitoring",
    layout="wide",
)

st.title("🌊 Hydro Risk Monitoring")

tab_user, tab_admin = st.tabs(
    ["👤 Uporabniški pogled", "🛠 Administratorski pogled"]
)


def risk_badge(risk_level: str):
    if risk_level == "HIGH":
        st.error("🔴 Visoko tveganje")
    elif risk_level == "MEDIUM":
        st.warning("🟠 Srednje tveganje")
    elif risk_level == "LOW":
        st.success("🟢 Nizko tveganje")
    else:
        st.info("⚪ Tveganje ni na voljo")


with tab_user:
    st.header("Napoved hidrološkega stanja")

    stations_response = requests.get(f"{API_URL}/stations")

    if not stations_response.ok:
        st.error("Ni mogoče pridobiti seznama postaj.")
        st.stop()

    stations = stations_response.json()["stations"]

    station = st.selectbox(
        "Izberi merilno postajo",
        stations,
        index=stations.index("2110_Ptuj") if "2110_Ptuj" in stations else 0,
    )

    map_response = requests.get(f"{API_URL}/stations/map")

    if map_response.ok:
        map_data = map_response.json()["stations"]

        if map_data:
            map_df = pd.DataFrame(map_data)

        st.subheader("Zemljevid merilnih postaj")

        m = folium.Map(
            location=[46.15, 14.99],
            zoom_start=8,
            tiles="CartoDB dark_matter",
        )

        for _, row in map_df.iterrows():
            station_id = row["station"]

            try:
                prediction_response = requests.get(
                    f"{API_URL}/predict/full/{station_id}",
                    timeout=10,
                )

                if prediction_response.ok:
                    prediction = prediction_response.json()
                    water = prediction["water_level_prediction"]
                    risk = prediction["risk_prediction"]

                    risk_level = risk.get("risk_level", "N/A")
                    predicted_water = water.get("predicted_water_level", "N/A")
                else:
                    risk_level = "N/A"
                    predicted_water = "N/A"

            except requests.exceptions.RequestException:
                risk_level = "N/A"
                predicted_water = "N/A"

            if risk_level == "HIGH":
                color = "red"
            elif risk_level == "MEDIUM":
                color = "orange"
            elif risk_level == "LOW":
                color = "green"
            else:
                color = "gray"

            popup_text = f"""
            <b>{station_id}</b><br>
            Napovedan vodostaj: {predicted_water} cm<br>
            Tveganje: {risk_level}
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                popup=popup_text,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
            ).add_to(m)

        st_folium(m, width=1200, height=500)
    else:
        st.info("Zemljevid trenutno ni na voljo.")

    if st.button("Prikaži napoved", type="primary"):
        response = requests.get(f"{API_URL}/predict/full/{station}")

        if response.ok:
            data = response.json()

            water = data["water_level_prediction"]
            risk = data["risk_prediction"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Zadnji izmerjeni vodostaj",
                    f'{water["latest_water_level"]:.2f} cm',
                )

            with col2:
                st.metric(
                    "Napovedan vodostaj",
                    f'{water["predicted_water_level"]:.2f} cm',
                )

            with col3:
                risk_badge(risk["risk_level"])

            st.divider()

            st.subheader("Podrobnosti napovedi")

            details = {
                "Merilna postaja": station,
                "Čas zadnje meritve": water["latest_timestamp"],
                "Model za vodostaj": water["model"],
                "Model za tveganje": risk.get("model", "N/A"),
                "Vremenska postaja": risk.get("weather_station", "N/A"),
                "Zanesljivost risk modela": risk.get("confidence", "N/A"),
            }

            st.json(details)
            history_response = requests.get(f"{API_URL}/history/{station}")

            if history_response.ok:
                history_data = history_response.json()["records"]

                if history_data:
                    history_df = pd.DataFrame(history_data)
                    history_df["datum"] = pd.to_datetime(history_df["datum"])
                    history_df = history_df.set_index("datum")

                    st.subheader("Zgodovina vodostaja")
                    st.line_chart(history_df["vodostaj"])
            else:
                st.warning("Zgodovine vodostaja ni bilo mogoče pridobiti.")

        else:
            st.error(response.json().get("detail", "Napaka pri napovedi."))


with tab_admin:
    st.header("Administratorski pogled")

    models_response = requests.get(f"{API_URL}/models")

    if models_response.ok:
        st.subheader("Status modelov")
        st.json(models_response.json())

    st.subheader("Registrirani modeli")
    st.write("**hidro_global_water_level_forecaster** — globalni LSTM model")
    st.write("**hidro_risk_classifier** — nevronski klasifikator tveganja")

    st.subheader("Eksperimenti")
    st.write("Eksperimenti in verzije modelov so shranjeni v DagsHub MLflow.")

    st.subheader("Lokalna poročila")

    metric_files = [
        "reports/modeling/hidro_global_model_metrics.csv",
        "reports/modeling/risk_classifier_metrics.csv",
        "reports/modeling/risk_classifier_used_station_mapping.csv",
        "reports/modeling/risk_classifier_skipped_stations.csv",
    ]

    for file_path in metric_files:
        try:
            df = pd.read_csv(file_path)
            st.write(f"**{file_path}**")
            st.dataframe(df, use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Datoteka ne obstaja: {file_path}")
