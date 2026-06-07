import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Hydro Risk Monitoring",
    layout="wide"
)

st.title("Hydro Risk Monitoring")

tab_user, tab_admin = st.tabs(
    ["Uporabniški pogled", "Administratorski pogled"])

with tab_user:
    st.header("Napoved hidrološkega tveganja")

    station = st.selectbox(
        "Izberi merilno postajo",
        ["2110_Ptuj", "6720_Celje", "3660_Litija"]
    )

    if st.button("Prikaži napoved"):
        water_response = requests.get(f"{API_URL}/predict/water-level-demo")
        risk_response = requests.get(f"{API_URL}/predict/risk-demo")

        if water_response.ok and risk_response.ok:
            water_data = water_response.json()
            risk_data = risk_response.json()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Napovedan vodostaj",
                    water_data["predicted_water_level"]
                )

            with col2:
                st.metric(
                    "Stopnja tveganja",
                    risk_data["risk_level"]
                )

        else:
            st.error("Napaka pri pridobivanju napovedi.")

with tab_admin:
    st.header("Administratorski pogled")

    models_response = requests.get(f"{API_URL}/models")

    if models_response.ok:
        st.json(models_response.json())

    st.subheader("Modeli")
    st.write("Globalni LSTM model za napoved vodostaja")
    st.write("Nevronski klasifikator za LOW / MEDIUM / HIGH tveganje")

    st.subheader("Sledenje eksperimentom")
    st.write("Eksperimenti in registrirani modeli so shranjeni v DagsHub MLflow.")
