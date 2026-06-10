import os
from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from pathlib import Path

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Hydro Risk Monitoring",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #A0AEC0;
        font-size: 17px;
        margin-bottom: 25px;
    }
    .info-card {
        padding: 18px;
        border-radius: 14px;
        background-color: #111827;
        border: 1px solid #2D3748;
        margin-bottom: 12px;
    }
    .small-muted {
        color: #A0AEC0;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🌊 Hydro Risk Monitoring</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Inteligentni sistem za spremljanje vodostaja, napovedovanje prihodnjega vodostaja in oceno hidrološkega tveganja.</div>',
    unsafe_allow_html=True,
)

tab_user, tab_admin = st.tabs(
    ["👤 Uporabniški pogled", "🛠 Administratorski pogled"])


def get_json(endpoint: str):
    response = requests.get(f"{API_URL}{endpoint}", timeout=20)
    response.raise_for_status()
    return response.json()


def risk_badge(risk_level: str):
    if risk_level == "HIGH":
        st.error("🔴 Visoko tveganje")
    elif risk_level == "MEDIUM":
        st.warning("🟠 Srednje tveganje")
    elif risk_level == "LOW":
        st.success("🟢 Nizko tveganje")
    else:
        st.info("⚪ Tveganje ni na voljo")


def risk_color(risk_level: str) -> str:
    return {
        "HIGH": "red",
        "MEDIUM": "orange",
        "LOW": "green",
    }.get(risk_level, "gray")


def load_metric_file(path: str):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


with tab_user:
    st.header("Pregled hidrološkega stanja")

    st.markdown(
        """
        Sistem na zemljevidu prikazuje izbrane merilne postaje. Barva označuje trenutno napovedano stopnjo tveganja:
        🟢 LOW, 🟠 MEDIUM, 🔴 HIGH, ⚪ ni napovedi.
        """
    )

    try:
        stations = get_json("/stations")["stations"]
    except Exception as e:
        st.error(f"Ni mogoče pridobiti seznama postaj: {e}")
        st.stop()

    try:
        map_data = get_json("/stations/map")["stations"]
    except Exception:
        map_data = []

    st.subheader("Zemljevid merilnih postaj")

    if map_data:
        map_df = pd.DataFrame(map_data)

        m = folium.Map(
            location=[46.15, 14.99],
            zoom_start=8,
            tiles="CartoDB positron",
            zoom_control=False,
            dragging=False,
            scrollWheelZoom=False,
            doubleClickZoom=False,
        )

        for _, row in map_df.iterrows():
            station_id = row["station"]

            try:
                prediction = get_json(f"/predict/full/{station_id}")
                water = prediction["water_level_prediction"]
                risk = prediction["risk_prediction"]

                risk_level = risk.get("risk_level", "N/A")
                predicted_water = water.get("predicted_water_level", "N/A")
            except Exception:
                risk_level = "N/A"
                predicted_water = "N/A"

            popup_text = f"""
            <b>{station_id}</b><br>
            Napovedan vodostaj: {predicted_water} cm<br>
            Tveganje: {risk_level}
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=9,
                popup=popup_text,
                color=risk_color(risk_level),
                fill=True,
                fill_color=risk_color(risk_level),
                fill_opacity=0.85,
            ).add_to(m)

        st_folium(m, width=1200, height=480)
    else:
        st.info("Zemljevid trenutno ni na voljo.")

    st.divider()

    st.subheader("Napoved za izbrano merilno postajo")

    station = st.selectbox(
        "Izberi merilno postajo",
        stations,
        index=stations.index("2110_Ptuj") if "2110_Ptuj" in stations else 0,
    )

    st.caption(
        "Model za napoved vodostaja uporablja zadnjih 24 meritev kot vhodno časovno okno in napove naslednjo vrednost vodostaja."
    )

    if st.button("Prikaži napoved", type="primary"):
        try:
            data = get_json(f"/predict/full/{station}")
        except Exception as e:
            st.error(f"Napaka pri napovedi: {e}")
            st.stop()

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
            risk_badge(risk.get("risk_level", "N/A"))

        st.subheader("Podrobnosti napovedi")

        details_df = pd.DataFrame(
            [
                ["Merilna postaja", station],
                ["Čas zadnje meritve", water["latest_timestamp"]],
                ["Model za vodostaj", water["model"]],
                ["Model za tveganje", risk.get("model", "N/A")],
                ["Vremenska postaja", risk.get("weather_station", "N/A")],
                ["Zanesljivost risk modela", risk.get("confidence", "N/A")],
            ],
            columns=["Lastnost", "Vrednost"],
        )

        st.dataframe(details_df, use_container_width=True, hide_index=True)

        try:
            history_data = get_json(f"/history/{station}")["records"]
        except Exception:
            history_data = []

        if history_data:
            history_df = pd.DataFrame(history_data)
            history_df["datum"] = pd.to_datetime(history_df["datum"])
            history_df = history_df.set_index("datum")

            st.subheader("Zgodovina vodostaja")
            st.line_chart(history_df["vodostaj"])
        else:
            st.warning("Zgodovine vodostaja ni bilo mogoče pridobiti.")


with tab_admin:
    st.header("Administratorski pogled")

    st.markdown(
        """
        Nadzorna plošča prikazuje stanje modelov, metrike modelov v produkciji,
        poročila validacije/testiranja podatkov ter povezave do MLflow sledenja eksperimentom.
        """
    )

    st.subheader("Status modelov")

    try:
        models = get_json("/models")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Model za vodostaj")
            st.write(models["water_level_model"]["name"])
            if models["water_level_model"]["loaded"]:
                st.success("Model je naložen")
            else:
                st.error("Model ni naložen")
            st.caption(models["water_level_model"]["type"])

        with col2:
            st.markdown("### Model za tveganje")
            st.write(models["risk_model"]["name"])
            if models["risk_model"]["loaded"]:
                st.success("Model je naložen")
            else:
                st.error("Model ni naložen")
            st.caption(models["risk_model"]["type"])

    except Exception as e:
        st.error(f"Status modelov ni na voljo: {e}")

    st.divider()

    st.subheader("Ovrednotenje modelov v produkciji")

    hidro_metrics = load_metric_file(
        "reports/modeling/hidro_global_model_metrics.csv")
    risk_metrics = load_metric_file(
        "reports/modeling/risk_classifier_metrics.csv")

    if hidro_metrics is not None and not hidro_metrics.empty:
        row = hidro_metrics.iloc[0]
        st.markdown("#### Napoved vodostaja")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test MAE", round(row["test_mae"], 4))
        c2.metric("Test RMSE", round(row["test_rmse"], 4))
        c3.metric("Postaje", int(row["num_stations"]))
        c4.metric("Vrstice", int(row["num_rows"]))

        with st.expander("Podrobne metrike modela za vodostaj"):
            st.dataframe(hidro_metrics, use_container_width=True)

    if risk_metrics is not None and not risk_metrics.empty:
        row = risk_metrics.iloc[0]
        st.markdown("#### Klasifikacija tveganja")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", round(row["accuracy"], 4))
        c2.metric("F1 macro", round(row["f1_macro"], 4))
        c3.metric("Postaje", int(row["num_stations"]))
        c4.metric("Vrstice", int(row["num_rows"]))

        with st.expander("Podrobne metrike modela za tveganje"):
            st.dataframe(risk_metrics, use_container_width=True)

    st.divider()

    st.subheader("MLflow eksperimenti in register modelov")

    st.info(
        "Eksperimenti, parametri, metrike, artefakti in verzije modelov so shranjeni v DagsHub MLflow."
    )

    mlflow_models = pd.DataFrame(
        [
            {
                "Model": "hidro_global_water_level_forecaster",
                "Naloga": "regresija vodostaja",
                "Tip": "LSTM",
                "Stanje": "Production candidate",
            },
            {
                "Model": "hidro_risk_classifier",
                "Naloga": "klasifikacija tveganja",
                "Tip": "Dense neural network",
                "Stanje": "Production candidate",
            },
        ]
    )

    st.dataframe(mlflow_models, use_container_width=True, hide_index=True)

    st.caption(
        "Migracija modelov med stanji se vodi v MLflow Model Registry na DagsHub."
    )

    st.divider()

    st.subheader("Poročila validacije podatkov - Great Expectations")

    ge_index = Path("gx/uncommitted/data_docs/local_site/index.html")

    if ge_index.exists():
        with open(ge_index, "r", encoding="utf-8") as f:
            components.html(f.read(), height=650, scrolling=True)
    else:
        st.warning("Great Expectations HTML poročilo ni najdeno.")
        st.code("gx/uncommitted/data_docs/local_site/index.html")

    st.divider()

    st.subheader("Poročila testiranja podatkov - Evidently")

    evidently_reports = sorted(Path("reports/data_testing").glob("*.html"))

    if evidently_reports:
        selected_report = st.selectbox(
            "Izberi Evidently poročilo",
            evidently_reports,
            format_func=lambda p: p.name,
        )

        with open(selected_report, "r", encoding="utf-8") as f:
            components.html(f.read(), height=700, scrolling=True)
    else:
        st.warning("Evidently poročila niso najdena.")
        st.code("reports/data_testing")

    st.divider()

    st.subheader("Uporabljene in preskočene postaje")

    mapping = load_metric_file(
        "reports/modeling/risk_classifier_used_station_mapping.csv")
    skipped = load_metric_file(
        "reports/modeling/risk_classifier_skipped_stations.csv")

    if mapping is not None:
        with st.expander("Uporabljene hidrološke in vremenske postaje"):
            st.dataframe(mapping, use_container_width=True)

    if skipped is not None:
        with st.expander("Preskočene postaje"):
            st.dataframe(skipped, use_container_width=True)
