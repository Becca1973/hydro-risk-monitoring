import os
from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium


# CONFIG

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Hydro Risk Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# STYLING

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #020617 0%, #071426 55%, #020617 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #071426 100%);
        border-right: 1px solid rgba(148,163,184,.18);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 5rem;
        padding-left: 2.3rem;
        padding-right: 2.3rem;
        max-width: 1600px;
    }

    h1, h2, h3 {
        color: #f8fafc;
        letter-spacing: -0.03em;
    }

    p, label, span {
        color: #cbd5e1;
    }

    .app-title {
        font-size: 58px;
        font-weight: 950;
        color: #f8fafc;
        line-height: 1.05;
        margin-bottom: 10px;
    }

    .app-subtitle {
        font-size: 18px;
        color: #cbd5e1;
        max-width: 930px;
        line-height: 1.7;
        margin-bottom: 26px;
    }

    .top-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(14,165,233,.13);
        border: 1px solid rgba(14,165,233,.35);
        color: #67e8f9;
        font-weight: 900;
        font-size: 12px;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 18px;
    }

    .section-card {
        background: rgba(15,23,42,.76);
        border: 1px solid rgba(148,163,184,.20);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 14px 34px rgba(0,0,0,.24);
    }

    .info-card {
        background: rgba(15,23,42,.82);
        border: 1px solid rgba(148,163,184,.20);
        border-radius: 22px;
        padding: 22px;
        margin-top: 16px;
        margin-bottom: 18px;
    }

    .info-title {
        color: #f8fafc;
        font-weight: 900;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .muted {
        color: #9fb2ca;
        font-size: 14px;
        line-height: 1.65;
    }

    .metric-card {
        background: rgba(15,23,42,.92);
        border: 1px solid rgba(148,163,184,.24);
        border-radius: 24px;
        padding: 25px 25px 22px 25px;
        min-height: 150px;
        box-shadow: 0 15px 36px rgba(0,0,0,.22);
    }

    .metric-label {
        color: #67e8f9;
        font-size: 12px;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .07em;
        margin-bottom: 16px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 34px;
        font-weight: 950;
        margin-bottom: 8px;
        letter-spacing: -0.04em;
    }

    .metric-caption {
        color: #9fb2ca;
        font-size: 13px;
        line-height: 1.45;
    }

    .risk-low {
        background: linear-gradient(135deg, rgba(22,101,52,.95), rgba(20,83,45,.72));
        border: 1px solid rgba(74,222,128,.38);
    }

    .risk-medium {
        background: linear-gradient(135deg, rgba(154,74,12,.96), rgba(120,53,15,.72));
        border: 1px solid rgba(251,191,36,.42);
    }

    .risk-high {
        background: linear-gradient(135deg, rgba(153,27,27,.96), rgba(127,29,29,.72));
        border: 1px solid rgba(248,113,113,.45);
    }

    .risk-na {
        background: linear-gradient(135deg, rgba(51,65,85,.95), rgba(30,41,59,.76));
        border: 1px solid rgba(148,163,184,.35);
    }

    .risk-value {
        color: #f8fafc;
        font-size: 30px;
        font-weight: 950;
        margin-bottom: 10px;
        letter-spacing: -0.03em;
    }

    .status-ok {
        background: rgba(22,101,52,.78);
        border: 1px solid rgba(74,222,128,.35);
        color: #bbf7d0;
        border-radius: 16px;
        padding: 14px 16px;
        font-weight: 900;
        margin-bottom: 12px;
    }

    .status-bad {
        background: rgba(127,29,29,.78);
        border: 1px solid rgba(248,113,113,.35);
        color: #fecaca;
        border-radius: 16px;
        padding: 14px 16px;
        font-weight: 900;
        margin-bottom: 12px;
    }

    .sidebar-card {
        background: rgba(15,23,42,.78);
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 18px;
        padding: 16px;
        margin: 14px 0;
    }

    .sidebar-status-ok {
        background: rgba(22,101,52,.68);
        border: 1px solid rgba(74,222,128,.35);
        border-radius: 16px;
        padding: 14px 15px;
        color: #bbf7d0;
        font-weight: 900;
        margin-top: 12px;
    }

    .sidebar-status-bad {
        background: rgba(127,29,29,.70);
        border: 1px solid rgba(248,113,113,.35);
        border-radius: 16px;
        padding: 14px 15px;
        color: #fecaca;
        font-weight: 900;
        margin-top: 12px;
    }

    .legend-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 8px 0;
        font-size: 14px;
        color: #cbd5e1;
    }

    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 999px;
        display: inline-block;
        border: 2px solid rgba(255,255,255,.52);
    }

    .dot-low { background: #16a34a; }
    .dot-medium { background: #f97316; }
    .dot-high { background: #ef4444; }
    .dot-na { background: #94a3b8; }

    .prediction-warning {
        background: rgba(146,64,14,.23);
        border: 1px solid rgba(251,191,36,.34);
        color: #fde68a;
        border-radius: 18px;
        padding: 18px 20px;
        line-height: 1.6;
        font-weight: 750;
    }

    .map-frame {
        border-radius: 24px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,.24);
        box-shadow: 0 22px 50px rgba(0,0,0,.27);
        margin-top: 20px;
        margin-bottom: 34px;
    }

    .details-spacer {
        height: 18px;
    }

    .stButton > button {
        border-radius: 16px;
        padding: .9rem 1.2rem;
        font-weight: 900;
        border: 0;
        color: white;
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        box-shadow: 0 14px 30px rgba(37,99,235,.28);
    }

    .stButton > button:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(15,23,42,.96);
        border: 1px solid rgba(148,163,184,.25);
        border-radius: 15px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,.82);
        border: 1px solid rgba(148,163,184,.20);
        border-radius: 18px;
        padding: 18px;
    }

    .streamlit-expanderHeader {
        font-weight: 900;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# API HELPERS

@st.cache_data(ttl=300)
def get_json_cached(endpoint: str):
    response = requests.get(f"{API_URL}{endpoint}", timeout=25)
    response.raise_for_status()
    return response.json()


def get_json(endpoint: str):
    response = requests.get(f"{API_URL}{endpoint}", timeout=25)
    response.raise_for_status()
    return response.json()


def safe_get_json(endpoint: str):
    try:
        return get_json(endpoint)
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_prediction_cached(station: str):
    try:
        return get_json(f"/predict/full/{station}")
    except Exception:
        return None


def load_metric_file(path: str):
    try:
        file_path = Path(path)
        if file_path.exists():
            return pd.read_csv(file_path)
    except Exception:
        return None
    return None


# UI HELPERS

def risk_color(risk_level: str) -> str:
    return {
        "HIGH": "#ef4444",
        "MEDIUM": "#f97316",
        "LOW": "#16a34a",
    }.get(str(risk_level).upper(), "#94a3b8")


def risk_class(risk_level: str) -> str:
    return {
        "HIGH": "risk-high",
        "MEDIUM": "risk-medium",
        "LOW": "risk-low",
    }.get(str(risk_level).upper(), "risk-na")


def risk_label_sl(risk_level: str) -> str:
    return {
        "HIGH": "Visoko tveganje",
        "MEDIUM": "Srednje tveganje",
        "LOW": "Nizko tveganje",
        "N/A": "Napoved trenutno ni na voljo",
        None: "Napoved trenutno ni na voljo",
    }.get(str(risk_level).upper(), "Napoved trenutno ni na voljo")


def fmt(value, decimals=2):
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "Ni podatka"


def fmt_conf(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "Ni podatka"


def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_card(risk_level, confidence=None):
    st.markdown(
        f"""
        <div class="metric-card {risk_class(risk_level)}">
            <div class="metric-label">Ocena hidrološkega tveganja</div>
            <div class="risk-value">{risk_label_sl(risk_level)}</div>
            <div class="metric-caption">Zanesljivost modela: {fmt_conf(confidence)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def unavailable_prediction_box(station: str):
    st.markdown(
        f"""
        <div class="prediction-warning">
            ⚠️ <b>Napoved trenutno ni na voljo za postajo {station}.</b><br>
            To ne pomeni napake v aplikaciji. Za izbrano postajo trenutno ni dovolj vhodnih podatkov,
            ustreznega povezovanja z vremensko postajo ali pa model za to postajo še nima stabilne produkcijske napovedi.
            Ob naslednjem zajemu podatkov in ponovnem učenju modela se lahko postaja ponovno vključi v napovedovanje.
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_popup_html(station_id, risk_level, latest_water, predicted_water, timestamp):
    risk_text = risk_label_sl(risk_level)
    latest = f"{fmt(latest_water)} cm" if latest_water not in [
        None, "N/A"] else "Ni podatka"
    predicted = f"{fmt(predicted_water)} cm" if predicted_water not in [
        None, "N/A"] else "Ni podatka"
    time_text = timestamp if timestamp not in [None, "N/A"] else "Ni podatka"

    return f"""
    <div style="font-family: Arial, sans-serif; min-width: 230px; line-height: 1.45;">
        <div style="font-size: 16px; font-weight: 800; margin-bottom: 8px;">🌊 {station_id}</div>
        <div><b>Zadnja meritev:</b> {latest}</div>
        <div><b>Napoved vodostaja:</b> {predicted}</div>
        <div><b>Tveganje:</b> <span style="font-weight:800;">{risk_text}</span></div>
        <div style="margin-top: 6px; color: #64748b;"><b>Čas:</b> {time_text}</div>
    </div>
    """


# SESSION STATE

if "selected_prediction" not in st.session_state:
    st.session_state.selected_prediction = None

if "selected_station" not in st.session_state:
    st.session_state.selected_station = None


# SIDEBAR

with st.sidebar:
    st.markdown("## 🌊 Hydro Risk")
    st.markdown(
        """
        <div class="muted">
        Inteligentni sistem za spremljanje vodostaja, napovedovanje prihodnjih sprememb
        in oceno hidrološkega tveganja.
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_ok = safe_get_json("/health") is not None

    if api_ok:
        st.markdown(
            '<div class="sidebar-status-ok">🟢 Server je povezan</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="sidebar-status-bad">🔴 Server ni povezan</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-card">
            <b>Funkcionalnosti</b>
            <div class="muted" style="margin-top: 8px;">
                • napoved vodostaja<br>
                • ocena hidrološkega tveganja<br>
                • zemljevid merilnih postaj<br>
                • zgodovina meritev<br>
                • administratorski nadzor
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <b>Legenda tveganja</b>
            <div class="legend-row"><span class="legend-dot dot-low"></span> Nizko tveganje</div>
            <div class="legend-row"><span class="legend-dot dot-medium"></span> Srednje tveganje</div>
            <div class="legend-row"><span class="legend-dot dot-high"></span> Visoko tveganje</div>
            <div class="legend-row"><span class="legend-dot dot-na"></span> Napoved ni na voljo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <b>Produkcijski pogled</b>
            <div class="muted" style="margin-top: 8px;">
            Če za postajo ni napovedi, sistem prikaže uporabniku razumljivo obvestilo
            in se ne prekine z napako.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# HEADER

st.markdown('<div class="top-badge">MLOps • Napovedovanje • Monitoring</div>',
            unsafe_allow_html=True)
st.markdown('<div class="app-title">🌊 Hydro Risk Monitoring</div>',
            unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-subtitle">
    Spletni inteligentni sistem za spremljanje merilnih postaj, napovedovanje prihodnjega vodostaja
    in oceno hidrološkega tveganja na podlagi hidroloških ter vremenskih podatkov.
    </div>
    """,
    unsafe_allow_html=True,
)


tab_user, tab_admin = st.tabs(
    ["👤 Uporabniški pogled", "🛠 Administratorski pogled"])


# USER VIEW

with tab_user:
    st.markdown("## Pregled hidrološkega stanja")
    st.markdown(
        """
        <p class="muted">
        Zemljevid prikazuje izbrane reprezentativne merilne postaje po Sloveniji. Barva označuje napovedano
        stopnjo tveganja, siva barva pa pomeni, da napoved za postajo trenutno ni na voljo.
        </p>
        """,
        unsafe_allow_html=True,
    )

    try:
        all_stations = get_json_cached("/stations")["stations"]
    except Exception as e:
        st.error(
            f"Seznama postaj ni mogoče pridobiti. Preveri, ali je backend server zagnan. Podrobnosti: {e}")
        st.stop()

    map_data = []
    try:
        map_data = get_json_cached("/stations/map")["stations"]
    except Exception:
        st.info("Koordinate postaj trenutno niso na voljo. Zemljevid zato ni prikazan.")

    if map_data:
        map_df = pd.DataFrame(map_data)

        m = folium.Map(
            location=[46.15, 14.90],
            zoom_start=8,
            tiles="CartoDB positron",
            zoom_control=True,
            scrollWheelZoom=False,
            min_zoom=7,
            max_zoom=11,
        )

        # Omejitev pogleda na Slovenijo in bližnjo okolico.
        slovenia_bounds = [[45.35, 13.35], [46.95, 16.65]]
        m.fit_bounds(slovenia_bounds)

        for _, row in map_df.iterrows():
            station_id = row["station"]
            prediction = get_prediction_cached(station_id)

            if prediction:
                water = prediction.get("water_level_prediction", {})
                risk = prediction.get("risk_prediction", {})

                risk_level = risk.get("risk_level", "N/A")
                predicted_water = water.get("predicted_water_level", "N/A")
                latest_water = water.get("latest_water_level", "N/A")
                timestamp = water.get("latest_timestamp", "N/A")
            else:
                risk_level = "N/A"
                predicted_water = "N/A"
                latest_water = "N/A"
                timestamp = "N/A"

            marker_color = risk_color(risk_level)
            tooltip_text = f"{station_id} | {risk_label_sl(risk_level)}"

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=10,
                tooltip=tooltip_text,
                popup=folium.Popup(
                    create_popup_html(
                        station_id=station_id,
                        risk_level=risk_level,
                        latest_water=latest_water,
                        predicted_water=predicted_water,
                        timestamp=timestamp,
                    ),
                    max_width=310,
                ),
                color="#0284c7",
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.9,
                weight=3,
            ).add_to(m)

        st.markdown('<div class="map-frame">', unsafe_allow_html=True)
        st_folium(m, width=None, height=470, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## Napoved za izbrano merilno postajo")

    form_col, result_col = st.columns([1.05, 2.25], gap="large")

    with form_col:
        default_station = "2110_Ptuj" if "2110_Ptuj" in all_stations else all_stations[0]

        station = st.selectbox(
            "Izberi merilno postajo",
            all_stations,
            index=all_stations.index(default_station),
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Kako deluje sistem?</div>
                <div class="muted">
                Regresijski nevronski model uporablja zadnjih 24 hidroloških meritev
                za napoved prihodnjega vodostaja. Klasifikacijski model združi hidrološke
                in vremenske podatke ter vrne razred LOW, MEDIUM ali HIGH.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        show_prediction = st.button(
            "Prikaži napoved", type="primary", use_container_width=True)

    if show_prediction:
        st.session_state.selected_prediction = safe_get_json(
            f"/predict/full/{station}")
        st.session_state.selected_station = station

    prediction_to_show = st.session_state.selected_prediction
    selected_station = st.session_state.selected_station

    with result_col:
        if prediction_to_show is None:
            if selected_station:
                unavailable_prediction_box(selected_station)
            else:
                st.markdown(
                    """
                    <div class="info-card">
                        <div class="info-title">Izberi merilno postajo in zaženi napoved.</div>
                        <div class="muted">
                        Tukaj se prikažejo zadnji izmerjeni vodostaj, napoved vodostaja,
                        ocena tveganja in podrobnosti uporabljenih modelov.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            water = prediction_to_show.get("water_level_prediction", {})
            risk = prediction_to_show.get("risk_prediction", {})

            latest = water.get("latest_water_level")
            predicted = water.get("predicted_water_level")

            try:
                delta = float(predicted) - float(latest)
                delta_text = f"Sprememba glede na zadnjo meritev: {delta:+.2f} cm"
            except Exception:
                delta_text = "Sprememba glede na zadnjo meritev ni na voljo"

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card(
                    "Zadnji izmerjeni vodostaj",
                    f"{fmt(latest)} cm",
                    water.get("latest_timestamp", ""),
                )

            with c2:
                metric_card(
                    "Napovedan vodostaj",
                    f"{fmt(predicted)} cm",
                    delta_text,
                )

            with c3:
                risk_card(
                    risk.get("risk_level", "N/A"),
                    risk.get("confidence", None),
                )

            st.markdown('<div class="details-spacer"></div>',
                        unsafe_allow_html=True)

            with st.expander("Podrobnosti napovedi", expanded=True):
                details_df = pd.DataFrame(
                    [
                        ["Merilna postaja", selected_station],
                        ["Čas zadnje meritve", water.get(
                            "latest_timestamp", "Ni podatka")],
                        ["Model za vodostaj", water.get(
                            "model", "Ni podatka")],
                        ["Model za tveganje", risk.get("model", "Ni podatka")],
                        ["Vremenska postaja", risk.get(
                            "weather_station", "Ni podatka")],
                        ["Zanesljivost modela tveganja",
                            fmt_conf(risk.get("confidence"))],
                    ],
                    columns=["Lastnost", "Vrednost"],
                )
                st.dataframe(details_df, use_container_width=True,
                             hide_index=True)

    if prediction_to_show is not None and selected_station:
        history_response = safe_get_json(f"/history/{selected_station}")
        history_data = history_response["records"] if history_response else []

        if history_data:
            history_df = pd.DataFrame(history_data)
            history_df["datum"] = pd.to_datetime(
                history_df["datum"], errors="coerce")
            history_df = history_df.dropna(subset=["datum"])
            history_df = history_df.set_index("datum")

            st.markdown("### Zgodovina vodostaja")
            st.line_chart(history_df["vodostaj"], use_container_width=True)
        else:
            st.warning("Zgodovine vodostaja trenutno ni mogoče pridobiti.")


# ADMIN VIEW

with tab_admin:
    st.markdown("## Administratorski pogled")
    st.markdown(
        """
        <p class="muted">
        Administratorski pogled združuje status produkcijskih modelov, metrike učenja,
        poročila validacije podatkov in pregled uporabljenih oziroma preskočenih postaj.
        </p>
        """,
        unsafe_allow_html=True,
    )

    try:
        models = get_json("/models")
        col1, col2 = st.columns(2)

        with col1:
            model = models["water_level_model"]
            st.markdown("### Model za vodostaj")
            st.write(f"**{model['name']}**")
            if model["loaded"]:
                st.markdown(
                    '<div class="status-ok">🟢 Model je naložen v produkciji</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="status-bad">🔴 Model ni naložen</div>', unsafe_allow_html=True)
            st.caption(model["type"])

        with col2:
            model = models["risk_model"]
            st.markdown("### Model za tveganje")
            st.write(f"**{model['name']}**")
            if model["loaded"]:
                st.markdown(
                    '<div class="status-ok">🟢 Model je naložen v produkciji</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="status-bad">🔴 Model ni naložen</div>', unsafe_allow_html=True)
            st.caption(model["type"])

    except Exception as e:
        st.error(f"Status modelov ni na voljo: {e}")

    st.divider()

    st.markdown("### Ovrednotenje modelov")

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
            st.dataframe(hidro_metrics, use_container_width=True,
                         hide_index=True)
    else:
        st.info("Metrike modela za vodostaj trenutno niso najdene.")

    if risk_metrics is not None and not risk_metrics.empty:
        row = risk_metrics.iloc[0]
        st.markdown("#### Klasifikacija hidrološkega tveganja")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", round(row["accuracy"], 4))
        c2.metric("F1 macro", round(row["f1_macro"], 4))
        c3.metric("Postaje", int(row["num_stations"]))
        c4.metric("Vrstice", int(row["num_rows"]))

        with st.expander("Podrobne metrike modela za tveganje"):
            st.dataframe(risk_metrics, use_container_width=True,
                         hide_index=True)
    else:
        st.info("Metrike modela za tveganje trenutno niso najdene.")

    st.divider()

    st.markdown("### MLflow eksperimenti in register modelov")

    st.info(
        "Eksperimenti, parametri, metrike, artefakti in verzije modelov so shranjeni v MLflow/Dagshub okolju. "
        "Ta pogled povzema produkcijska kandidata, ki ju uporablja aplikacija."
    )

    mlflow_models = pd.DataFrame(
        [
            {
                "Model": "hidro_global_water_level_forecaster",
                "Naloga": "regresija vodostaja",
                "Tip": "LSTM neural network",
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

    st.divider()

    st.markdown("### Poročila validacije podatkov — Great Expectations")

    ge_index = Path("gx/uncommitted/data_docs/local_site/index.html")

    if ge_index.exists():
        with st.expander("Odpri Great Expectations poročilo", expanded=False):
            with open(ge_index, "r", encoding="utf-8") as f:
                components.html(f.read(), height=650, scrolling=True)
    else:
        st.warning("Great Expectations HTML poročilo ni najdeno.")
        st.code("gx/uncommitted/data_docs/local_site/index.html")

    st.divider()

    st.markdown("### Poročila testiranja podatkov — Evidently")

    evidently_reports = sorted(Path("reports/data_testing").glob("*.html"))

    if evidently_reports:
        selected_report = st.selectbox(
            "Izberi Evidently poročilo",
            evidently_reports,
            format_func=lambda p: p.name,
        )

        with st.expander("Odpri Evidently poročilo", expanded=False):
            with open(selected_report, "r", encoding="utf-8") as f:
                components.html(f.read(), height=700, scrolling=True)
    else:
        st.warning("Evidently poročila niso najdena.")
        st.code("reports/data_testing")

    st.divider()

    st.markdown("### Uporabljene in preskočene postaje")

    mapping = load_metric_file(
        "reports/modeling/risk_classifier_used_station_mapping.csv")
    skipped = load_metric_file(
        "reports/modeling/risk_classifier_skipped_stations.csv")
    hidro_skipped = load_metric_file(
        "reports/modeling/hidro_global_skipped_stations.csv")

    if mapping is not None and not mapping.empty:
        with st.expander("Uporabljene hidrološke in vremenske postaje", expanded=False):
            st.dataframe(mapping, use_container_width=True, hide_index=True)

    if skipped is not None and not skipped.empty:
        with st.expander("Preskočene postaje pri modelu tveganja", expanded=False):
            st.dataframe(skipped, use_container_width=True, hide_index=True)

    if hidro_skipped is not None and not hidro_skipped.empty:
        with st.expander("Preskočene postaje pri modelu vodostaja", expanded=False):
            st.dataframe(hidro_skipped, use_container_width=True,
                         hide_index=True)
