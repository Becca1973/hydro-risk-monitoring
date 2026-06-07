from fastapi import FastAPI

app = FastAPI(title="Hydro Risk Monitoring API")


@app.get("/")
def root():
    return {
        "message": "Hydro Risk Monitoring API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/models")
def models():
    return {
        "water_level_model": "hidro_global_water_level_forecaster",
        "risk_model": "hidro_risk_classifier",
        "status": "loaded later"
    }


@app.get("/predict/water-level-demo")
def predict_water_level_demo():
    return {
        "station": "2110_Ptuj",
        "predicted_water_level": 0.52,
        "unit": "normalized value",
        "model": "hidro_global_water_level_forecaster"
    }


@app.get("/predict/risk-demo")
def predict_risk_demo():
    return {
        "station": "2110_Ptuj",
        "risk_level": "MEDIUM",
        "model": "hidro_risk_classifier"
    }
