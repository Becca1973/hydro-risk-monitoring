#!/bin/bash
set -e

echo "Starting Hydro Risk service..."

if [ "$SERVICE_TYPE" != "ui" ]; then
  echo "Pulling DVC artifacts for API service..."

  poetry run dvc pull \
    validate_data \
    test_data \
    prepare_production_data \
    train_hidro \
    train_risk_classifier -v

  echo "DVC artifacts pulled."
else
  echo "UI service: skipping DVC pull."
fi

echo "Starting service..."

if [ "$SERVICE_TYPE" = "ui" ]; then
  streamlit run src/ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
else
  uvicorn src.api.main:app --host 0.0.0.0 --port=${PORT:-8000}
fi