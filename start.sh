#!/bin/bash
set -e

echo "Pulling DVC artifacts..."

poetry run dvc pull \
  validate_data \
  test_data \
  prepare_production_data \
  train_hidro \
  train_risk_classifier -v

echo "Starting service..."

if [ "$SERVICE_TYPE" = "ui" ]; then
  streamlit run src/ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
else
  uvicorn src.api.main:app --host 0.0.0.0 --port=${PORT:-8000}
fi