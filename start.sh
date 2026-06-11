#!/bin/bash
set -e

echo "Pulling model artifacts..."
poetry run dvc pull models/hidro_global.dvc models/risk.dvc -v

echo "Starting service..."

if [ "$SERVICE_TYPE" = "ui" ]; then
  streamlit run src/ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
else
  uvicorn src.api.main:app --host 0.0.0.0 --port=${PORT:-8000}
fi