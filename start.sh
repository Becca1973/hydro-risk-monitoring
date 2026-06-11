#!/bin/bash
set -e

echo "Starting Hydro Risk service..."
echo "Pulling DVC artifacts from remote storage..."

poetry run dvc pull \
  models/hidro_global.dvc \
  models/risk.dvc \
  models/onnx.dvc \
  preprocess_hidro \
  preprocess_weather \
  -v

echo "DVC artifacts pulled successfully."

if [ "$SERVICE_TYPE" = "ui" ]; then
  streamlit run src/ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
else
  uvicorn src.api.main:app --host 0.0.0.0 --port=${PORT:-8000}
fi