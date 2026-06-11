#!/bin/bash

echo "Pulling DVC artifacts..."

if [ -n "$DAGSHUB_ACCESS_KEY_ID" ] && [ -n "$DAGSHUB_SECRET_ACCESS_KEY" ]; then
  dvc remote modify origin --local access_key_id "$DAGSHUB_ACCESS_KEY_ID"
  dvc remote modify origin --local secret_access_key "$DAGSHUB_SECRET_ACCESS_KEY"
fi

dvc pull -v models/hidro_global models/risk models/onnx || echo "DVC model pull failed, continuing..."

if [ "$SERVICE_TYPE" = "ui" ]; then
  streamlit run src/ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
else
  uvicorn src.api.main:app --host 0.0.0.0 --port=${PORT:-8000}
fi