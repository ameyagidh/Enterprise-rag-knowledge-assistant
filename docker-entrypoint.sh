#!/usr/bin/env bash
# Runs both the FastAPI backend and the Streamlit UI in a single container.
# Streamlit talks to the API over localhost, so no extra networking is needed.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

uvicorn rag_assistant.api.main:app --host 0.0.0.0 --port "${API_PORT}" &
API_PID=$!

# Give the API a moment to finish its startup index-build before the UI's
# first health check, purely cosmetic (the UI retries on its own regardless).
sleep 2

API_BASE_URL="http://localhost:${API_PORT}" \
    streamlit run ui/streamlit_app.py \
    --server.port "${STREAMLIT_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true &
UI_PID=$!

# If either process dies, bring the whole container down so orchestrators
# (compose, k8s) can restart it, rather than limping along half-broken.
wait -n "${API_PID}" "${UI_PID}"
exit $?
