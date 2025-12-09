# Helper script to start the FastAPI backend from repository root
# This ensures the Python import path resolves the `api` package under `backend`.
cd $PSScriptRoot\backend
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
