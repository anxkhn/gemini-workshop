#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv --python 3.11 .venv
fi

source .venv/bin/activate
uv pip install -r requirements.txt

echo ""
echo "Starting Gemini Workshop server at http://127.0.0.1:8000"
echo "Press Ctrl+C to stop."
echo ""
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
