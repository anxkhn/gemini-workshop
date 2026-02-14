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
echo "Running tests..."
echo "(Set GEMINI_API_KEY env var to also run integration tests)"
echo ""
python -m pytest tests/ -v --tb=short "$@"
