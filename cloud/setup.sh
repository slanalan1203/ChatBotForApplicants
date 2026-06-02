#!/bin/bash
set -e

REPO_DIR="${HOME}/admission-bot"

echo "=== 1. system deps ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip curl git wget

echo "=== 2. install ollama ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
ollama serve &>/dev/null &
sleep 3

echo "=== 3. pull models (in background, may take 10-20 min for first time) ==="
ollama pull qwen2.5:7b-instruct &
ollama pull gemma4:e4b &
ollama pull mistral:latest &
ollama pull llama3:latest &
wait
echo "models ready"

echo "=== 4. python env ==="
cd "$REPO_DIR"
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q

echo "=== 5. populate KB ==="
if [ ! -d "data/chroma" ]; then
    python scripts/download_docs.py
    python scripts/ingest_all.py --reset
fi

echo "=== 6. cache retrieved context for exp 5 ==="
python scripts/cache_retrieved.py

echo "=== setup done. ready for experiments. ==="
echo "next: python scripts/run_gen_eval.py --model qwen2.5:7b-instruct --prompt simple --temperature 0.3 --out data/eval/exp5/qwen_simple_t03.json"
    