#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_EMB=0
for a in "$@"; do
  if [ "$a" = "--with-embeddings" ]; then WITH_EMB=1; fi
done
python -m pip install -q -r "$ROOT/requirements-repro.txt"
if [ "$WITH_EMB" = "1" ]; then python -m pip install -q -r "$ROOT/requirements-full.txt"; fi
python "$ROOT/scripts/smoke.py" "$@"
