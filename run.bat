@echo off
setlocal
set PYTHONUTF8=1
set PYTHONHASHSEED=42
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
set ROOT=%~dp0
python -m pip install -q -r "%ROOT%requirements-repro.txt"
echo %* | find "--with-embeddings" >nul && python -m pip install -q -r "%ROOT%requirements-full.txt"
python "%ROOT%scripts\smoke.py" %*
