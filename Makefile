PY?=python
PIP?=pip

install-repro:
	$(PIP) install -r requirements-repro.txt

install-full:
	$(PIP) install -r requirements-repro.txt
	$(PIP) install -r requirements-full.txt

smoke:
	PYTHONUTF8=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false $(PY) scripts/smoke.py --offline

reproduce:
	PYTHONUTF8=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false $(PY) scripts/smoke.py --use-index-cache --use-judge-cache

live:
	$(PY) scripts/smoke.py --live n=20

verify:
	$(PY) scripts/check_hashes.py

freeze:
	$(PY) scripts/freeze.py
