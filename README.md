# waggle

A customer-support agent for AmazonHelp Twitter threads. It reconstructs conversations from raw tweets, triages incoming messages, drafts grounded replies from past resolutions, and escalates risky cases to a human.

CPU-only, zero-cost to run, fully reproducible in under 15 minutes.

## Features

- Thread reconstruction from reply-link columns (BFS, cycle-safe, fan-out linearized)
- Text normalization pipeline: unescape, mention/URL masking, language filter, signature strip, exact + near-duplicate removal
- Leak-free temporal split (train before 2017-10-15, 24h embargo gap, cross-split near-dup audit)
- Deterministic end to end: pinned seed, hashed manifests, offline fallbacks

## Quickstart

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
run.bat
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
./run.sh
```

The default install is lightweight (no torch). Add `--with-embeddings` for the embedding path.

## How it works

1. **Fetch**: downloads the raw tweet pool via `kagglehub` (pinned version, sha256 logged) and filters to AmazonHelp.
2. **Threads**: rebuilds conversations with BFS over `response_tweet_id` / `in_response_to_tweet_id`, splits on >24h gaps, flags long threads.
3. **Clean**: normalizes text, filters non-English, removes exact and near duplicates (MinHash + TF-IDF verify), scans for residual PII.
4. **Freeze**: writes the temporal split, leakage audit, and `MANIFEST.sha256` so any run can be verified with `python scripts/check_hashes.py`.

```bash
make smoke      # fast offline check
make reproduce  # full cached run
make verify     # hash + gap verification
```

## Project structure

```
config/base.yaml      # single source of config (seed, paths, cut, thresholds)
src/                  # library: config, io, threads, normalize, split, manifest
scripts/              # runnable pipeline: fetch, build_threads, clean, freeze
tests/                # regression tests
cache/                # frozen pool, temporal splits, model, MANIFEST.sha256
data-sample/          # sample.csv, eval_200.parquet, audits
manifests/            # fetch / threads / clean / split / leakage JSON
```

## Configuration

Everything lives in `config/base.yaml`: random seed (42), data paths, temporal cut (`2017-10-15T00:00:00Z`) with a 24h embargo gap, dedup thresholds (MinHash 128-perm, char-5gram, Jaccard 0.85), and language threshold (English > 0.65). Launchers enforce `PYTHONUTF8=1`, `PYTHONHASHSEED=42`, and single-threaded math libs for determinism.

## Data

Source: `thoughtvector/customer-support-on-twitter` (Kaggle, ~2.8M tweets). The raw 516MB download is fetched at runtime and never committed. After brand filtering and cleaning, the pool is ~68.5k AmazonHelp threads / ~236k turns, split into train (12.6k threads) and golden (54.7k threads) pools with zero exact cross-split overlap. Headline numbers are defined on `data-sample/eval_200.parquet`.

## Tech stack

Python 3.11, pandas/pyarrow, scikit-learn, faiss-cpu, datasketch, kagglehub. Heavy/LLM clients are lazy-loaded behind the `--with-embeddings` flag.

## License

Code is mine. The tweet dataset belongs to its Kaggle publishers and is fetched separately under its own terms : it is not redistributed here.
