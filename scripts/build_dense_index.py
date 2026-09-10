from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seed import seed_everything


def build_corpus(train_parquet: Path):
    import pandas as pd

    from src.normalize import EMAIL_RE, MENTION_RE, ORDER_RE, PHONE_RE, URL_RE

    df = pd.read_parquet(train_parquet)
    df = df.sort_values(["thread_id", "pos"], kind="mergesort")
    try:
        from src.golden import pseudo_intent
    except Exception:
        pseudo_intent = None
    docs: list[dict] = []
    for tid, grp in df.groupby("thread_id", sort=False):
        inbound = grp[grp["inbound"]]["text_norm"].astype(str).tolist()
        if not inbound:
            inbound = grp["text_norm"].astype(str).tolist()
        text = " | ".join(inbound)
        turns = grp.sort_values("pos")
        ev_ids = [str(t) for t in turns["tweet_id"].astype(str).tolist()]
        start = str(turns["created_at"].iloc[0])
        intent = "other"
        if pseudo_intent is not None:
            try:
                intent, _ = pseudo_intent(text)
            except Exception:
                intent = "other"
        entities = {
            "order": sorted(set(ORDER_RE.findall(text))),
            "url": sorted(set(URL_RE.findall(text))),
            "mention": sorted(set(MENTION_RE.findall(text))),
            "email": sorted(set(EMAIL_RE.findall(text))),
            "phone": sorted(set(PHONE_RE.findall(text))),
        }
        docs.append(
            {
                "id": str(tid),
                "text": text,
                "intent": intent,
                "lang": "en",
                "date": start,
                "evidence_ids": ev_ids,
                "entities": entities,
            }
        )
    docs.sort(key=lambda d: (d["date"], d["id"]))
    return docs


def main(argv: list[str] | None = None) -> int:
    seed_everything(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-index-cache", dest="use_index_cache", action="store_true", default=True)
    ap.add_argument("--no-index-cache", dest="use_index_cache", action="store_false")
    ap.add_argument("--refresh-cache", action="store_true", default=False)
    ap.add_argument("--model", default="minilm")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    import pandas as pd

    from src.embed_index import (
        build_index,
        cache_key,
        encode_corpus,
        load_index_meta,
        resolve_model,
        save_index_meta,
        validate_meta,
    )
    from src.io import log_rows_sha
    from src.logging_utils import get_logger, sha256_file
    from src.manifest import cache_size_mb, write_json, write_manifest

    logger = get_logger("build_dense_index")
    train_path = ROOT / "cache" / "train.parquet"
    corpus_sha = sha256_file(train_path)
    key = cache_key(args.model, corpus_sha)
    key_path = ROOT / "cache" / "faiss_key.json"

    if args.use_index_cache and not args.refresh_cache:
        try:
            index, meta = load_index_meta()
            if key_path.exists() and json.loads(key_path.read_text()).get("key") == key:
                logger.info("index cache hit ntotal=%d, skipping re-encode", index.ntotal)
                from src.embed_index import _index_paths

                paths = _index_paths()
                write_json(
                    {
                        "version": "1.0.0",
                        "seed": 42,
                        "ok": True,
                        "model": resolve_model(args.model)["hf_id"],
                        "dim": index.d,
                        "ntotal": index.ntotal,
                        "encode_seconds": 0.0,
                        "self_recall_at_1": 1.0,
                        "cache": "hit",
                        "cache_mb": round(cache_size_mb(), 2),
                    },
                    ROOT / "manifests" / "s3_2_dense.json",
                )
                write_manifest(sorted((ROOT / "cache").glob("*")))
                print(json.dumps({"cache": "hit", "ntotal": index.ntotal, "key": key[:12]}, sort_keys=True))
                print("DENSE CACHE HIT")
                return 0
            logger.warning("index cache key mismatch, rebuilding")
        except Exception as e:
            logger.info("index cache miss (%s), building", f"{type(e).__name__}: {str(e)[:120]}")

    t0 = time.perf_counter()
    docs = build_corpus(train_path)
    texts = [d["text"] for d in docs]
    cfg = resolve_model(args.model)
    try:
        vecs = encode_corpus(texts, model=args.model, batch_size=args.batch_size)
    except Exception as e:
        reason = f"{type(e).__name__}: {str(e)[:200]}"
        logger.info("dense skipped (%s); sparse-only fallback", reason)
        write_json(
            {"version": "1.0.0", "seed": 42, "ok": False, "reason": reason, "model": cfg["hf_id"]},
            ROOT / "manifests" / "s3_2_dense.json",
        )
        print("DENSE SKIPPED (sparse-only)")
        return 0
    enc_s = time.perf_counter() - t0
    index = build_index(vecs, dim=cfg["dim"])
    assert index.ntotal == len(docs) == len(texts)
    assert index.d == cfg["dim"]
    validate_meta(docs, index.ntotal)
    # brute-force self sanity on sample
    import numpy as np

    from src.embed_index import l2_normalize

    xb = l2_normalize(np.ascontiguousarray(vecs, dtype=np.float32))
    sample = min(100, index.ntotal)
    D, I = index.search(xb[:sample], 1)
    self_rate = float((I[:, 0] == np.arange(sample)).mean())
    assert self_rate == 1.0, f"self-search sanity {self_rate}"
    info = save_index_meta(index, docs, model=args.model, corpus_sha=corpus_sha)
    key_path.write_text(json.dumps({"key": key, "model": cfg["hf_id"]}, sort_keys=True) + "\n", encoding="utf-8")
    rss_mb = -1.0
    if _has_psutil():
        import psutil

        rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    manifest = {
        "version": "1.0.0",
        "seed": 42,
        "ok": True,
        "model": cfg["hf_id"],
        "dim": cfg["dim"],
        "ntotal": index.ntotal,
        "encode_seconds": round(enc_s, 1),
        "self_recall_at_1": self_rate,
        "ram_mb": {"rss": rss_mb, "index_mb": round(int(info["index_bytes"]) / 1e6, 2)},
        "index": info,
        "train_rows": int(pd.read_parquet(train_path).shape[0]),
        "cache_mb": round(cache_size_mb(), 2),
    }
    write_json(manifest, ROOT / "manifests" / "s3_2_dense.json")
    write_manifest(sorted((ROOT / "cache").glob("*")))
    log_rows_sha("faiss.index", ROOT / "cache" / "faiss.index")
    print(json.dumps({"s3_2_dense": manifest}, indent=2, sort_keys=True))
    print("DENSE PASS")
    return 0


def _has_psutil() -> bool:
    try:
        import psutil  # noqa: F401

        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
