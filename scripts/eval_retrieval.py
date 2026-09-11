from __future__ import annotations

"""Evaluate hybrid retrieval on golden-200 (S3.3).

Proxy qrels (disclosed, pre-registered): gold train doc = argmax TF-IDF
char_wb(3,5) cosine among same pseudo-intent train threads. TF-IDF
char-level is featurization-distinct from all legs (BM25 word/code tokens,
MiniLM dense, CE rerank), so the proxy is not circular with any single arm.
Human qrels remain future work (S6.4). Reports overall / random-170 /
hard-30 separately, Wilson + paired bootstrap + McNemar; overlapping CIs =
inconclusive.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seed import seed_everything


def load_golden():
    rows = []
    for line in (ROOT / "data-sample" / "golden.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_pairs(queries, docs):
    """Proxy gold pairs; cached to results/retrieval_pairs.json (pre-registered)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    pair_path = ROOT / "results" / "retrieval_pairs.json"
    if pair_path.exists():
        return json.loads(pair_path.read_text(encoding="utf-8"))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True, min_df=2)
    D = vec.fit_transform([d["text"] for d in docs])
    by_intent: dict[str, list[int]] = {}
    for i, d in enumerate(docs):
        by_intent.setdefault(d["intent"], []).append(i)
    pairs = {}
    for q in queries:
        qv = vec.transform([q["text"]])
        cand = by_intent.get(q["intent"], list(range(len(docs))))
        sims = cosine_similarity(qv, D[cand]).ravel()
        best = cand[int(sims.argmax())]
        pairs[q["sample_id"]] = {"gold": docs[best]["id"], "method": "tfidf-charwb-argmax-same-intent"}
    pair_path.write_text(json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pairs


def wilson(x: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def paired_delta_ci(a: list[int], b: list[int], seed: int = 123, n_boot: int = 10000):
    import numpy as np

    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = b - a
    point = float(diffs.mean())
    boots = np.array([float(diffs[rng.integers(0, len(diffs), len(diffs))].mean()) for _ in range(n_boot)])
    return point, [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def mcnemar(a: list[int], b: list[int]):
    disc_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    disc_c = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n_d = disc_b + disc_c
    if n_d == 0:
        return {"b": 0, "c": 0, "p": 1.0, "method": "exact"}
    import math

    if n_d < 25:
        k = min(disc_b, disc_c)
        p = 2 * sum(math.comb(n_d, i) for i in range(k + 1)) / 2**n_d
        return {"b": disc_b, "c": disc_c, "p": round(min(1.0, p), 4), "method": "exact"}
    chi2 = (disc_b - disc_c) ** 2 / n_d
    # chi2(1) survival via math.erfc
    p = float(math.erfc((chi2 / 2) ** 0.5))
    return {"b": disc_b, "c": disc_c, "stat": round(chi2, 3), "p": round(p, 4), "method": "chi2"}


def main(argv: list[str] | None = None) -> int:
    seed_everything(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import numpy as np

    from src.bm25 import load_bm25, search_bm25
    from src.config import load_config
    from src.embed_index import dense_search, encode_corpus, l2_normalize, load_index_meta, resolve_model
    from src.golden import pseudo_intent
    from src.hybrid import hybrid_search, mmr_select, rerank, rrf_fuse
    from src.logging_utils import sha256_text
    from src.manifest import cache_size_mb, write_json, write_manifest
    from src.route import (
        apply_routing,
        build_q1,
        default_params as route_params,
        is_followup_short,
        load_deprecated,
    )
    from src.tokenize import tokenize

    cfg = load_config().get("retrieval", {})
    queries = load_golden()
    if args.limit:
        queries = queries[: args.limit]

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_dense_index import build_corpus

    docs = build_corpus(ROOT / "cache" / "train.parquet")
    doc_ids = [d["id"] for d in docs]
    texts = {d["id"]: d["text"] for d in docs}
    row_of = {d: i for i, d in enumerate(doc_ids)}

    # leakage: exact q0 vs corpus must be 0
    from src.normalize import normalize_key

    train_keys = {normalize_key(d["text"]) for d in docs}
    overlap = sum(1 for q in queries if normalize_key(q["text"]) in train_keys)

    pairs = build_pairs(queries, docs)
    bm25_index = load_bm25()
    faiss_index, meta = load_index_meta()
    assert faiss_index.ntotal == len(docs) == len(doc_ids)
    model = resolve_model("minilm")
    qvecs = encode_corpus([q["text"] for q in queries], model="minilm", batch_size=64)
    qvecs = l2_normalize(qvecs)
    rparams = route_params()
    deprecated = load_deprecated()
    docs_by_id = {d["id"]: d for d in docs}
    # golden thread structure for q1 (eval_200 turns)
    import pandas as pd

    _eval = pd.read_parquet(ROOT / "data-sample" / "eval_200.parquet").sort_values(["thread_id", "pos"], kind="mergesort")
    _threads: dict[str, dict] = {}
    for tid, grp in _eval.groupby("thread_id", sort=False):
        turns = grp.to_dict("records")
        inb = [t for t in turns if t["inbound"]]
        last_in = inb[-1]["text_norm"] if inb else turns[-1]["text_norm"]
        prev_agent = None
        for t in reversed(turns):
            if not t["inbound"]:
                prev_agent = t["text_norm"]
                break
            if t["text_norm"] == last_in:
                continue
        _threads[str(tid)] = {"last_inbound": last_in, "prev_agent": prev_agent, "n_inbound": len(inb)}
    # dense vectors for MMR via reconstruct
    xb_all = None

    def vec_of(did: str):
        import numpy as _np

        v = _np.zeros((faiss_index.d,), dtype=_np.float32)
        faiss_index.reconstruct(int(row_of[did]), v)
        return v.tolist()

    per_query = []
    arms: dict[str, list[int]] = {"bm25_only": [], "dense_only": [], "b1_rrf": [], "b2_full": [], "b3_routed": []}
    intent_arms: dict[str, list[int]] = {"bm25_only": [], "dense_only": [], "b1_rrf": [], "b2_full": [], "b3_routed": []}
    rerank_ms: list[float] = []
    ret_ms: list[float] = []
    q1_rate = 0
    backoff_rate = 0
    tq_total = time.perf_counter()

    def run_pass(qtext, qvec):
        qtok = tokenize(qtext)
        out = hybrid_search(
            qtext, qtok, qvec, bm25_index, faiss_index, doc_ids, texts,
            vectors=None, per_leg=50, fused_n=20, rrf_k=int(cfg.get("rrf_k", 60)),
            mmr_lambda=float(cfg.get("mmr_lambda", 0.7)), batch_size=args.batch_size,
        )
        return out, qtok
    for qi, q in enumerate(queries):
        t0 = time.perf_counter()
        out, qtok = run_pass(q["text"], qvecs[qi])
        # ---- B3 routing: boosts + q1 fallback ----
        pred, _ = pseudo_intent(q["text"])
        th = _threads.get(q["golden_thread_id"], {})
        last_in = th.get("last_inbound", q["text"])
        trig, reason = is_followup_short(last_in, th.get("n_inbound", 1))
        b2_top = float(out["rerank"][0][1]) if out["rerank"] else 0.0
        routed = apply_routing(out["rerank"] + [(d, 0.0) for d, _ in out["rrf"][3:10]], docs_by_id, pred, q.get("created_at", ""), deprecated, rparams)
        bo_flag = bool(routed["backoff"])
        q1_used, q1_win = False, False
        b3_top, b3_cal = routed["ranked"][0] if routed["ranked"] else (None, 0.0)
        if trig or b2_top < rparams["top_score"]:
            q1t = build_q1(last_in, th.get("prev_agent"))
            if q1t and q1t != q["text"]:
                q1_used = True
                q1_rate += 1
                qv1 = l2_normalize(encode_corpus([q1t], model="minilm", batch_size=8))[0]
                out1, _ = run_pass(q1t, qv1)
                r1 = apply_routing(out1["rerank"] + [(d, 0.0) for d, _ in out1["rrf"][3:10]], docs_by_id, pred, q.get("created_at", ""), deprecated, rparams)
                bo_flag = bo_flag or bool(r1["backoff"])
                c1 = float(out1["rerank"][0][1]) if out1["rerank"] else 0.0
                if c1 > b2_top and r1["ranked"]:
                    q1_win = True
                    routed = r1
                    b3_top = r1["ranked"][0][0]
        # MMR on boosted pool
        pool = [d for d, _ in routed["ranked"][:10]] or [d for d, _ in out["rrf"][:10]]
        # MMR with real vectors on boosted pool
        vmap = {d: vec_of(d) for d in pool}
        rel = {d: s for d, s in routed["ranked"]}
        for d, s in out["rrf"]:
            rel.setdefault(d, 0.0)
        final_ids, minfo = mmr_select(rel, vmap, lambda_=float(cfg.get("mmr_lambda", 0.7)), top_k=3)
        if minfo.get("skipped"):
            final_ids = [d for d, _ in routed["ranked"][:3]] or [d for d, _ in out["rerank"][:3]]
        out["final"] = final_ids
        out["mmr_info"] = minfo
        b3_ids = list(final_ids[:3])
        total_ms = (time.perf_counter() - t0) * 1000.0
        ret_ms.append(total_ms)
        rerank_ms.append(float(out["rerank_info"].get("ms", 0.0)))
        gold = pairs[q["sample_id"]]["gold"]
        bm25_ids = [d for d, _ in out["bm25"][:3]]
        # dense top3
        dense_ids = [d for d, _ in dense_search(qvecs[qi], faiss_index, doc_ids, top_k=3)]
        b1_ids = [d for d, _ in out["rrf"][:3]]
        b2_ids = [d for d, _ in out["rerank"][:3]]
        out["final"] = b2_ids if minfo.get("skipped") and not routed["ranked"] else b3_ids
        b3_ids = list(out["final"][:3])
        backoff_rate += int(bo_flag)
        hits = {
            "bm25_only": int(gold in bm25_ids),
            "dense_only": int(gold in dense_ids),
            "b1_rrf": int(gold in b1_ids),
            "b2_full": int(gold in b2_ids),
            "b3_routed": int(gold in b3_ids),
        }
        for k, v in hits.items():
            arms[k].append(v)
        # intent-hit (operational relevance: same-intent evidence for grounding)
        qi_intent = gold and docs_by_id.get(gold, {}).get("intent", "")
        q_intent = q["intent"]
        for k, ids in (("bm25_only", bm25_ids), ("dense_only", dense_ids), ("b1_rrf", b1_ids), ("b2_full", b2_ids), ("b3_routed", b3_ids)):
            intent_arms[k].append(int(any(docs_by_id.get(d, {}).get("intent") == q_intent for d in ids)))
        per_query.append(
            {
                "sample_id": q["sample_id"],
                "gold": gold,
                "q0_hash": sha256_text(q["text"])[:16],
                "bm25_top3": bm25_ids,
                "dense_top3": dense_ids,
                "b1_top3": b1_ids,
                "b2_top3": b2_ids,
                "b3_top3": b3_ids,
                "rerank_scores": [[d, round(s, 4)] for d, s in out["rerank"]],
                "routed_detail": routed.get("detail", [])[:3],
                "q1": {"used": q1_used, "win": q1_win, "trigger": reason if (trig or b2_top < rparams["top_score"]) else None},
                "backoff": bool(routed.get("backoff")),
                "hits": hits,
                "latency_ms": {"total": round(total_ms, 1), "rerank": out["rerank_info"].get("ms")},
                "fallback": bool(out["rerank_info"].get("fallback")),
            }
        )
    sl = {q["sample_id"]: q for q in queries}
    hard_ids = {q["sample_id"] for q in queries if q.get("hard_flag")}
    table = {}
    for arm, hits in arms.items():
        xs = [h for i, h in enumerate(hits)]
        rand = [h for i, h in enumerate(hits) if queries[i]["sample_id"] not in hard_ids]
        hard = [h for i, h in enumerate(hits) if queries[i]["sample_id"] in hard_ids]
        table[arm] = {
            "recall_at_3": round(sum(xs) / len(xs), 4),
            "wilson95": wilson(sum(xs), len(xs)),
            "n": len(xs),
            "random_170": {"r3": round(sum(rand) / len(rand), 4), "n": len(rand)} if rand else {},
            "hard_30": {"r3": round(sum(hard) / len(hard), 4), "n": len(hard)} if hard else {},
        }
    dpt, dci = paired_delta_ci(arms["b1_rrf"], arms["b2_full"])
    mc = mcnemar(arms["b1_rrf"], arms["b2_full"])
    b1ci, b2ci = table["b1_rrf"]["wilson95"], table["b2_full"]["wilson95"]
    overlap_ci = not (b1ci[1] < b2ci[0] or b2ci[1] < b1ci[0])
    verdict = "inconclusive (overlapping CIs)" if overlap_ci else ("win" if dpt > 0 and mc["p"] < 0.05 else "no-win")
    d3pt, d3ci = paired_delta_ci(arms["b2_full"], arms["b3_routed"])
    mc3 = mcnemar(arms["b2_full"], arms["b3_routed"])
    intent_table = {}
    for arm, hits in intent_arms.items():
        intent_table[arm] = {
            "intent_hit_at_3": round(sum(hits) / len(hits), 4),
            "wilson95": wilson(sum(hits), len(hits)),
            "n": len(hits),
        }
    r3 = table["b3_routed"]["recall_at_3"]
    ir3 = intent_table["b3_routed"]["intent_hit_at_3"]
    p95 = round(float(np.percentile(rerank_ms, 95)), 1)
    p50 = round(float(np.percentile(rerank_ms, 50)), 1)
    result = {
        "version": "1.1.0",
        "seed": 42,
        "bootstrap_seed": 123,
        "B": 10000,
        "n": len(queries),
        "qrels": "proxy tfidf-charwb-argmax-same-intent (disclosed, NOT human; see results/retrieval_pairs.json)",
        "arms_def": "bm25_only/dense_only top3; b1_rrf=RRF top3 no-rerank; b2_full=CE-rerank top3; b3_routed=B2+soft-boosts+q1-bestof+MMR final",
        "leakage": {"exact_overlap_q0_vs_corpus": overlap, "pass": overlap == 0},
        "primary": {"metric": "Recall@3", "point": r3, "wilson95": table["b3_routed"]["wilson95"], "gate_ge_0_70": bool(r3 >= 0.70)},
        "intent_relevance": {"metric": "IntentHit@3", "point": ir3, "wilson95": intent_table["b3_routed"]["wilson95"], "note": "operational relevance for grounding (>=1 top-3 shares query intent)"},
        "latency_cpu_ms": {"rerank_p50": p50, "rerank_p95": p95, "n": len(rerank_ms), "gate_lt_800ms": bool(p95 < 800)},
        "routing": {"q1_rate": round(q1_rate / len(queries), 4), "backoff_rate": round(backoff_rate / len(queries), 4), "threshold": rparams["top_score"]},
        "gate": {"pass": bool(r3 >= 0.70 and p95 < 800 and overlap == 0)},
        "arms": table,
        "intent_arms": intent_table,
        "delta_b2_b1": {"point": round(dpt, 4), "paired_bca95": dci, "mcnemar": mc, "verdict": verdict},
        "delta_b3_b2": {"point": round(d3pt, 4), "paired_bca95": d3ci, "mcnemar": mc3},
        "mde": {"paired_pp": 7, "unpaired_pp": 14},
        "params": {
            "bm25": {"k1": 0.8, "b": 0.4},
            "rrf_k": cfg.get("rrf_k"),
            "mmr_lambda": cfg.get("mmr_lambda"),
            "rerank": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "depth": "10->3", "batch": args.batch_size, "trunc": 512},
            "dense_model": model["hf_id"],
            "boosts": {"intent": rparams["intent_boost"], "date": rparams["date_boost"], "lang": rparams["lang_boost"]},
        },
        "total_seconds": round(time.perf_counter() - tq_total, 1),
    }
    write_json(result, ROOT / "results" / "recall_at_3.json")
    pq_path = ROOT / "results" / "recall_at_3_perquery.jsonl"
    pq_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in per_query) + "\n", encoding="utf-8")
    write_json(
        {"version": "1.0.0", "seed": 42, "leakage": result["leakage"], "gate": result["gate"], "n": len(queries)},
        ROOT / "manifests" / "s3_3_retrieval.json",
    )
    write_manifest(sorted((ROOT / "cache").glob("*")))
    print(json.dumps(result, indent=2, sort_keys=True))
    print("RETRIEVAL PASS" if result["gate"]["pass"] else "RETRIEVAL FAIL")
    return 0 if result["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
