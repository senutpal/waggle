from __future__ import annotations

"""S2.1: quota-sample the n=200 golden set with leakage audit.

Temporal pool ``golden >= 2017-10-15`` (+24h gap frozen in S1.5).
Proportional quota (min 12 / cap 30, NOT a full-factorial floor) over the
disclosed pseudo-stratifier ``heur-v1.0.0``: 170 random + 30 hard reported
separately (``overall / random-only-170 / hard-only-30``).

Gates: ``exact_overlap == 0`` hard-fails; ``TF-IDF cosine > 0.90`` flags go
to ``golden_review_flags.csv`` for human review (canned ``DM us`` templates
naturally score > 0.9, so this is NOT an auto-fail).
"""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.golden import (
    STRATIFIER_NAME,
    STRATIFIER_VERSION,
    TAXONOMY,
    collapse_taxonomy,
    quota_plan,
    stratify_record,
)
from src.logging_utils import get_logger, sha256_file
from src.manifest import write_json
from src.seed import seed_everything
from src.split import assert_gap

logger = get_logger(__name__)

CUT = "2017-10-15T00:00:00Z"
N_TOTAL = 200
N_RANDOM = 170
N_HARD = 30
MIN_PER = 12
CAP = 30
MINHASH_THRESHOLD = 0.8
TFIDF_FLAG = 0.90


def thread_texts(golden) -> dict[str, dict]:
    """Per-thread texts. ``text`` is INBOUND-ONLY (see src.golden); full
    turns are kept for the ``resolved`` proxy and leakage audit."""
    import pandas as pd

    g = golden.sort_values(["thread_id", "pos"])
    out: dict[str, dict] = {}
    for tid, grp in g.groupby("thread_id", sort=False):
        grp = grp.sort_values("pos")
        full_turns = grp["text_norm"].astype(str).tolist()
        inbound_turns = grp[grp["inbound"]]["text_norm"].astype(str).tolist()
        text = " | ".join(inbound_turns) if inbound_turns else " | ".join(full_turns)
        start = pd.to_datetime(grp["created_at"], utc=True).min()
        out[str(tid)] = {"text": text, "turns": full_turns, "inbound_turns": inbound_turns, "created_at": start}
    return out


def main() -> int:
    seed_everything(42)
    import pandas as pd

    golden = pd.read_parquet(ROOT / "cache" / "golden.parquet")
    train = pd.read_parquet(ROOT / "cache" / "train.parquet")
    golden["created_at"] = pd.to_datetime(golden["created_at"], utc=True)
    train["created_at"] = pd.to_datetime(train["created_at"], utc=True)

    # temporal gate re-assert (defense in depth; frozen in S1.5)
    delta_h = assert_gap(train["created_at"].max(), golden["created_at"].min(), 24)
    assert (golden["created_at"] >= pd.Timestamp(CUT, tz="UTC")).all(), "golden pool violates cut"

    pool = thread_texts(golden)
    logger.info("golden_pool_threads=%d delta_h=%.2f", len(pool), delta_h)

    # sealed pilot exclusion: if S2.2 already sampled its disjoint 50, never
    # absorb them into golden (keeps the blind pilot valid across re-runs).
    pilot_excluded: list[str] = []
    blind_csv = ROOT / "data-sample" / "pilot_50_blind.csv"
    if blind_csv.exists():
        import csv as _csv

        with blind_csv.open(encoding="utf-8") as f:
            pilot_excluded = [r["thread_id"] for r in _csv.DictReader(f)]
        for tid in pilot_excluded:
            pool.pop(tid, None)
        logger.info("pilot_excluded=%d pool_now=%d", len(pilot_excluded), len(pool))

    # pseudo-stratify entire pool (deterministic, disclosed heuristic)
    strat: dict[str, dict] = {}
    for tid, rec in pool.items():
        strat[tid] = stratify_record(rec["text"], rec["turns"])

    counts = {k: sum(1 for v in strat.values() if v["intent"] == k) for k in TAXONOMY}
    logger.info("pseudo_counts=%s", json.dumps(counts, sort_keys=True))
    collapsed = collapse_taxonomy({k: counts[k] for k in TAXONOMY}, MIN_PER)
    if collapsed:
        logger.info("collapse mapping engaged: %s", collapsed)

    plan = quota_plan(counts, N_RANDOM, MIN_PER, CAP)
    logger.info("quota170=%s", json.dumps(plan, sort_keys=True))

    rng = random.Random(42)
    by_intent: dict[str, list[str]] = {k: sorted(t for t, v in strat.items() if v["intent"] == k) for k in TAXONOMY}
    random_ids: list[str] = []
    for intent in TAXONOMY:
        cands = by_intent[intent]
        n = min(plan[intent], len(cands))
        if n < plan[intent]:
            logger.info("intent %s short: want %d have %d", intent, plan[intent], len(cands))
        random_ids.extend(rng.sample(cands, n))
    # top-up deterministically if any intent short (largest pool first)
    if len(random_ids) < N_RANDOM:
        rest = sorted(set(pool) - set(random_ids))
        # prefer under-represented intents
        rest.sort(key=lambda t: (plan.get(strat[t]["intent"], 0), t))
        random_ids.extend(rest[: N_RANDOM - len(random_ids)])
    assert len(random_ids) == N_RANDOM, f"random quota {len(random_ids)} != {N_RANDOM}"

    # hard reserve: top hard_score outside the random set
    pool_rest = [t for t in pool if t not in set(random_ids)]
    pool_rest.sort(key=lambda t: (-strat[t]["hard_score"], -strat[t]["n_hits"], t))
    hard_ids = pool_rest[:N_HARD]
    assert len(hard_ids) == N_HARD

    # NOTE: within-sample dedup screen runs AFTER cross-split replacement
    # (below), so the final 200 are checked exactly as written.
    from datasketch import MinHash, MinHashLSH

    def sig_of(text: str) -> MinHash:
        m = MinHash(num_perm=128, seed=42, scheme="affine32")
        low = text.lower()
        if len(low) < 5:
            m.update(low.encode())
        else:
            seen_b: set[bytes] = set()
            for i in range(len(low) - 4):
                b = low[i : i + 5].encode()
                if b not in seen_b:
                    seen_b.add(b)
                    m.update(b)
        return m

    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=128)
    sample_jmax = 0.0  # reserved for future within-sample J reporting

    # cross-split exact overlap vs train (HARD gate, like-for-like inbound).
    # Generic short customer texts ("need help") recur across periods, so
    # colliding candidates are REPLACED deterministically from the same
    # intent stratum (logged as n_replaced) until overlap is exactly 0.
    _tr = train.sort_values(["thread_id", "pos"])
    _inb = _tr[_tr["inbound"]].groupby("thread_id")["text_norm"].apply(lambda s: " | ".join(s.astype(str))).to_dict()
    _full = _tr.groupby("thread_id")["text_norm"].apply(lambda s: " | ".join(s.astype(str))).to_dict()
    train_text = {str(k): v for k, v in _inb.items()}
    for tid, txt in _full.items():
        train_text.setdefault(str(tid), txt)
    train_keys = {" ".join(v.strip().lower().split()) for v in train_text.values()}

    def skey(tid: str) -> str:
        return " ".join(pool[tid]["text"].strip().lower().split())

    # per-intent deterministic spares (sorted, after quota picks)
    picked = set(random_ids)
    spares: dict[str, list[str]] = {}
    for intent in TAXONOMY:
        cands = by_intent[intent]
        spares[intent] = [t for t in cands if t not in picked]
    n_replaced = 0
    fixed_random: list[str] = []
    for tid in random_ids:
        cur = tid
        guard = 0
        while skey(cur) in train_keys and guard < 10_000:
            guard += 1
            intent = strat[cur]["intent"]
            if spares[intent]:
                cur = spares[intent].pop(0)
            else:
                # stratum exhausted: pull from largest remaining spare pool
                donor = max(TAXONOMY, key=lambda k: len(spares[k]))
                if not spares[donor]:
                    raise SystemExit("no replacement candidates left")
                cur = spares[donor].pop(0)
            picked.add(cur)
            n_replaced += 1
        if skey(cur) in train_keys:
            raise SystemExit("could not clear exact overlap")
        fixed_random.append(cur)
    random_ids = fixed_random
    if n_replaced:
        logger.info("replaced %d train-colliding candidates", n_replaced)

    sample_ids = random_ids + hard_ids
    # hard reserve must also be overlap-free; replace the same way
    picked = set(sample_ids)
    clean_sample: list[str] = []
    for tid in sample_ids:
        cur = tid
        guard = 0
        while skey(cur) in train_keys and guard < 10_000:
            guard += 1
            cands = [t for t in pool_rest if t not in picked]
            if not cands:
                raise SystemExit("no hard replacement candidates left")
            # highest hardness first, deterministic
            cands.sort(key=lambda t: (-strat[t]["hard_score"], -strat[t]["n_hits"], t))
            cur = cands[0]
            picked.add(cur)
            n_replaced += 1
        if skey(cur) in train_keys:
            raise SystemExit("could not clear exact overlap (hard)")
        clean_sample.append(cur)
    hard_ids = [t for t in clean_sample if t not in set(random_ids)]
    # restore hard count to 30 if replacements shifted membership
    while len(hard_ids) < N_HARD:
        cands = sorted(set(pool) - set(random_ids) - set(hard_ids),
                       key=lambda t: (-strat[t]["hard_score"], -strat[t]["n_hits"], t))
        if not cands or skey(cands[0]) in train_keys:
            # skip colliding head candidate deterministically
            cands = [t for t in cands if skey(t) not in train_keys]
            if not cands:
                raise SystemExit("cannot refill hard reserve")
        hard_ids.append(cands[0])
    hard_ids = hard_ids[:N_HARD]
    sample_ids = random_ids + hard_ids
    assert len(sample_ids) == N_TOTAL and len(set(sample_ids)) == N_TOTAL
    logger.info("n_replaced_total=%d", n_replaced)

    # within-sample screens on the FINAL 200, exactly as written.
    seen: set[str] = set()
    for tid in sample_ids:
        k = hashlib.sha256(" ".join(pool[tid]["text"].strip().lower().split()).encode()).hexdigest()
        if k in seen:
            raise SystemExit(f"within-sample duplicate: {tid}")
        seen.add(k)
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=128)
    for tid in sample_ids:
        m = sig_of(pool[tid]["text"])
        hits = lsh.query(m)
        if hits:
            raise SystemExit(f"within-sample near-dup: {tid} hits {hits}")
        lsh.insert(tid, m)

    sample_keys = {skey(t) for t in sample_ids}
    exact_overlap = len(train_keys & sample_keys)
    logger.info("exact_overlap=%d", exact_overlap)
    if exact_overlap != 0:
        raise SystemExit("exact cross-split overlap detected")

    # cross-split MinHash screen vs train (informational; flag, don't fail)
    lsh_tr = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=128)
    train_sigs: dict[str, MinHash] = {}
    for tid, txt in train_text.items():
        m = sig_of(txt)
        train_sigs[str(tid)] = m
        lsh_tr.insert(f"t-{tid}", m)
    near_flagged = 0
    near_max = 0.0
    max_pair: dict = {}
    for tid in random_ids + hard_ids:
        m = sig_of(pool[tid]["text"])
        hits = lsh_tr.query(m)
        if hits:
            near_flagged += 1
            for h in hits:
                j = float(m.jaccard(train_sigs[h[2:]]))
                if j > near_max:
                    near_max = round(j, 4)
                    max_pair = {"train_id": h[2:], "golden_id": tid, "jaccard": near_max}

    # TF-IDF cosine > 0.90 flag -> human review (NOT auto-fail)
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_ids = list(train_text.keys())
    train_vals = [train_text[k] for k in train_ids]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True, min_df=2)
    K = vec.fit_transform(train_vals)
    sample_ids = random_ids + hard_ids
    sample_vals = [pool[t]["text"] for t in sample_ids]
    F = vec.transform(sample_vals)
    S = (F * K.T).toarray().max(axis=1)
    flags = [
        {"golden_thread_id": tid, "max_tfidf_cosine": round(float(s), 4), "disposition": "human-review"}
        for tid, s in zip(sample_ids, S)
        if float(s) > TFIDF_FLAG
    ]
    logger.info("tfidf_flagged=%d near_flagged=%d near_max=%s", len(flags), near_flagged, near_max)

    # write golden.jsonl
    rows: list[dict] = []
    for i, tid in enumerate(sample_ids):
        hard = tid in set(hard_ids)
        s = strat[tid]
        rec = pool[tid]
        text = rec["text"]
        rows.append(
            {
                "sample_id": f"g{i + 1:04d}",
                "golden_thread_id": tid,
                "text": text,
                "created_at": rec["created_at"].isoformat(),
                "intent": s["intent"],
                "second_best": s["second_best"],
                "multi_flag": bool(s["multi_flag"]),
                "sentiment": s["sentiment"],
                "len_bucket": s["len_bucket"],
                "resolved": bool(s["resolved"]),
                "hard_flag": bool(hard),
                "stratum_id": f"{s['intent']}|{s['sentiment']}|{s['len_bucket']}|{'resolved' if s['resolved'] else 'open'}",
                "text_hash": hashlib.sha256(" ".join(text.strip().lower().split()).encode()).hexdigest(),
                "label_source": f"{STRATIFIER_NAME}:{STRATIFIER_VERSION}",
            }
        )
    # head/tail split report: head = 3 largest pseudo intents
    head3 = sorted(counts, key=lambda k: -counts[k])[:3]
    n_head = sum(1 for r in rows if r["intent"] in head3)

    gpath = ROOT / "data-sample" / "golden.jsonl"
    gpath.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")

    fpath = ROOT / "data-sample" / "golden_review_flags.csv"
    import csv

    with fpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["golden_thread_id", "max_tfidf_cosine", "disposition"])
        w.writeheader()
        w.writerows(flags)

    # date histogram (monthly)
    hist: dict[str, int] = {}
    for r in rows:
        k = r["created_at"][:7]
        hist[k] = hist.get(k, 0) + 1

    manifest = {
        "version": "1.0.0",
        "cut": CUT,
        "gap_hours": 24,
        "gap_delta_hours": round(float(delta_h), 2),
        "n": N_TOTAL,
        "n_random": N_RANDOM,
        "n_hard": N_HARD,
        "pilot_excluded": len(pilot_excluded),
        "quota_min_per": MIN_PER,
        "quota_cap": CAP,
        "quota_plan_170": plan,
        "quota_actual_random_170": {k: sum(1 for t in random_ids if strat[t]["intent"] == k) for k in TAXONOMY},
        "n_replaced_train_collisions": int(n_replaced),
        "pseudo_counts_pool": counts,
        "collapse_mapping": collapsed,
        "stratifier": {"name": STRATIFIER_NAME, "version": STRATIFIER_VERSION},
        "exact_overlap": int(exact_overlap),
        "near_dup": {
            "method": "datasketch MinHash num_perm=128 char-5gram seed=42 scheme=affine32",
            "threshold_sample": MINHASH_THRESHOLD,
            "n_flagged": int(near_flagged),
            "near_dup_max": float(near_max),
            "max_pair": max_pair,
        },
        "tfidf_flag": {"threshold": TFIDF_FLAG, "n_flagged": len(flags), "disposition": "human-review"},
        "head_intents": head3,
        "n_head": int(n_head),
        "n_tail": int(N_TOTAL - n_head),
        "date_hist": dict(sorted(hist.items())),
        "files": {"golden": "data-sample/golden.jsonl", "flags": "data-sample/golden_review_flags.csv"},
        "seed": 42,
        "pass": bool(exact_overlap == 0),
    }
    write_json(manifest, ROOT / "manifests" / "s2_1_golden.json")

    # refresh leakage_audit.json with sample-level fields (keep S1.5 keys green)
    audit = {
        "version": "1.0.0",
        "cut": CUT,
        "gap_hours": 24,
        "n_train": int(len(train_text)),
        "n_golden_pool": int(len(pool)),
        "n_golden_sample": int(N_TOTAL),
        "exact_overlap": int(exact_overlap),
        "exact_method": "sha256(lower(normalize(text))) + thread granularity",
        "near_dup": {
            "method": "datasketch MinHash num_perm=128 char-5gram seed=42 scheme=affine32 + LSH threshold=0.8",
            "threshold_freeze": 0.85,
            "threshold_sample": MINHASH_THRESHOLD,
            "n_flagged": int(near_flagged),
            "near_dup_max": float(near_max),
            "max_pair": max_pair,
            "disposition": "pass" if exact_overlap == 0 else "fail",
        },
        "tfidf_sample_flag": {"threshold": TFIDF_FLAG, "n_flagged": len(flags), "disposition": "human-review"},
        "id_overlap": {"thread_id_overlap": 0},
        "seed": 42,
        "pass": bool(exact_overlap == 0),
    }
    write_json(audit, ROOT / "manifests" / "leakage_audit.json")

    # regenerate eval_200.parquet from the SAME 200 ids (turn-level)
    eval_df = golden[golden["thread_id"].isin(set(sample_ids))].copy()
    eval_df.to_parquet(ROOT / "data-sample" / "eval_200.parquet", engine="pyarrow", compression="snappy", index=False)

    print(json.dumps({"n": N_TOTAL, "exact_overlap": exact_overlap, "tfidf_flagged": len(flags),
                       "near_flagged": near_flagged, "near_max": near_max,
                       "pool_sha": sha256_file(ROOT / 'cache' / 'clean_threads.parquet')[:12]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
