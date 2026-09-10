from __future__ import annotations

"""S2.2: pilot agreement harness (50 disjoint double-blind threads).

Two-phase flow keeps the human rater blind:

1. ``python scripts/label_pilot.py`` -> writes
   ``data-sample/pilot_50_blind.csv`` (pilot_id, thread_id, text,
   created_at — NO labels) and exits.
2. Human labels ``data-sample/pilot_50_rater_a.csv`` (pilot_id,label)
   using ``codebook.md`` WITHOUT opening ``golden.jsonl`` or the script
   output below.
3. ``python scripts/label_pilot.py`` again -> rater B is the disclosed
   pseudo-stratifier ``heur-v1.0.0`` (independent of the human), writes
   ``pilot_50_rater_b.csv``, ``adjudication_log.csv`` and
   ``results/agreement.json`` (Cohen kappa + per-class + Krippendorff
   alpha, all with percentile bootstrap 95% CIs; NO QWK — nominal task).

Gate: overall kappa >= 0.6 AND lower CI > 0.4, else rewrite the rubric
and HALT (do not proceed to S2.3).
"""

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agree import bootstrap_ci, cohen_kappa, krippendorff_alpha_nominal, per_class_kappa
from src.golden import STRATIFIER_NAME, STRATIFIER_VERSION, TAXONOMY, pseudo_intent
from src.logging_utils import get_logger
from src.manifest import write_json
from src.seed import seed_everything

logger = get_logger(__name__)

N_PILOT = 50
PILOT_SEED = 43


def main() -> int:
    seed_everything(42)
    import pandas as pd

    golden_ids = {json.loads(line)["golden_thread_id"] for line in (ROOT / "data-sample" / "golden.jsonl").read_text(encoding="utf-8").splitlines()}
    golden = pd.read_parquet(ROOT / "cache" / "golden.parquet")
    pool_ids = sorted(set(golden["thread_id"].astype(str)) - golden_ids)
    rng = random.Random(PILOT_SEED)
    pilot_ids = rng.sample(pool_ids, N_PILOT)

    g = golden.set_index("thread_id")
    blind_path = ROOT / "data-sample" / "pilot_50_blind.csv"
    if not blind_path.exists():
        with blind_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pilot_id", "thread_id", "text", "created_at"])
            w.writeheader()
            for i, tid in enumerate(pilot_ids):
                grp = golden[golden["thread_id"] == tid].sort_values("pos")
                inbound = grp[grp["inbound"]]["text_norm"].astype(str).tolist() or grp["text_norm"].astype(str).tolist()
                w.writerow({
                    "pilot_id": f"p{i + 1:02d}",
                    "thread_id": tid,
                    "text": " | ".join(inbound),
                    "created_at": pd.to_datetime(grp["created_at"], utc=True).min().isoformat(),
                })
        print(json.dumps({"phase": 1, "blind": str(blind_path), "n": N_PILOT,
                          "msg": "label data-sample/pilot_50_rater_a.csv (pilot_id,label) from codebook.md, then re-run"}, indent=2))
        return 0

    # phase 2: rater A must exist; rater B = disclosed heuristic (independent run)
    a_path = ROOT / "data-sample" / "pilot_50_rater_a.csv"
    if not a_path.exists():
        print(json.dumps({"phase": 1, "waiting_for": str(a_path)}))
        return 0
    blind = {r["pilot_id"]: r for r in csv.DictReader(blind_path.open(encoding="utf-8"))}
    labels_a = {r["pilot_id"]: r["label"].strip() for r in csv.DictReader(a_path.open(encoding="utf-8"))}
    bad = [k for k, v in labels_a.items() if v not in TAXONOMY]
    if bad:
        raise SystemExit(f"rater_a has off-taxonomy labels: {bad}")
    if set(labels_a) != set(blind):
        raise SystemExit("rater_a pilot_id set != blind set")

    order = sorted(blind)
    y1 = [labels_a[k] for k in order]
    y2 = [pseudo_intent(blind[k]["text"])[0] for k in order]

    b_path = ROOT / "data-sample" / "pilot_50_rater_b.csv"
    with b_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pilot_id", "label", "label_source"])
        w.writeheader()
        for k, lab in zip(order, y2):
            w.writerow({"pilot_id": k, "label": lab, "label_source": f"{STRATIFIER_NAME}:{STRATIFIER_VERSION}"})

    kappa = bootstrap_ci(cohen_kappa, y1, y2, n_boot=10_000, seed=42)
    alpha = bootstrap_ci(krippendorff_alpha_nominal, y1, y2, n_boot=10_000, seed=42)
    per = per_class_kappa(y1, y2, TAXONOMY)
    multi_rate = sum(1 for k in order if len(__import__("src.golden", fromlist=["intent_hits"]).intent_hits(blind[k]["text"])) >= 2) / len(order)

    # adjudication: priority order wins (codebook rule); log every disagreement
    from src.golden import INTENT_INDEX

    adj_path = ROOT / "data-sample" / "adjudication_log.csv"
    n_dis = 0
    with adj_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pilot_id", "thread_id", "rater_a", "rater_b", "adjudicated", "rule", "note"])
        w.writeheader()
        for k, a, b in zip(order, y1, y2):
            if a == b:
                continue
            n_dis += 1
            win = a if INTENT_INDEX[a] <= INTENT_INDEX[b] else b
            w.writerow({"pilot_id": k, "thread_id": blind[k]["thread_id"], "rater_a": a, "rater_b": b,
                        "adjudicated": win, "rule": "taxonomy-priority-wins",
                        "note": "multi-intent; lowest priority index kept per codebook"})

    agreement = {
        "version": "1.0.0",
        "n": N_PILOT,
        "disjoint_from_golden200": True,
        "rater_a": "human-blind-codebook",
        "rater_b": f"{STRATIFIER_NAME}:{STRATIFIER_VERSION}",
        "cohen_kappa": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kappa.items()},
        "krippendorff_alpha_nominal": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in alpha.items()},
        "per_class_kappa": {k: round(v, 4) for k, v in per.items()},
        "multi_label_rate": round(multi_rate, 4),
        "n_disagreements": int(n_dis),
        "gate": {"kappa_ge_0_6": bool(kappa["point"] >= 0.6), "lower_ci_gt_0_4": bool(kappa["lower"] > 0.4)},
        "pass": bool(kappa["point"] >= 0.6 and kappa["lower"] > 0.4),
        "seed": 42,
    }
    write_json(agreement, ROOT / "results" / "agreement.json")
    logger.info("kappa=%.3f [%.3f, %.3f] alpha=%.3f disagreements=%d",
                kappa["point"], kappa["lower"], kappa["upper"], alpha["point"], n_dis)
    print(json.dumps(agreement, indent=2, sort_keys=True))
    if not agreement["pass"]:
        raise SystemExit("agreement gate FAILED: rewrite codebook rubric and halt (do not proceed to S2.3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
