from __future__ import annotations

"""Query routing with soft boosts and q0/q1 fallback (S3.4).

Rules:
- q0 = inbound-only turns joined (never agent ``DM us`` boilerplate).
- SOFT boosts (additive, post-rerank, calibrated [0,1] units): intent +0.2,
  date +0.05*exp(-age/365*0.15), lang +0.02. Never hard intent/lang/date
  filters (T1 ~60-68% noisy on human truth).
- Sole hard rule: exact-match block of deprecated ``evidence_ids``,
  pre-RRF-union. ``filtered<3`` backs off to unfiltered (deprecated still
  blocked) with ``backoff=True``.
- q1 = ``last_inbound [SEP] prev_agent_action_only`` (apology/DM stripped,
  Tier-V signals kept) when follow-up/short OR calibrated top_score<0.35.
  Best-of wins, else abstain path. Threshold 0.35 is an INIT prior tuned to
  hit S4 abstain 8-15% (final sweep in S4).
"""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT
from src.logging_utils import get_logger

logger = get_logger(__name__)

APOLOGY_RE = re.compile(
    r"(?i)\b(sorry|apolog\w+|thanks?\s+for\s+(your\s+)?patience)\b[^.!?]*[.!?]?"
)
DM_BOILER_RE = re.compile(
    r"(?i)(please\s+)?(follow\s+and\s+)?DM\s+(us|@USER)[^.!?]*[.!?]?|send\s+(us\s+)?a\s+DM[^.!?]*[.!?]?|let us know if.*|anything else.*help.*"
)
FOLLOWUP_RE = re.compile(
    r"(?i)^\s*(yes|yeah|yep|no|ok|okay|thanks|thank you|still|but|it|that|this|there|they|he|she)\b"
)
TIERV_RE = re.compile(
    r"(?i)(ORD-12345|<URL>|__phone__|__email__|refund|return|\brma\b|policy|amount|\$|£|order\s*(#|number|id)?|track)"
)


def default_params() -> dict:
    try:
        from src.config import load_config

        r = load_config().get("retrieval", {})
        return {
            "intent_boost": float(r.get("boosts", {}).get("intent", 0.2)),
            "date_boost": float(r.get("boosts", {}).get("date", 0.05)),
            "date_decay": float(r.get("boosts", {}).get("date_decay", 0.15)),
            "lang_boost": float(r.get("boosts", {}).get("lang", 0.03)),
            "top_score": float(r.get("thresholds", {}).get("top_score", 0.35)),
            "backoff_min": int(r.get("thresholds", {}).get("backoff_min_hits", 3)),
        }
    except Exception:
        return {
            "intent_boost": 0.2, "date_boost": 0.05, "date_decay": 0.15,
            "lang_boost": 0.03, "top_score": 0.35, "backoff_min": 3,
        }


def build_q0(inbound_texts: list[str]) -> str:
    """Inbound-only query (falls back to all turns if no inbound)."""
    texts = [t for t in inbound_texts if t] or []
    return " | ".join(texts)


def is_followup_short(last_inbound: str, n_inbound: int = 1) -> tuple[bool, str]:
    """Cheap lexical pre-check for context-dependent queries."""
    t = (last_inbound or "").strip()
    if not t:
        return True, "empty"
    if len(t) < 140:
        # short is necessary but not sufficient alone; combine below
        pass
    else:
        return False, "long"
    if len(t.split()) < 8:
        return True, "short-tokens"
    if FOLLOWUP_RE.search(t):
        return True, "pronoun-lead"
    if ("?" in t and "!" in t) or t.count("?") >= 2:
        return True, "chase-punct"
    if not TIERV_RE.search(t) and n_inbound >= 2:
        return True, "anaphoric"
    return False, "standalone-short"


def strip_agent_action(agent_text: str) -> str:
    """Keep Tier-V action signals, strip apology + DM boilerplate."""
    t = APOLOGY_RE.sub(" ", agent_text or "")
    t = DM_BOILER_RE.sub(" ", t)
    kept = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    action = [s for s in kept if TIERV_RE.search(s)]
    out = " ".join(action if action else kept).strip()
    return re.sub(r"\s+", " ", out)


def build_q1(last_inbound: str, prev_agent: str | None) -> str:
    """Fallback query: last customer turn + distilled agent action."""
    action = strip_agent_action(prev_agent or "")
    if action:
        return f"{last_inbound} [SEP] {action}"
    return last_inbound


def load_deprecated() -> set[str]:
    for cand in (ROOT / "config" / "deprecated_ids.json", ROOT / "cache" / "deprecated_ids.json"):
        try:
            if cand.exists():
                obj = json.loads(cand.read_text(encoding="utf-8"))
                ids = obj.get("ids", obj) if isinstance(obj, dict) else obj
                return {str(i).strip() for i in ids if str(i).strip()}
        except Exception as e:
            logger.warning("deprecated load failed %s: %s", cand, e)
    return set()


def _parse_date(s: str):
    try:
        import pandas as pd

        return pd.to_datetime(s, utc=True)
    except Exception:
        return None


def soft_boost(
    calibrated: float,
    doc: dict,
    pred_intent: str,
    query_date: str = "",
    params: dict | None = None,
) -> tuple[float, dict]:
    """Additive boosts in calibrated units. Returns (final, breakdown)."""
    p = params or default_params()
    parts = {"intent": 0.0, "date": 0.0, "lang": 0.0}
    if pred_intent and pred_intent != "other" and doc.get("intent") == pred_intent:
        parts["intent"] = p["intent_boost"]
    qd, dd = _parse_date(query_date), _parse_date(doc.get("date", ""))
    if qd is not None and dd is not None and not (hasattr(qd, "isna") and False):
        try:
            import pandas as pd

            if pd.isna(qd) or pd.isna(dd):
                raise ValueError("nat")
            age_days = max(0.0, (qd - dd).total_seconds() / 86400.0)
            parts["date"] = p["date_boost"] * math.exp(-age_days / 365.0 * p["date_decay"])
        except Exception:
            parts["date"] = 0.0
    if doc.get("lang") and doc.get("lang") == "en":
        parts["lang"] = p["lang_boost"]
    final = min(1.0, max(0.0, calibrated + sum(parts.values())))
    return final, parts


def apply_routing(
    reranked: list[tuple[str, float]],
    docs_by_id: dict[str, dict],
    pred_intent: str,
    query_date: str = "",
    deprecated: set[str] | None = None,
    params: dict | None = None,
) -> dict:
    """Apply deprecated block + soft boosts + backoff. Pure function."""
    p = params or default_params()
    dep = set(deprecated or set())
    # block deprecated pre-rank (exact evidence_ids match)
    kept = []
    excluded = []
    for did, score in reranked:
        doc = docs_by_id.get(did, {})
        ev = set(doc.get("evidence_ids", []))
        if ev & dep or did in dep:
            excluded.append(did)
            continue
        kept.append((did, score))
    backoff = False
    if len(kept) < p["backoff_min"]:
        backoff = True  # backs off to unfiltered (deprecated still blocked)
        kept = list(reranked[: max(p["backoff_min"], len(reranked))])
        kept = [(d, s) for d, s in kept if d not in dep and not (set(docs_by_id.get(d, {}).get("evidence_ids", [])) & dep)]
    boosted = []
    for did, score in kept:
        final, parts = soft_boost(score, docs_by_id.get(did, {}), pred_intent, query_date, p)
        boosted.append((did, final, score, parts))
    boosted.sort(key=lambda t: (-t[1], t[0]))
    return {
        "ranked": [(d, f) for d, f, _, _ in boosted],
        "detail": [{"id": d, "final": round(f, 4), "cal": round(c, 4), **{k: round(v, 4) for k, v in parts.items()}} for d, f, c, parts in boosted],
        "excluded": excluded,
        "backoff": backoff,
    }
