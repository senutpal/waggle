from __future__ import annotations

"""Two-tier grounding: citation + entity-verbatim + NLI tie-break (S4.2).

Twitter threads are NOT a KB — most agent turns are generic ``DM us``
boilerplate. Tier-V (verifiable claims: link/phone/policy/refund/amount/
date/order) REQUIRES ``set(out.evidence_ids)<=set(retrieved_ids)`` + entity
allowlist verbatim + NLI. Tier-B (empathy/process boilerplate
``sorry, DM with details``) requires NO citation.

``faithful()`` — ANY-claim-fail → hedge/escalate (NOT all-fail).
NLI ``DeBERTa-v3-base-mnli-fever-anli`` e-c>=0.2 per claim as medium-band
TIE-BREAK only (MiniLM swap for CPU) — claim-splitter pinned + versioned.
Abstain if cross-encoder ``ret_score<0.30`` or ``contra>0.6`` or any Tier-V
NLI fail → ``INSUFFICIENT_EVIDENCE``. Semantic backstop: claim not entailed
by versioned source → DO-NOT-AUTO, not just string blocklist.
"""

import os
import re
import unicodedata
from dataclasses import dataclass, field

from src import _lazy
from src.logging_utils import get_logger

logger = get_logger(__name__)

SPLITTER_VERSION = "splitter@v1.0.0"
NLI_MODEL_DEFAULT = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_MODEL_SWAP = "cross-encoder/nli-MiniLM2-L6-H768"

TIERV_RE = re.compile(
    r"(https?://|www\.|<URL>|__phone__|__email__|ORD-12345|refund|return|\brma\b|policy|amount|\$|\u00a3|order\s*(#|number|id)?|track|\b\d{4}-\d{2}-\d{2}\b|money\s?back|pay\s?back|reimburs\w*|compensat\w*|guarantee\w*|payback)",
    re.IGNORECASE,
)
TIERB_RE = re.compile(
    r"(?i)^\s*(sorry|apolog\w+|thanks?\b)[^.!?]*[.!?]?\s*(please\s+)?(DM\s+(us|@USER)|send\s+(us\s+)?a\s+DM)[^.!?]*[.!?]?\s*$"
)
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'@#$])")
ENTITY_PATTERNS: dict[str, re.Pattern] = {
    "url": re.compile(r"https?://[^\s<>\]\)\"']+|<URL>|www\.[^\s<>\]\)\"']+", re.IGNORECASE),
    "phone": re.compile(r"__phone__|\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "order": re.compile(r"ORD-12345|\b\d{3}-\d{7}-\d{7}\b", re.IGNORECASE),
    "email": re.compile(r"__email__|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "amount": re.compile(r"\$ ?\d[\d,.]*|\u00a3 ?\d[\d,.]*|\b\d+(?:\.\d{2})\s?(?:USD|dollars?)\b", re.IGNORECASE),
    "date": re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        re.IGNORECASE,
    ),
}
_TRAILING = ".,;:!?)\\]}'\""

_nli = None


@dataclass(frozen=True)
class ClaimResult:
    text: str
    tier: str  # Tier-V | Tier-B
    cited: bool
    entity_ok: bool
    entail: float | None
    contra: float | None
    verdict: str


@dataclass(frozen=True)
class FaithfulResult:
    ok: bool
    reason: str
    claims: list[ClaimResult] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)


def default_ground_params() -> dict:
    try:
        from src.config import load_config

        g = load_config().get("generation", {})
        nli = g.get("nli", {}) if isinstance(g.get("nli", {}), dict) else {}
        t = g.get("thresholds", {}) if isinstance(g.get("thresholds", {}), dict) else {}
        return {
            "nli_model": str(nli.get("model", NLI_MODEL_DEFAULT)),
            "nli_min": float(nli.get("entail_min", 0.2)),
            "ret_min": float(t.get("ret_score_min", 0.30)),
            "contra_max": float(t.get("contra_max", 0.6)),
        }
    except Exception:
        return {"nli_model": NLI_MODEL_DEFAULT, "nli_min": 0.2, "ret_min": 0.30, "contra_max": 0.6}


def split_claims(body: str) -> list[str]:
    parts = [s.strip() for s in SENT_RE.split((body or "").strip()) if s.strip()]
    if not parts and (body or "").strip():
        return [(body or "").strip()]
    return parts


def claim_tier(claim: str) -> str:
    c = (claim or "").strip()
    if not c:
        return "Tier-B"
    if TIERB_RE.match(c):
        # a DM-request template with a Tier-V sentence appended is still Tier-V
        if TIERV_RE.search(c) and len(c) > 60:
            pass
        else:
            return "Tier-B"
    if TIERV_RE.search(c):
        return "Tier-V"
    try:
        ents = extract_entities(c)
        if any(ents.values()):
            return "Tier-V"
    except Exception:
        pass
    # short empathy scaffolding without entities is boilerplate
    if len(c) < 80 and not ENTITY_PATTERNS["url"].search(c):
        return "Tier-B"
    return "Tier-B"


def extract_entities(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for kind, rx in ENTITY_PATTERNS.items():
        vals = []
        for m in rx.finditer(text or ""):
            v = m.group(0).strip().rstrip(_TRAILING)
            if kind == "phone" and "__phone__" not in v:
                digits = re.sub(r"\D", "", v)
                if len(digits) not in (10, 11):
                    continue
            if v:
                vals.append(v)
        out[kind] = vals
    return out


def _evidence_blob(evidence_texts: list[str]) -> tuple[str, str]:
    blob = "\n".join(evidence_texts or [])
    return blob, blob.lower()


def check_entities_verbatim(draft: str, evidence_texts: list[str]) -> tuple[bool, list[str]]:
    blob, blob_low = _evidence_blob(evidence_texts)
    blob_digits = re.sub(r"\D", "", blob)
    violations: list[str] = []
    ents = extract_entities(draft or "")
    for u in ents["url"]:
        if u == "<URL>":
            if "<URL>" not in blob:
                violations.append(f"url:{u}")
        elif u not in blob and u.lower() not in blob_low:
            violations.append(f"url:{u}")
    for p in ents["phone"]:
        if p == "__phone__":
            if "__phone__" not in blob:
                violations.append(f"phone:{p}")
        elif p not in blob and re.sub(r"\D", "", p) not in blob_digits:
            violations.append(f"phone:{p}")
    for o in ents["order"]:
        if o not in blob:
            violations.append(f"order:{o}")
    for e in ents["email"]:
        if e == "__email__":
            if "__email__" not in blob:
                violations.append(f"email:{e}")
        elif e not in blob and e.lower() not in blob_low:
            violations.append(f"email:{e}")
    for a in ents["amount"]:
        norm = re.sub(r"\s+", "", a.lower().replace(",", ""))
        if norm not in re.sub(r"\s+", "", blob.lower().replace(",", "")):
            violations.append(f"amount:{a}")
    for d in ents["date"]:
        if d.lower().strip(_TRAILING) not in blob_low:
            violations.append(f"date:{d}")
    return (len(violations) == 0, violations)


def get_nli(model_id: str | None = None):
    """Lazy NLI cross-encoder. Raises RuntimeError on repro (caller fail-closes)."""
    global _nli
    if _nli is not None:
        return _nli
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _lazy.require_extra(
        "transformers", "pip install -r requirements-full.txt or run with --with-embeddings"
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    mid = model_id or default_ground_params()["nli_model"]
    if os.environ.get("WAGGLE_NLI_SWAP", "0") == "1":
        mid = NLI_MODEL_SWAP
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(mid, trust_remote_code=False)
    try:
        model.eval()
        import torch  # type: ignore[import-not-found]

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _nli = (tok, model)
    return _nli


def nli_scores(
    premises: list[str],
    hypotheses: list[str],
    *,
    model_id: str | None = None,
    batch_size: int = 16,
    max_length: int = 256,
    _inject: tuple[list[float], list[float]] | None = None,
) -> tuple[list[float], list[float]]:
    """Return (entail[], contra[]) in [0,1]. ``_inject`` bypasses model (tests/repro)."""
    if _inject is not None:
        return _inject
    tok, model = get_nli(model_id)
    import torch  # type: ignore[import-not-found]

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    ent_idx = next((i for i, l in id2label.items() if "entail" in l), 0)
    con_idx = next((i for i, l in id2label.items() if "contra" in l), None)
    ent: list[float] = []
    con: list[float] = []
    with torch.no_grad():
        for i in range(0, len(premises), batch_size):
            p = premises[i : i + batch_size]
            h = hypotheses[i : i + batch_size]
            feat = tok(p, h, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            logits = model(**{k: v for k, v in feat.items()}).logits
            probs = torch.softmax(logits, dim=-1).tolist()
            for row in probs:
                ent.append(float(row[ent_idx]))
                con.append(float(row[con_idx]) if con_idx is not None else 0.0)
    return ent, con


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[\"'`\u201c\u201d\u2018\u2019\u00ab\u00bb]", "", s)).strip()


def resolve_quote_offsets(doc_text: str, quote: str, rho: float = 0.85) -> tuple[int, int] | None:
    if not doc_text or not quote:
        return None
    a = doc_text.find(quote)
    if a >= 0:
        return (a, a + len(quote))
    nd, nq = _norm(doc_text), _norm(quote)
    if not nq or len(nq) < 4 or nq not in nd:
        return None
    import difflib

    a = nd.find(nq)
    sigma = difflib.SequenceMatcher(None, nq, nd[max(0, a - 20) : a + len(nq) + 20]).ratio()
    if sigma < rho:
        return None
    m = re.search(re.escape(quote[:30].strip()), doc_text)
    if m:
        return (m.start(), min(len(doc_text), m.start() + len(quote)))
    return None


def extract_evidence_spans(
    draft_entities: list[str],
    docs_by_id: dict[str, dict],
    retrieved_ids: list[str],
) -> list[dict]:
    spans: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for ent in sorted({e.strip() for e in draft_entities if e and e.strip()}, key=len, reverse=True):
        for doc_id in retrieved_ids:
            text = str((docs_by_id.get(doc_id) or {}).get("text", ""))
            loc = resolve_quote_offsets(text, ent)
            if loc is None:
                continue
            key = (doc_id, loc[0], loc[1])
            if key in seen:
                continue
            seen.add(key)
            spans.append({"doc_id": doc_id, "quote": text[loc[0] : loc[1]], "start": loc[0], "end": loc[1]})
            break
    return spans


def faithful(
    body: str,
    evidence_ids: list[str],
    retrieved_ids: list[str],
    evidence_texts: list[str],
    *,
    ret_score: float = 1.0,
    contra: float = 0.0,
    nli_entails: list[float] | None = None,
    nli_contras: list[float] | None = None,
    nli_min: float | None = None,
    ret_min: float | None = None,
    contra_max: float | None = None,
    _nli_inject: tuple[list[float], list[float]] | None = None,
) -> FaithfulResult:
    """ANY-claim-fail → False. Pure except optional lazy NLI (fail-closed on miss)."""
    p = default_ground_params()
    nli_min = float(nli_min if nli_min is not None else p["nli_min"])
    ret_min = float(ret_min if ret_min is not None else p["ret_min"])
    contra_max = float(contra_max if contra_max is not None else p["contra_max"])
    out_ids = [str(x) for x in (evidence_ids or [])]
    ret_ids = [str(x) for x in (retrieved_ids or [])]
    if set(out_ids) - set(ret_ids):
        return FaithfulResult(False, "BAD_CITATION", [], ["citation"])
    if ret_score < ret_min:
        return FaithfulResult(False, "INSUFFICIENT_EVIDENCE", [], ["ret_score"])
    if contra > contra_max:
        return FaithfulResult(False, "INSUFFICIENT_EVIDENCE", [], ["contra"])
    claims = split_claims(body or "")
    tierv = [(c, claim_tier(c)) for c in claims]
    verifiable = [c for c, t in tierv if t == "Tier-V"]
    if not verifiable:
        return FaithfulResult(True, "TIER_B_EXEMPT", [ClaimResult(c, t, True, True, None, None, "exempt") for c, t in tierv])
    # every Tier-V claim needs >=1 citation
    if not out_ids:
        bad = [ClaimResult(c, "Tier-V", False, False, None, None, "uncited") for c in verifiable]
        return FaithfulResult(False, "UNCITED_TIERV", bad, ["citation"])
    # entity verbatim (ANY entity fail → False)
    ok_ent, viols = check_entities_verbatim(body or "", evidence_texts or [])
    if not ok_ent:
        bad = [ClaimResult(c, "Tier-V", True, False, None, None, f"entity:{viols[0]}") for c in verifiable]
        return FaithfulResult(False, f"ENTITY_GROUND_FAIL:{viols[0]}", bad, ["entity"])
    # NLI tie-break per verifiable claim (premise = concat cited evidence)
    premise = "\n".join(evidence_texts or [])
    try:
        if nli_entails is None or nli_contras is None:
            ent, con = nli_scores([premise] * len(verifiable), verifiable, _inject=_nli_inject)
        else:
            ent, con = list(nli_entails), list(nli_contras)
    except Exception as e:
        logger.warning("nli_skip %s: %s", type(e).__name__, str(e)[:100])
        bad = [ClaimResult(c, "Tier-V", True, True, None, None, "NLI_SKIP") for c in verifiable]
        return FaithfulResult(False, "INSUFFICIENT_EVIDENCE", bad, ["nli_skip"])
    results: list[ClaimResult] = []
    for c, e, k in zip(verifiable, ent, con):
        if k > contra_max:
            results.append(ClaimResult(c, "Tier-V", True, True, float(e), float(k), "contra"))
        elif float(e) < nli_min:
            results.append(ClaimResult(c, "Tier-V", True, True, float(e), float(k), "nli_fail"))
        else:
            results.append(ClaimResult(c, "Tier-V", True, True, float(e), float(k), "pass"))
    fails = [r for r in results if r.verdict != "pass"]
    if fails:
        v0 = fails[0].verdict
        reason = "INSUFFICIENT_EVIDENCE" if v0 in ("nli_fail", "contra") else f"NLI_FAIL:{v0}"
        return FaithfulResult(False, reason, results, ["nli" if v0 != "pass" else "ground"])
    for c, t in tierv:
        if t == "Tier-B":
            results.append(ClaimResult(c, t, True, True, None, None, "exempt"))
    return FaithfulResult(True, "ok", results)
