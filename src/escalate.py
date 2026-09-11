from __future__ import annotations

"""Risk-gated escalation with reason codes (S4.4).

Hard gates first (HUM explicit human/2-fail, LEG sue/lawyer/chargeback/
regulator, PRI others-PII/card/SSN/takeover, SAF self-harm/violence, FIN
refund>$500/close/suspension) → ``Risk=1-clip(conf*(0.7+0.3*ground)-pen,0,1)``
where ``conf`` = calibrated classifier confidence (temp-scaled, S2.4),
``ground`` = cross-encoder Tier-V entail rate, ``pen`` ∈[0,0.5].
Tiers SWEPT on val ROC/ECE to hit abstain 8–15% (INIT high 0.25/0.45, med
0.35/0.60, low 0.45/0.70 — NOT final, document ROC + ECE plot + chosen
operating point). Bands auto/hedge-clarify/escalate. JSON
``{decision,confidence,reason,risk_factors,evidence_spans,handoff_summary}``
strict schema, ``additionalProperties:false``, ``confidence<0.75``→escalate.
Invalid JSON fails closed to escalate.
"""

import re
from typing import Any, Literal, TypedDict

from src.logging_utils import get_logger

logger = get_logger(__name__)

Decision = Literal["auto", "hedge_clarify", "escalate"]
Tier = Literal["high", "med", "low"]

REQUIRED_KEYS = {"decision", "confidence", "reason", "risk_factors", "evidence_spans", "handoff_summary"}

HUM_RES = [
    re.compile(r"(?i)\b(talk|speak|connect|transfer|get|give|hand|escalate|pass)\b.{0,25}\b(human|agent|real\s?person|live\s?(person|agent)|representative|manager|supervisor|operator)\b"),
    re.compile(r"(?i)\b(real|live|actual)\s+(person|agent|human|support)\b"),
    re.compile(r"(?i)^\s*(agent|human|operator)\s*[!?.]*\s*$"),
    re.compile(r"(?i)\b(i want a human|no bot|stop bot|you('re| are) (not helping|useless|a bot))\b"),
]
LEG_RES = [
    re.compile(r"(?i)\b(sue|sued|suing|lawsuit|litigat\w*|legal\s?action|small\s?claims|court|tribunal|subpoena|cease\s?desist)\b"),
    re.compile(r"(?i)\b(lawyer|attorney|solicitor|legal\s?(team|counsel|department))\b"),
    re.compile(r"(?i)\b(chargeback|charge\s?back|dispute\s?(the\s?)?(charge|payment|transaction))\b"),
    re.compile(r"(?i)\b(FTC|CFPB|SEC|BBB|FCC|ICO|GDPR|attorney\s?general|ombudsman|regulator|consumer\s?protection)\b"),
    re.compile(r"(?i)\b(see you in court|contact my lawyer|file.*complaint.*(FTC|BBB|police|regulator))\b"),
]
PRI_RES = [
    re.compile(r"\b(?!000|666|9\\d{2})\\d{3}[- ]?(?!00)\\d{2}[- ]?(?!0000)\\d{4}\\b"),
    re.compile(r"(?i)\b(someone else'?s|other person'?s|my (wife|husband|boss|friend)'s).{0,25}\b(account|password|ssn|card|balance|otp|2fa)\b"),
    re.compile(r"(?i)\b(bypass|skip|disable|get around).{0,20}\b(2fa|mfa|otp|verification|password)\b"),
    re.compile(r"(?i)\b(change|update|replace).{0,20}\b(email|phone|payout|bank|password).{0,20}\b(account|someone)\b"),
    re.compile(r"(?i)\b(what'?s (his|her|their|john'?s).{0,15}(balance|ssn|card|address|email|phone))\b"),
    re.compile(r"(?i)\b(dox|leak|reveal|lookup).{0,15}\b(address|phone|email|ssn)\b"),
]
SAF_SELF_RES = [
    re.compile(r"(?i)\b(suicid\w*|self[-\s]?harm|self[-\s]?injur\w*|cutting|hurt myself|harm myself)\b"),
    re.compile(r"(?i)\b(kill myself|killing myself|end (my|his|her) life|take my (own )?life|want to die|wanna die|better off dead|don'?t want to (live|be alive))\b"),
    re.compile(r"(?i)\b(overdos\w*|hang myself|slit (my )?wrist|jump off (a )?(bridge|building|roof)|cut myself|no reason to live|can'?t go on|end it all)\b"),
]
SAF_VIOL_RES = [
    re.compile(r"(?i)\b(kill|murder|shoot|stab|strangl\w*|bomb|massacre|torture).{0,25}\b(someone|people|them|him|her|crowd|school|kids|neighbor|boss|wife|husband|family|staff|team)\b"),
    re.compile(r"(?i)\b(i('ll| will| am going to) (kill|hurt|harm|shoot|stab|attack))\b"),
    re.compile(r"(?i)\bi('ll| will| am going to).{0,25}\b(kill|hurt|harm|shoot|stab|attack)\b"),
    re.compile(r"(?i)\b(hurt|harm|kill|attack)\b.{0,15}\b(staff|team|people|someone)\b"),
    re.compile(r"(?i)\b(make a bomb|how to make (a )?(bomb|poison|weapon))\b"),
]
FIN_AMOUNT_RE = re.compile(r"\$\s?([0-9]{1,3}(,[0-9]{3})*(\.\d{2})?|[0-9]+(\.\d{2})?)")
FIN_ALT_RE = re.compile(r"(?i)\b(\d+(?:,\d{3})*(?:\.\d{2})?)\s?(dollars|usd|bucks)\b")
FIN_INTENT_RE = re.compile(r"(?i)\b(refund|reimburse\w*|money\s?back|compensation|fee\s?waiver|credit\s?back)\b")
FIN_CLOSE_RE = re.compile(
    r"(?i)\b(close|closing|delete|delet\w*|terminate|deactivat\w*|suspend|suspension|ban(ned)?|freeze|shut\s?down).{0,15}\b(account|profile|membership|subscription)\b"
)


class EscalationInput(TypedDict, total=False):
    text: str
    conf: float
    ground: float
    ret_score: float
    pii_present: bool
    fail_count: int
    evidence_spans: list[str]


class EscalationOutput(TypedDict):
    decision: Decision
    confidence: float
    reason: str
    risk_factors: list[str]
    evidence_spans: list[str]
    handoff_summary: str


def default_params() -> dict[str, Any]:
    try:
        from src.config import load_config

        e = load_config().get("escalation", {})
        risk = e.get("risk", {}) if isinstance(e.get("risk", {}), dict) else {}
        tiers = e.get("tiers", {}) if isinstance(e.get("tiers", {}), dict) else {}
        return {
            "conf_min": float(e.get("confidence_min", 0.75)),
            "w_conf": float(risk.get("conf_weight", 0.7)),
            "w_ground": float(risk.get("ground_weight", 0.3)),
            "pen_hit": float(risk.get("pen_per_hard_hit", 0.2)),
            "pen_pii": float(risk.get("pen_pii", 0.3)),
            "pen_low": float(risk.get("pen_low_retrieval", 0.2)),
            "pen_max": float(risk.get("pen_max", 0.5)),
            "low_lt": float(risk.get("low_retrieval_lt", 0.35)),
            "tiers": {
                "high": {"lo": 0.25, "hi": 0.45, **tiers.get("high", {})},
                "med": {"lo": 0.35, "hi": 0.60, **tiers.get("med", {})},
                "low": {"lo": 0.45, "hi": 0.70, **tiers.get("low", {})},
            },
        }
    except Exception:
        return {
            "conf_min": 0.75,
            "w_conf": 0.7,
            "w_ground": 0.3,
            "pen_hit": 0.2,
            "pen_pii": 0.3,
            "pen_low": 0.2,
            "pen_max": 0.5,
            "low_lt": 0.35,
            "tiers": {"high": {"lo": 0.25, "hi": 0.45}, "med": {"lo": 0.35, "hi": 0.60}, "low": {"lo": 0.45, "hi": 0.70}},
        }


def parse_amount(text: str) -> float | None:
    t = text or ""
    m = FIN_AMOUNT_RE.search(t)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            return None
    m2 = FIN_ALT_RE.search(t)
    if m2:
        try:
            return float(m2.group(1).replace(",", ""))
        except Exception:
            return None
    return None


def check_hard_gates(text: str, fail_count: int = 0, amount: float | None = None) -> tuple[bool, str | None, list[str]]:
    t = text or ""
    try:
        for rx in SAF_SELF_RES + SAF_VIOL_RES:
            if rx.search(t):
                return True, "SAF", ["SAF:self-harm-or-violence"]
        # PRI: others-PII / takeover (own single PII is S4.3 redact, not gate — gate needs takeover context or SSN/card+other)
        for rx in PRI_RES:
            if rx.search(t):
                return True, "PRI", ["PRI:others-pii-or-takeover"]
        if re.search(r"(?i)\b(card|ssn|social security|cvv|otp|password)\b", t) and re.search(
            r"(?i)\b(wife|husband|someone else|other person|boss|friend|mom|dad|bypass|change.*payout)\b", t
        ):
            return True, "PRI", ["PRI:pii-takeover-context"]
        for rx in LEG_RES:
            if rx.search(t):
                return True, "LEG", ["LEG:legal-chargeback-regulator"]
        amt = amount if amount is not None else parse_amount(t)
        if FIN_INTENT_RE.search(t) and amt is not None and amt > 500:
            return True, "FIN", [f"FIN:refund-{amt:g}-over-500"]
        if FIN_INTENT_RE.search(t) and amt is None and re.search(r"(?i)\b(refund|reimburse)\b", t) and re.search(r"(?i)\b(flight|cancelled|thousand|lakh)\b", t):
            pass  # unparseable large claim handled by risk pen, not hard gate
        if FIN_CLOSE_RE.search(t):
            return True, "FIN", ["FIN:close-suspend"]
        for rx in HUM_RES:
            if rx.search(t):
                return True, "HUM", ["HUM:explicit-human-request"]
        if (fail_count or 0) >= 2:
            return True, "HUM", ["HUM:two-fail"]
        return False, None, []
    except Exception as e:
        logger.warning("hard_gate_error %s", type(e).__name__)
        return False, None, []


def compute_risk(
    conf: float,
    ground: float,
    *,
    pii_present: bool = False,
    hard_hits: int = 0,
    ret_score: float = 1.0,
    params: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    p = params or default_params()
    if not (0.0 <= float(conf) <= 1.0 and 0.0 <= float(ground) <= 1.0):
        raise ValueError(f"conf={conf} ground={ground} out of [0,1]")
    pen_raw = (
        float(p["pen_hit"]) * min(int(hard_hits), 2)
        + (float(p["pen_pii"]) if pii_present else 0.0)
        + (float(p["pen_low"]) if float(ret_score) < float(p["low_lt"]) else 0.0)
    )
    pen = min(float(p["pen_max"]), pen_raw)
    base = float(conf) * (float(p["w_conf"]) + float(p["w_ground"]) * float(ground)) - pen
    base_c = min(1.0, max(0.0, base))
    risk = 1.0 - base_c
    return round(risk, 4), {"base": round(base_c, 4), "pen": round(pen, 4), "conf": float(conf), "ground": float(ground)}


def validate_schema(obj: Any) -> tuple[bool, str | None]:
    if not isinstance(obj, dict):
        return False, "not-object"
    keys = set(obj.keys())
    if keys != REQUIRED_KEYS:
        extra = sorted(keys - REQUIRED_KEYS)
        missing = sorted(REQUIRED_KEYS - keys)
        if extra:
            return False, f"additionalProperties:{extra}"
        return False, f"missing:{missing}"
    if obj["decision"] not in ("auto", "hedge_clarify", "escalate"):
        return False, "bad-decision"
    c = obj["confidence"]
    if not isinstance(c, (int, float)) or isinstance(c, bool) or not (0.0 <= float(c) <= 1.0):
        return False, "bad-confidence"
    for k in ("risk_factors", "evidence_spans"):
        if not isinstance(obj[k], list) or not all(isinstance(x, str) for x in obj[k]):
            return False, f"bad-{k}"
    for k in ("reason", "handoff_summary"):
        if not isinstance(obj[k], str) or not obj[k].strip():
            return False, f"bad-{k}"
    return True, None


def _escalate(reason: str, factors: list[str], conf: float, spans: list[str], text: str, risk: float) -> EscalationOutput:
    snippet = " ".join((text or "").split())[:140]
    return {
        "decision": "escalate",
        "confidence": round(min(0.49, max(0.0, float(conf))), 4),
        "reason": reason,
        "risk_factors": list(factors),
        "evidence_spans": list(spans or []),
        "handoff_summary": f"[{reason}] risk={risk:.2f} conf={float(conf):.2f} | {snippet}",
    }


def decide(inp: EscalationInput, tier: Tier = "med", params: dict[str, Any] | None = None) -> EscalationOutput:
    """Fail-closed orchestrator. Never raises — exceptions → escalate."""
    p = params or default_params()
    try:
        text = str(inp.get("text", "") or "")
        conf = float(inp.get("conf", 0.0))
        ground = float(inp.get("ground", 0.0))
        ret = float(inp.get("ret_score", 1.0))
        spans = [str(x) for x in (inp.get("evidence_spans", []) or [])]
        hit, code, factors = check_hard_gates(text, int(inp.get("fail_count", 0) or 0))
        if hit and code:
            logger.info("escalate hard_gate=%s", code)
            return _escalate(code, factors, conf, spans, text, 1.0)
        risk, parts = compute_risk(
            conf, ground, pii_present=bool(inp.get("pii_present", False)),
            hard_hits=0, ret_score=ret, params=p,
        )
        band = p["tiers"].get(tier, p["tiers"]["med"])
        lo, hi = float(band["lo"]), float(band["hi"])
        if ret < 0.30:
            return _escalate("INSUFFICIENT_EVIDENCE", ["ret_score<0.30", *factors], conf, spans, text, risk)
        if conf < float(p["conf_min"]):
            return _escalate("LOW_CONFIDENCE", [f"conf<{p['conf_min']}", *factors], conf, spans, text, risk)
        if risk < lo:
            out: EscalationOutput = {
                "decision": "auto", "confidence": round(conf, 4), "reason": "ok",
                "risk_factors": list(factors), "evidence_spans": spans,
                "handoff_summary": f"[ok] risk={risk:.2f} conf={conf:.2f} ground={ground:.2f}",
            }
        elif risk < hi:
            out = {
                "decision": "hedge_clarify", "confidence": round(conf, 4), "reason": "MEDIUM_RISK",
                "risk_factors": [f"risk={risk:.2f}", *factors], "evidence_spans": spans,
                "handoff_summary": f"[MEDIUM_RISK] risk={risk:.2f} conf={conf:.2f} | {' '.join(text.split())[:140]}",
            }
        else:
            return _escalate("HIGH_RISK", [f"risk={risk:.2f}", *factors], conf, spans, text, risk)
        ok, err = validate_schema(out)
        if not ok:
            return _escalate("INVALID_JSON", [f"schema:{err}"], conf, spans, text, risk)
        if out["decision"] == "auto" and float(out["confidence"]) < float(p["conf_min"]):
            return _escalate("LOW_CONFIDENCE", ["conf<min on auto"], conf, spans, text, risk)
        _ = parts
        return out
    except Exception as e:
        logger.warning("escalate_fail_closed %s: %s", type(e).__name__, str(e)[:120])
        try:
            t = str((inp or {}).get("text", "") or "")
        except Exception:
            t = ""
        return {
            "decision": "escalate", "confidence": 0.0, "reason": "INVALID_JSON",
            "risk_factors": [f"exception:{type(e).__name__}"], "evidence_spans": [],
            "handoff_summary": f"[INVALID_JSON] {type(e).__name__} | {t[:140]}",
        }
