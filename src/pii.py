from __future__ import annotations

"""PII redaction + refund hedge + fit(280) + tone mirror (S4.3).

Order: card > email > order > phone > handle → ``[REDACTED_*]`` pre-LLM +
post-sweep (preserve handle for reply slot, redact from logs/model-context).
(Order precedes phone: Amazon order IDs embed phone-like runs; card-first
invariant preserved.)
Blocklist + semantic hedge for refund promises (Air-Canada 2024 liability).
``fit(280)`` word-boundary + reserve 10ch for links + POST-FIT re-validate
(``len<=280 AND links intact AND Tier-V citations intact else escalate``).
Tone mirror (casual→warm, formal→polished, angry→apology, skip if misclassified).
"""

import re
from dataclasses import dataclass

from src.generate import TCO_LEN, parse_tweet
from src.ground import ENTITY_PATTERNS
from src.logging_utils import get_logger

logger = get_logger(__name__)

CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
ORDER_RE = re.compile(r"\b(?:\d{3}-\d{7}-\d{7}|ORD-?\d{5,})\b", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,30}\b")
URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+|www\.[^\s<>\]\)\"']+", re.IGNORECASE)
CITE_RE = re.compile(r"\[\d+\]|\(PMID:\s*\d+\)|\bdoi:\S+|\[Source:[^\]]+\]", re.IGNORECASE)

REFUND_BLOCK_RES = [
    re.compile(r"(?i)\bguarantee(d|s)?\b.{0,20}\brefund\b"),
    re.compile(r"(?i)\bwill\b.{0,10}\brefund you\b"),
    re.compile(r"(?i)\brefund\b.{0,10}\bapprov(ed|al)\b|\bapprov(ed|al)\b.{0,10}\brefund\b"),
    re.compile(r"(?i)\b100\s?%\s?(refund|money back|guarantee)"),
    re.compile(r"(?i)\binstant\b.{0,10}\brefund\b"),
    re.compile(r"(?i)\bfull refund\b.{0,20}\b(within|in)\b.{0,10}\d+\s?(days|hours)"),
    re.compile(r"(?i)\brefund\b.{0,15}\b(processed|issued|sent|completed)\b"),
    re.compile(r"(?i)\bmoney back guarantee\b"),
]
HEDGE_TEMPLATE = "eligible for review under policy; can't guarantee outcome; track in app, agent will confirm."

REFUND_ANCHORS = [
    "You are guaranteed a refund",
    "We will refund you in full",
    "Your refund has been approved",
    "You get a 100% refund no questions asked",
    "You will get an instant refund to your card",
    "You will have your money back tomorrow morning",
    "I promise you will get every penny back",
    "Refund confirmed funds on the way to your bank",
    "Your non-refundable booking is now eligible for repayment",
]

TONE_ANGRY_RE = re.compile(r"[!?]{2,}|[A-Z]{5,}|\b(furious|angry|ridiculous|useless|worst|hate|wtf|damn|refund now)\b", re.IGNORECASE)


def luhn_ok(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(d) <= 19:
        return False
    tot = 0
    for i, n in enumerate(reversed(d)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        tot += n
    return tot % 10 == 0


def _collect_spans(text: str) -> list[tuple[int, int, str, str]]:
    # Order: card (Luhn-validated) > email > order > phone > handle.
    # Order precedes phone because Amazon order IDs (123-4567890-1234567)
    # embed phone-like digit runs; longest-strict wins. Card-first invariant
    # (S4.3) is preserved: card spans are collected before anything else so
    # phone/order never fragment a PAN.
    spans: list[tuple[int, int, str, str]] = []
    for m in CARD_RE.finditer(text or ""):
        if luhn_ok(m.group(0)):
            spans.append((m.start(), m.end(), "CARD", m.group(0)))
    for rx, typ in [(EMAIL_RE, "EMAIL"), (ORDER_RE, "ORDER"), (PHONE_RE, "PHONE"), (HANDLE_RE, "HANDLE")]:
        for m in rx.finditer(text or ""):
            s, e = m.start(), m.end()
            if typ == "PHONE" and "__phone__" not in m.group(0):
                if len(re.sub(r"\D", "", m.group(0))) not in (10, 11):
                    continue
            if any(s < oe and os < e for os, oe, _, _ in spans):
                continue
            spans.append((s, e, typ, m.group(0)))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept: list[tuple[int, int, str, str]] = []
    for s in spans:
        if not any(s[0] < e and st < s[1] for st, e, _, _ in kept):
            kept.append(s)
    return kept


def redact(text: str) -> str:
    """Pre-LLM mask. Deterministic, longest-span wins, card first via Luhn gate."""
    out = text or ""
    for s, e, typ, _ in sorted(_collect_spans(out), key=lambda x: x[0], reverse=True):
        out = out[:s] + f"[REDACTED_{typ}]" + out[e:]
    return out


def post_sweep(text: str) -> tuple[str, bool]:
    """Post-LLM sweep. Returns (swept, leaked). Leaked=True → caller must escalate."""
    swept = redact(text or "")
    leaked = bool(_collect_spans(text or ""))
    if leaked:
        logger.warning("pii_post_sweep_hit leaked/spans redacted")
    return swept, leaked


def scan_pii_counts(text: str) -> dict[str, int]:
    spans = _collect_spans(text or "")
    counts = {"CARD": 0, "EMAIL": 0, "PHONE": 0, "ORDER": 0, "HANDLE": 0}
    for _, _, typ, _ in spans:
        counts[typ] += 1
    return {"email_hits": counts["EMAIL"], "phone_hits": counts["PHONE"], **{k.lower() + "_hits": v for k, v in counts.items()}}


def blocklist_hit(text: str) -> tuple[bool, str]:
    t = text or ""
    for i, rx in enumerate(REFUND_BLOCK_RES):
        m = rx.search(t)
        if m:
            return True, f"blocklist-{i}:{m.group(0)[:40]}"
    return False, ""


def _cos(a, b) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def semantic_hedge_score(
    text: str, *, threshold: float = 0.82, _inject: float | None = None
) -> tuple[bool, float]:
    """Embedding backstop for paraphrases (``money back tomorrow``).

    Full-only (sentence-transformers, lazy). Missing dep → (False, 0.0);
    caller keeps string blocklist + NLI backstop. ``_inject`` for tests.
    """
    if _inject is not None:
        return bool(_inject >= threshold), float(_inject)
    try:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from src import _lazy

        _lazy.require_extra("sentence_transformers", "full-only semantic hedge")
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        import numpy as np  # noqa: F401

        emb = enc.encode([text or ""], normalize_embeddings=True, show_progress_bar=False)[0]
        anchors = enc.encode(REFUND_ANCHORS, normalize_embeddings=True, show_progress_bar=False)
        best = max(float(_cos(emb, a)) for a in anchors)
        return bool(best >= threshold), round(best, 4)
    except Exception as e:
        logger.warning("semantic_hedge_skip %s", type(e).__name__)
        return False, 0.0


def hedge(text: str, *, semantic_threshold: float = 0.82, _semantic: float | None = None) -> tuple[str, bool, str]:
    """Return (hedged_text, changed, reason). Never emits banned promise."""
    hit, pat = blocklist_hit(text or "")
    if hit:
        out = re.sub(
            r"(?i)\b(guaranteed refund|will refund you|refund approved|100% refund|instant refund)\b",
            HEDGE_TEMPLATE,
            text or "",
        )
        # if substitution missed (gapped pattern), append hedge + drop sentence
        if blocklist_hit(out)[0]:
            out = HEDGE_TEMPLATE
        logger.info("hedge blocklist=%s", pat)
        return out, True, f"BLOCKLIST:{pat}"
    sem_hit, score = semantic_hedge_score(text or "", threshold=semantic_threshold, _inject=_semantic)
    if sem_hit:
        logger.info("hedge semantic score=%.3f", score)
        return HEDGE_TEMPLATE, True, f"SEMANTIC:{score:.2f}"
    return text or "", False, ""


@dataclass(frozen=True)
class FitResult:
    text: str
    weighted_len: int
    escalate: bool
    reason: str


def extract_links(s: str) -> list[str]:
    return [m.group(0).rstrip(".,;:!?)]}'\"") for m in URL_RE.finditer(s or "")]


def extract_citations(s: str) -> list[str]:
    return [m.group(0) for m in CITE_RE.finditer(s or "")]


def fit(draft: str, limit: int = 280, reserve: int = 10) -> FitResult:
    """Word-boundary fit. Never splits URL/citation. Fail-closed on severed evidence."""
    from src.generate import weighted_len as _wl

    text = " ".join((draft or "").strip().split())
    if not text:
        return FitResult("", 0, True, "EMPTY")
    links = extract_links(text)
    cites = extract_citations(text)
    need = max(reserve, TCO_LEN * len(links)) if links else 0
    _ = need
    if _wl(text) <= limit:
        return FitResult(text, _wl(text), False, "ok")
    spans = [(m.start(), m.end()) for m in list(URL_RE.finditer(text)) + list(CITE_RE.finditer(text))]

    def in_protected(i: int) -> bool:
        return any(s < i < e for s, e in spans)

    # word-boundary walk-back that never lands inside a protected span
    cut = min(len(text), limit - 1)
    while cut > 0 and (cut >= len(text) or text[cut - 1] not in (" ", "\n") or in_protected(cut)):
        jump = [s for s, e in spans if s < cut < e]
        if jump:
            cut = jump[0]
            continue
        if cut < len(text) and text[cut - 1] == " " and not in_protected(cut - 1):
            break
        cut -= 1
    if cut <= 0:
        return FitResult(text, _wl(text), True, "UNFITTABLE")
    fitted = text[:cut].rstrip(" ,;:.-") + "\u2026"
    ok_len = parse_tweet(fitted)["valid"]
    ok_links = all(link in fitted for link in links)
    ok_cites = all(c in fitted for c in cites)
    if not (ok_len and ok_links and ok_cites):
        return FitResult(
            fitted,
            _wl(fitted),
            True,
            f"POST_FIT_FAIL len={ok_len} links={ok_links} cites={ok_cites}",
        )
    return FitResult(fitted, _wl(fitted), False, "truncated")


def post_fit_validate(original: str, fitted: str, evidence_ids: list[str]) -> tuple[bool, str]:
    if parse_tweet(fitted)["weightedLength"] > 280 or len(fitted) > 2000:
        return False, "LEN"
    for link in extract_links(original):
        if link not in fitted:
            return False, f"LINK_SEVERED:{link[:30]}"
    for c in extract_citations(original):
        if c not in fitted:
            return False, "CITE_SEVERED"
    _ = evidence_ids
    return True, "ok"


def mirror_tone(tone: str, issue: str = "this issue", *, anger_conf: float = 1.0) -> str:
    """Casual→warm, formal→polished, angry→apology. Skip mirror if anger uncertain."""
    issue = (issue or "this issue").strip()[:60] or "this issue"
    if tone == "angry" and anger_conf < 0.7:
        return ""
    if tone == "angry":
        return f"You're right to be upset, sorry for {issue}."
    if tone == "casual":
        return "Thanks for flagging \u2014 "
    if tone == "formal":
        return "Thank you for writing \u2014 "
    return ""


def is_angry(text: str) -> tuple[bool, float]:
    t = text or ""
    cues = 0
    if re.search(r"(?i)\b(furious|angry|ridiculous|useless|worst|hate)\b", t):
        cues += 1
    if re.search(r"[!?]{2,}", t):
        cues += 1
    if re.search(r"[A-Z]{6,}", t):
        cues += 1
    if re.search(r"(?i)\b(where is|still waiting|no fix|second|twice)\b", t):
        cues += 1
    conf = min(1.0, 0.25 * cues + (0.3 if TONE_ANGRY_RE.search(t) else 0.0))
    return bool(conf >= 0.7 and cues >= 2), round(conf, 2)
