from __future__ import annotations

"""Golden-set taxonomy + disclosed pseudo-stratifier (heur-v1.0.0).

The 200 golden labels are *pseudo-labels* produced deterministically from the
frozen pool alone (no human in the loop, no external data). The stratifier
name + version is logged in every manifest so a reviewer can tell proxy
signal from human truth. S2.2 adjudicates a disjoint 50-thread pilot;
humans may overwrite ``data-sample/golden.jsonl`` intents and re-run.

Design note: intent / sentiment / length / hardness run on INBOUND-ONLY
(customer) turns. Agent boilerplate (``cancel`` / ``courier`` /
``disappointment`` templates) flips ~29% of full-thread pseudo-labels, and
the production classifier only ever sees the customer message (same rule
as S3.4 q0 inbound-only). ``resolved`` still reads full turns (agent
close-outs + customer thanks). The golden ``text`` field therefore stores
inbound turns; full turns are recoverable via ``golden_thread_id``.
"""

import re

STRATIFIER_NAME = "heur-pseudo-stratifier"
STRATIFIER_VERSION = "heur-v1.2.0"

# Priority 1 -> 10. On multi-intent hits pick the LOWEST index (highest
# priority). This matches phases/02-golden-intent.md exactly.
TAXONOMY: list[str] = [
    "account-security",
    "cancel-modify",
    "returns-refunds",
    "payment-promo",
    "shipping-tracking",
    "order-status",
    "device-tech",
    "product-info",
    "complaint-feedback",
    "other",
]

INTENT_INDEX = {name: i for i, name in enumerate(TAXONOMY)}

# Conservative tweet-style patterns tuned for masked text
# (@USER, <URL>, ORD-12345, __email__, __phone__ preserved).
_PATTERNS: dict[str, str] = {
    "account-security": r"(password|passcode|otp|one[\s-]?time|2fa|two[\s-]?factor|verif\w* code|locked out|lockout|hacked|unauthori[sz]ed|phish|suspicious login|reset.*password|change.*password|account.*secur|login.*fail|sign[\s-]?in.*(problem|issue|error|fail)|(get|getting)\s+into\s+my\s+account|can'?t\s+(get\s+into|access|log\s*in|sign\s*in))",
    "cancel-modify": r"(cancel\w*[^.?!]{0,40}(order|subscription|prime|item|shipment|delivery|package|parcel)|cancel\s+(it|this|that|them|everything)|(modify|change|edit|update)[^.?!]{0,40}order|cancel.*asap|stop.*(order|shipment|delivery))",
    "returns-refunds": r"(return|refund|\brma\b|money back|chargeback|return label|return policy|(send|take|ship)\s+(it|them|this|that|these|those|my|the)\b[^.?!]{0,15}\bback\b|exchange|replacement.*(order|item)|refund.*(status|where|when|issu)|rembours|reembolso|erstattung|retoure)",
    "payment-promo": r"(payment|charged|charge|billing|invoice|promo(code)?|coupon|discount|gift[\s-]?card|declined|failed payment|overcharg|double charg|price.*(wrong|error|match)|pay[^.?!]{0,40}(issue|problem|fail))",
    "shipping-tracking": r"(track\w*|where.*(package|parcel|order|shipment)|shipping|delivery (status|update|track|late|delay|schedule)|carrier|ups|fedex|usps|dhl|dpd|hermes|amzl|amazon logistics|royal mail|lasership|shipped|in transit|out for delivery|arriving|estimated delivery|pick(ed)?\s+up|\bhub\b|circulat|\bstuck\b|(receiving|delivery|arrival)\s+date|(next day|one.day|1-day|2-day|two-day|same.day)|prime[^.?!]{0,25}deliver|deliver[^.?!]{0,25}prime|sendung|paket|lieferung|packstation|versand|colis|livraison|pedido|entrega|enviado|(package|parcel|packag\w*)[^.?!]{0,30}(deliver|arriv|track|ship|late|delay|missing|lost|damag|stolen|wrong|not))",
    "order-status": r"(order.*(status|where|when|update|pending|processing|confirm|placed|missing|not.*(arriv|receiv|deliver))|confirmation.*(email|number)|ORD-12345|order\s*(#|number|id)|no\s+update|without\s+update|(haven't|hasn't|havent|hasnt|never|not yet)[^.?!]{0,25}(receiv|arriv|got|gotten|came|deliver)|(purchase|bought|ordered)[^.?!]{0,30}(arriv|receiv|update|status|pending|missing|not|never)|(where|what|how)[^.?!]{0,25}(my\s+)?(order|packag\w*|parcel)|happen[^.?!]{0,25}(order|package|parcel))",
    "device-tech": r"(kindle|echo|alexa|fire(\s*(tv|stick|tablet))?|after the update|since[^.?!]{0,20}update|update (broke|breaks|broken)|device.*(not working|frozen|restart|update|setup|wifi|connect)|app.*(crash|not working|error|update)|\b(crash\w*|frozen|reboot|wifi|wi-?fi|bluetooth)\b|firmware|troubleshoot|factory reset)",
    "product-info": r"(product.*(info|detail|spec|question)|how\s+to\b|what.*(compatible|size|color|model|version)|\b(in stock|availability|available|dimensions|warranty|manual|guide|feature)\b|\b(watch|stream)\b|(where|what|how)[^.?!]{0,30}(game|match|episode|movie|show|series))",
    "complaint-feedback": r"(complaint|complain|compliant|terrible|awful|horrible|appaling|appalling|worst|disappoint|sucks|sucked|\bsuck\b|lame|useless|pathetic|shame|angry|furious|frustrat|ridiculous|unacceptable|\b(hell|damn|shit\w*|bullshit|screw)\b|never.*again|done with|goodbye amazon|waste.*(time|money))",
}

_COMPILED: dict[str, re.Pattern] = {k: re.compile(v, re.IGNORECASE) for k, v in _PATTERNS.items()}

_OTHER = "other"

# ---- sentiment lexicon (tiny, disclosed, no external model) ----
_POS = frozenset(
    "thanks thank great awesome perfect love excellent amazing wonderful glad happy pleased appreciate appreciated helpful fantastic nice good".split()
)
_NEG = frozenset(
    "angry terrible awful horrible worst hate disappointed disappointed frustrat sucks sucked suck stupid dumb lame useless pathetic shame furious ridiculous unacceptable broken wrong late missing lost damaged".split()
)

# ---- resolved proxy markers (agent close-outs + customer thanks) ----
_RESOLVED_AGENT = re.compile(
    r"(glad.*(help|resolved|sorted)|resolved|fixed|sorted for you|replacement.*(shipped|sent)|refund.*(issued|processed)|dm.*(resolved|sorted)|let us know if.*else|anything else.*help)",
    re.IGNORECASE,
)
_RESOLVED_THANKS = re.compile(r"(thank you|thanks.*(so much|a lot|for.*help)|great.*help|perfect.*(thanks|thank)|appreciated)", re.IGNORECASE)

# ---- hard-case markers: sarcasm / multi / low-margin ----
_SARCASM = re.compile(
    r"(yeah[\s,]+right|sure[\s.]*\.{2,}|great job|nice (going|work)|thanks? for nothing|lol.*(sucks|terrible|worst)|:-\)\s*$|wow.*(great|amazing).*(not|never)|/s\b|\?\?\?|!!!)",
    re.IGNORECASE,
)


def intent_hits(text: str) -> list[str]:
    """All matching intents in priority order (highest priority first)."""
    t = text or ""
    hits = [name for name in TAXONOMY if name != _OTHER and _COMPILED[name].search(t)]
    return hits


def pseudo_intent(text: str) -> tuple[str, list[str]]:
    """Primary pseudo-intent + all hits. Falls back to ``other``."""
    hits = intent_hits(text)
    if not hits:
        return _OTHER, []
    return hits[0], hits[1:]


def pseudo_sentiment(text: str) -> str:
    t = (text or "").lower()
    toks = re.findall(r"[a-z']+", t)
    if not toks:
        return "neutral"
    pos = sum(1 for w in toks if w in _POS or any(w.startswith(n.rstrip("t")) for n in ("frustra",)))
    neg = sum(1 for w in toks if w in _NEG or "frustrat" in w)
    # substring fallback for stems above
    pos += sum(1 for w in toks if w in ("thank", "thanks"))
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def len_bucket(text: str) -> str:
    n = len(text or "")
    if n < 140:
        return "short"
    if n <= 280:
        return "medium"
    return "long"


def pseudo_resolved(thread_texts: list[str]) -> bool:
    """Resolved proxy: agent close-out marker, or customer thanks in last turn."""
    if not thread_texts:
        return False
    joined_agent = " ".join(thread_texts[1:])
    if _RESOLVED_AGENT.search(joined_agent):
        return True
    last = thread_texts[-1] if thread_texts else ""
    return bool(_RESOLVED_THANKS.search(last))


def hard_score(text: str, n_hits: int) -> int:
    """Deterministic hardness score: sarcasm(2) + multi(2) + low-margin cues(1)."""
    s = 0
    if _SARCASM.search(text or ""):
        s += 2
    if n_hits >= 2:
        s += 2
    t = text or ""
    if ("?" in t and "!" in t) or t.count("?") >= 2:
        s += 1
    if len(t) > 280 and n_hits >= 1:
        s += 1
    return s


def stratify_record(thread_text: str, thread_turns: list[str]) -> dict:
    intent, second = pseudo_intent(thread_text)
    return {
        "intent": intent,
        "second_best": second[0] if second else "",
        "multi_flag": bool(second),
        "n_hits": 1 + len(second) if intent != _OTHER else 0,
        "sentiment": pseudo_sentiment(thread_text),
        "len_bucket": len_bucket(thread_text),
        "resolved": bool(pseudo_resolved(thread_turns)),
        "hard_score": int(hard_score(thread_text, 1 + len(second) if intent != _OTHER else 0)),
    }


def quota_plan(counts: dict[str, int], total: int, min_per: int = 12, cap: int = 30) -> dict[str, int]:
    """Proportional quota with min floor + head cap, largest-remainder fixup.

    ``counts`` maps intent -> pool size. Returns intent -> n summing to
    ``total``. Never a full-factorial floor: only the intent margin is
    quota-controlled; sent/len/resolved ride along proportionally and are
    reported, not floored.
    """
    intents = list(TAXONOMY)
    pool = sum(max(0, counts.get(k, 0)) for k in intents)
    if pool <= 0:
        raise ValueError("empty pool counts")
    raw = {k: total * counts.get(k, 0) / pool for k in intents}
    # floor + cap
    alloc = {k: max(min_per, min(cap, int(round(raw[k])))) for k in intents}
    # headroom: intents with zero pool get 0 (collapse handled by caller)
    for k in intents:
        if counts.get(k, 0) == 0:
            alloc[k] = 0
    diff = total - sum(alloc.values())
    # largest remainder on fractional parts, respecting caps and pool sizes
    order = sorted(intents, key=lambda k: (raw[k] - int(raw[k]), counts.get(k, 0)), reverse=(diff > 0))
    i = 0
    guard = 0
    while diff != 0 and guard < 10_000:
        guard += 1
        k = order[i % len(order)]
        i += 1
        if diff > 0 and alloc[k] < min(cap, counts.get(k, 0)):
            alloc[k] += 1
            diff -= 1
        elif diff < 0 and alloc[k] > 0:
            alloc[k] -= 1
            diff += 1
    if sum(alloc.values()) != total:
        raise AssertionError(f"quota does not sum to {total}: {alloc}")
    return alloc


def collapse_taxonomy(counts: dict[str, int], min_per: int = 12) -> dict[str, str]:
    """Return mapping intent -> collapsed label if any tail < min_per.

    Collapse rule (spec): ``order-status+shipping-tracking -> order-shipping``,
    ``complaint-feedback+other -> complaint-other`` + secondary complaint tag.
    Empty mapping means no collapse needed.
    """
    tails = [k for k in TAXONOMY if counts.get(k, 0) < min_per and counts.get(k, 0) > 0]
    missing = [k for k in TAXONOMY if counts.get(k, 0) == 0]
    if not tails and not missing:
        return {}
    mapping: dict[str, str] = {}
    if "order-status" in tails + missing or "shipping-tracking" in tails + missing:
        mapping["order-status"] = "order-shipping"
        mapping["shipping-tracking"] = "order-shipping"
    if "complaint-feedback" in tails + missing or "other" in tails + missing:
        mapping["complaint-feedback"] = "complaint-other"
        mapping["other"] = "complaint-other"
    return mapping
