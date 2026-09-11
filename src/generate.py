from __future__ import annotations

"""Grounded reply generation (S4.1).

Pipeline: ``build_prompt (pure) -> generate_draft (I/O boundary, DI client)
-> validate_output (pure, fail-closed to escalate)``.

Rules (Phase 04 S4.1):
- template vars ``{tone,lang,intent,inbound_clean,thread_context,E1..E3}``
  (threads break without prior-turn context).
- ``BODY`` field only ``<=240ch`` (not whole JSON), 1 action, reuse
  links/phones verbatim never invent.
- internal ``evidence_ids/spans`` in JSON ONLY — user ``body`` NEVER shows
  ``[D#]`` (breaks AmazonHelp voice; user gets verbatim help-link or nothing).
- missing-answer template when evidence insufficient.
- gen ``gemini-2.5-flash`` temp 0.2 ``response_mime_type: application/json``.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol

from src import _lazy
from src.logging_utils import get_logger
from src.seed import seed_everything

logger = get_logger(__name__)

PROMPT_VERSION = "s4.1-v1.0.0"
MISSING_ANSWER = "Sorry, can't confirm from our docs \u2014 please DM @Support with details."
BODY_MAX = 240
FIT_MAX = 280
FIT_RESERVE = 10
TCO_LEN = 23

LEAK_RE = re.compile(r"\[[DB]\d+\]|\(Doc\s*\d+\)|\bdoc_id\b|\bchunk_id\b|\bevidence_ids\b|\bevidence_spans\b", re.IGNORECASE)
IMPERATIVE_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:please\s+)?(?:dm|send|share|provide|confirm|reply|check|track|visit|click|open|go to|try|restart|update|enter)\b",
    re.IGNORECASE,
)
_CLAUSE_LEAD_RE = re.compile(
    r"^(?:please\s+|kindly\s+)?(?:dm|send|share|provide|confirm|reply|check|track|visit|click|open|go to|try|restart|update|enter)\b",
    re.IGNORECASE,
)
JOIN_RE = re.compile(r"(?i)\b(and also|then also|or alternatively|either\s+.*\s+or)\b")
URL_RE = re.compile(r"https?://[^\s<>\[\]{}()\"']+", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?-'\"\u201d')]}>"
EMOJI_RE = re.compile(
    r"(?:[\U0001F1E6-\U0001F1FF]{2}"
    r"|[#\*0-9]\uFE0F?\u20E3"
    r"|(?:[\u2600-\u27BF\U0001F000-\U0001FAFF\U0001F3FB-\U0001F3FF\u2B00-\u2BFF\uFE0F]"
    r"(?:\u200D[\u2600-\u27BF\U0001F000-\U0001FAFF\U0001F3FB-\U0001F3FF\uFE0F]+)*))"
)
WEIGHT_1_RANGES = [(0, 4351), (8192, 8205), (8208, 8223), (8242, 8247)]

_llm_client = None


@dataclass(frozen=True)
class GenerateConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    body_max: int = BODY_MAX
    fit_max: int = FIT_MAX
    fit_reserve: int = FIT_RESERVE
    seed: int = 42


@dataclass(frozen=True)
class PromptContext:
    tone: str
    lang: str
    intent: str
    inbound_clean: str
    thread_context: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str
    risk_factors: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    def generate_json(self, prompt: str, *, temperature: float = 0.2) -> dict[str, Any]: ...


def default_generate_params() -> dict[str, Any]:
    try:
        from src.config import load_config

        g = load_config().get("generation", {})
        t = g.get("thresholds", {}) if isinstance(g.get("thresholds", {}), dict) else {}
        return {
            "model": str(g.get("model", "gemini-2.5-flash")),
            "temperature": float(g.get("temp", 0.2)),
            "body_max": int(t.get("body_max", 240)),
            "fit_max": int(g.get("fit_max_chars", 280)),
            "prompt_version": str(g.get("prompt_version", PROMPT_VERSION)),
        }
    except Exception:
        return {
            "model": "gemini-2.5-flash",
            "temperature": 0.2,
            "body_max": 240,
            "fit_max": 280,
            "prompt_version": PROMPT_VERSION,
        }


def get_llm_client(model: str | None = None):
    """Lazy Gemini client. Never imported at module top (repro stays torch-free)."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _lazy.require_extra(
        "google.generativeai", "pip install google-generativeai or use offline fallback (no key)"
    )
    import google.generativeai as genai  # type: ignore[import-not-found]

    _llm_client = genai.GenerativeModel(model or default_generate_params()["model"])
    return _llm_client


def offline_fallback(ctx: PromptContext, reason: str = "LLM_UNAVAILABLE") -> dict[str, Any]:
    logger.warning("generate_offline reason=%s intent=%s", reason, ctx.intent)
    return {
        "body": MISSING_ANSWER,
        "evidence_ids": [],
        "evidence_spans": [],
        "handoff_summary": f"offline fallback ({reason}); intent={ctx.intent}",
    }


def build_prompt(ctx: PromptContext, cfg: GenerateConfig | None = None) -> str:
    cfg = cfg or GenerateConfig()
    e = list(ctx.evidence or []) + [{}, {}, {}]
    e1 = str((e[0] or {}).get("text", ""))
    e2 = str((e[1] or {}).get("text", ""))
    e3 = str((e[2] or {}).get("text", ""))
    return (
        "You are AmazonHelp support. Reply in {tone} in {lang} for intent {intent}.\n"
        "Customer last turn: {inbound}\n"
        "Thread context (prior turns, resolve pronouns, do not repeat answered steps):\n"
        "{thread}\n"
        "Evidence (verbatim only, do not invent):\n"
        "E1:{e1}\nE2:{e2}\nE3:{e3}\n"
        "Output STRICT JSON ONLY: {{\"body\": str, \"evidence_ids\": [], \"evidence_spans\": [], \"handoff_summary\": str|null}}\n"
        "Rules:\n"
        "- body: ONLY user-visible reply, <={bmax} chars, 1-2 sentences, empathy clause + EXACTLY ONE action.\n"
        "- ONE action = exactly one imperative: DM with details OR click verbatim help-link OR share order# OR try one fix. "
        "Never give 2+ options in one reply (no 'and also / or alternatively / either..or').\n"
        "- Reuse links/phones verbatim from E1..E3. Never invent. No [D#], [1], (Doc 1), doc_id, chunk_id in body. "
        "evidence_ids/spans JSON-only, must be subset of retrieved IDs.\n"
        "- If no Tier-V evidence for link/phone/refund/amount/date/order: body MUST be exactly: {missing}\n"
        "- If Tier-V claim present it must be entailed by cited evidence; else use missing template with evidence_ids=[]."
    ).format(
        tone=ctx.tone,
        lang=ctx.lang,
        intent=ctx.intent,
        inbound=ctx.inbound_clean,
        thread=e[0] and ctx.thread_context or ctx.thread_context,
        e1=e1,
        e2=e2,
        e3=e3,
        bmax=(cfg.body_max if cfg else BODY_MAX),
        missing=MISSING_ANSWER,
    )


def generate_draft(
    ctx: PromptContext,
    *,
    client: LLMClient | None = None,
    cfg: GenerateConfig | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Impure boundary. ``client`` injectable for tests; None -> lazy Gemini or offline fallback."""
    seed_everything(seed)
    cfg = cfg or GenerateConfig()
    prompt = build_prompt(ctx, cfg)
    if client is not None:
        try:
            out = client.generate_json(prompt, temperature=cfg.temperature)
            logger.info("draft via=injected body_len=%d", len(str(out.get("body", ""))))
            return out
        except Exception as e:
            logger.warning("draft_injected_fail %s: %s", type(e).__name__, str(e)[:120])
            return offline_fallback(ctx, reason=f"INJECTED_FAIL:{type(e).__name__}")
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        return offline_fallback(ctx, reason="NO_API_KEY")
    try:
        llm = get_llm_client(cfg.model)
        import json

        resp = llm.generate_content(
            prompt,
            generation_config={
                "temperature": cfg.temperature,
                "response_mime_type": "application/json",
                "max_output_tokens": 500,
            },
        )
        raw = (getattr(resp, "text", "") or "").strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        out = json.loads(raw)
        if not isinstance(out, dict):
            raise ValueError("root-not-object")
        logger.info("draft via=gemini body_len=%d", len(str(out.get("body", ""))))
        return out
    except Exception as e:
        logger.warning("draft_gemini_fail %s: %s", type(e).__name__, str(e)[:120])
        fb = offline_fallback(ctx, reason=f"LLM_FAIL:{type(e).__name__}")
        return fb


def _char_weight(cp: int) -> int:
    for s, e in WEIGHT_1_RANGES:
        if s <= cp <= e:
            return 1
    return 2


def _find_urls(s: str) -> list[tuple[int, int]]:
    spans = []
    for m in URL_RE.finditer(s):
        st, en = m.start(), m.end()
        while en > st and s[en - 1] in _TRAILING_PUNCT:
            en -= 1
        if en - st > 8 and "." in s[st:en]:
            spans.append((st, en))
    return spans


def parse_tweet(text: str) -> dict[str, Any]:
    s = unicodedata.normalize("NFC", text or "")
    if not s.strip():
        return {"weightedLength": 0, "valid": False, "permillage": 0}
    url_spans = _find_urls(s)
    url_map = {st: en for st, en in url_spans}
    emap: dict[int, int] = {}
    for m in EMOJI_RE.finditer(s):
        st, en = m.start(), m.end()
        if st == en:
            continue
        if any(us <= st < ue for us, ue in url_spans):
            continue
        if st not in emap or en > emap[st]:
            emap[st] = en
    weighted = 0
    i, n = 0, len(s)
    while i < n:
        if i in url_map:
            weighted += TCO_LEN
            i = url_map[i]
            continue
        if i in emap:
            weighted += 2
            i = emap[i]
            continue
        weighted += _char_weight(ord(s[i]))
        i += 1
    valid = 0 < weighted <= FIT_MAX
    return {"weightedLength": weighted, "valid": valid, "permillage": (weighted * 1000) // FIT_MAX}


def weighted_len(text: str) -> int:
    return int(parse_tweet(text)["weightedLength"])


def count_imperatives(body: str) -> int:
    text = (body or "").strip()
    if not text:
        return 0
    # strip leading empathy prefix (not counted as action)
    text = re.sub(
        r"^(?:thanks for flagging\s*\u2014\s*|thank you for writing\s*\u2014\s*|you're right to be upset, sorry for [^.]+\.\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip() or text
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    c = sum(1 for s in sents if IMPERATIVE_RE.search(s))
    # comma/and/em-dash-separated imperatives inside one sentence: "DM x, check y, and try z"
    clauses: list[str] = []
    for s in sents:
        clauses.extend([x.strip() for x in re.split(r"[,;]|\s+and\s+|\s+or\s+|\s+then\s+|\u2014|\u2013", s) if x.strip()])
    lead = 0
    for cl in clauses:
        stripped = re.sub(r"^(?:and|or|then|also)\b\s*", "", cl, flags=re.IGNORECASE).strip()
        if _CLAUSE_LEAD_RE.search(stripped):
            lead += 1
    n = max(c, lead)
    n += len(JOIN_RE.findall(text))
    if n == 0 and "?" in text:
        n = 1  # question-ask counts as the single action
    return n


def check_single_action(body: str) -> tuple[bool, int, str]:
    n = count_imperatives(body or "")
    if n == 1:
        return True, n, "ok"
    if n == 0:
        return False, n, "NO_ACTION"
    return False, n, "MULTI_ACTION"


def detect_tone(inbound: str) -> str:
    t = inbound or ""
    if re.search(r"(?i)\b(dear|sir|madam|kindly|would you|could you)\b", t) and not re.search(
        r"(?i)\b(hey|lol|thx|thanks!|omg)\b", t
    ):
        return "formal"
    if re.search(r"[!?]{2,}|[A-Z]{5,}", t) or re.search(
        r"\b(furious|angry|ridiculous|useless|worst|hate|wtf|damn)\b", t, re.IGNORECASE
    ):
        return "angry"
    if re.search(r"(?i)\b(hey|hi+|lol|omg|thx|thanks|btw|plz)\b", t):
        return "casual"
    return "neutral"


def mirror_prefix(tone: str, issue: str = "this") -> str:
    issue = (issue or "this").strip()[:60] or "this"
    if tone == "angry":
        return f"You're right to be upset, sorry for {issue}."
    if tone == "casual":
        return "Thanks for flagging \u2014 "
    if tone == "formal":
        return "Thank you for writing \u2014 "
    return ""


def validate_output(
    draft: Any, retrieved_ids: set[str], *, body_max: int = BODY_MAX
) -> ValidationResult:
    """Pure fail-closed shape check for S4.1 (grounding lives in ground.py)."""
    if not isinstance(draft, dict):
        return ValidationResult(False, "NOT_OBJECT", ["shape"])
    body = draft.get("body", "")
    ev_ids = draft.get("evidence_ids", [])
    if not isinstance(body, str) or not body.strip():
        return ValidationResult(False, "EMPTY_BODY", ["shape"])
    if len(body) > body_max:
        return ValidationResult(False, "OVER_240", ["length"])
    if LEAK_RE.search(body):
        return ValidationResult(False, "CITATION_LEAK", ["leak"])
    if not isinstance(ev_ids, list) or not all(isinstance(x, str) for x in ev_ids):
        return ValidationResult(False, "BAD_EVIDENCE_IDS", ["shape"])
    if set(ev_ids) - set(retrieved_ids):
        return ValidationResult(False, "BAD_CITATION", ["citation"])
    return ValidationResult(True, "ok")


def build_thread_context(
    prior_turns: list[dict[str, Any]], max_chars: int = 1800
) -> tuple[str, str, str]:
    """Pure helper: prior turns [{speaker,inbound,text}] (pos order) -> (context, facts, note).

    Keeps newest Tier-V turns first under budget, restores chronological order.
    """
    case_re = re.compile(
        r"ORD-12345|<URL>|__phone__|__email__|@USER|\$|\u00a3|refund|return|\brma\b|track|order\s*(#|number|id)?",
        re.IGNORECASE,
    )
    lines = []
    for r in prior_turns or []:
        sp = str(r.get("speaker") or ("Customer" if r.get("inbound") else "Agent"))
        tx = str(r.get("text", "")).strip()
        if tx:
            lines.append(f"{sp}: {tx}")
    ents = sorted(set(re.findall(r"ORD-12345|<URL>|__phone__|__email__|@USER", " ".join(lines))))
    facts = f"CASE_FACTS: {' ; '.join(ents) if ents else 'none'}"
    budget = max(0, max_chars - len(facts) - 1)

    def vital(s: str) -> bool:
        return bool(case_re.search(s))

    newest = list(reversed(lines))
    picked: list[str] = []
    used = 0
    for pool in ([s for s in newest if vital(s)], [s for s in newest if not vital(s)]):
        for s in pool:
            cost = len(s) + 1
            if used + cost <= budget:
                picked.append(s)
                used += cost
    rank = {s: i for i, s in enumerate(lines)}
    picked.sort(key=lambda s: rank.get(s, 0))
    ctx = "\n".join(picked) if picked else "(no prior turns)"
    return (f"{facts}\n{ctx}" if picked else facts, facts, "newest-kept")
