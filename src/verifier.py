from __future__ import annotations

"""Dissimilar verifier for high risk (S4.5).

Cross-family second model (``gpt-oss-20b`` strict checks Gemini draft —
pinned verifier prompt/version: citation-subset + entity-verbatim +
refund-policy-entailment checks ONLY, no vague ``strict``) on
high-tier/middle-band only. ``VERIFIER_REJECT`` → escalate/hedge. Log
verifier gain separately (self-check is ~0 gain, never use). 20B may
over-reject → monitor abstain inflation.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from src import _lazy
from src.ground import check_entities_verbatim
from src.logging_utils import get_logger

logger = get_logger(__name__)

VERIFIER_MODEL = "openai/gpt-oss-20b"
VERIFIER_PROMPT_VERSION = "verifier-v1.2.0-gptoss20b-low-20260911"
VERIFIER_REASONING = "low"


@dataclass(frozen=True)
class VerifierResult:
    verdict: str  # VERIFIER_ACCEPT | VERIFIER_REJECT | SKIPPED
    failed_check: str  # none | citation | entity | entailment | error
    reason: str
    latency_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


def default_verifier_params() -> dict[str, Any]:
    try:
        from src.config import load_config

        v = load_config().get("generation", {}).get("verifier", {})
        if not isinstance(v, dict):
            v = {}
        return {
            "model": str(v.get("model", VERIFIER_MODEL)),
            "prompt_version": str(v.get("prompt_version", VERIFIER_PROMPT_VERSION)),
            "scope": str(v.get("scope", "high-tier-only")),
        }
    except Exception:
        return {"model": VERIFIER_MODEL, "prompt_version": VERIFIER_PROMPT_VERSION, "scope": "high-tier-only"}


def verifier_prompt(question: str, sources: list[dict[str, str]], draft: dict[str, Any]) -> str:
    src_lines = "\n".join(f"[{s.get('id', f'D{i+1}')}] {s.get('text', '')}" for i, s in enumerate(sources or []))
    return (
        "You are VERIFIER-v1.2.0, a strict skeptical checker. Check a Gemini draft against PROVIDED SOURCES ONLY. "
        "No web, no prior knowledge, no rewriting, no fluency judgment. Output JSON ONLY "
        '{"verdict":"VERIFIER_ACCEPT|VERIFIER_REJECT","failed_check":"none|citation|entity|entailment","reason":str}.\n'
        "PERFORM EXACTLY 3 CHECKS, FAIL-CLOSED:\n"
        "1) CITATION-SUBSET: every citation in DRAFT.citations MUST be subset of SOURCES ids; every factual sentence "
        "with amount/date/eligibility/deadline MUST have >=1 citation.\n"
        "2) ENTITY-VERBATIM: MONEY/DATES/POLICY-NAMES must appear verbatim in a cited SOURCE.text. "
        '"30 days" != "one month"; dropped negation = FAIL.\n'
        "3) REFUND-POLICY-ENTAILMENT: for each claim H with premise P=concat(cited SOURCE.texts): ENTAILED pass, "
        "NEUTRAL=FAIL, CONTRADICTION=HARD FAIL.\n"
        f"QUESTION: {question}\nSOURCES:\n{src_lines}\nDRAFT: {draft}\n"
    )


def _local_checks(
    draft: dict[str, Any], sources: list[dict[str, str]], retrieved_ids: list[str]
) -> VerifierResult:
    cites = [str(x) for x in (draft.get("evidence_ids", draft.get("citations", [])) or [])]
    if set(cites) - set(retrieved_ids):
        return VerifierResult("VERIFIER_REJECT", "citation", "citation-subset fail")
    body = str(draft.get("body", draft.get("answer_text", "")) or "")
    texts = [str(s.get("text", "")) for s in (sources or [])]
    ok, viols = check_entities_verbatim(body, texts)
    if not ok:
        return VerifierResult("VERIFIER_REJECT", "entity", f"entity-verbatim fail:{viols[0]}")
    return VerifierResult("VERIFIER_ACCEPT", "none", "local 2/3 pass; entailment needs model")


def verify_draft(
    question: str,
    sources: list[dict[str, str]],
    draft: dict[str, Any],
    retrieved_ids: list[str],
    *,
    tier: str = "high",
    force: bool = False,
    _inject: str | None = None,
) -> VerifierResult:
    """High-tier/middle-band only. Low-tier → SKIPPED (latency). Repro/no-key → SKIPPED or local."""
    import time

    t0 = time.perf_counter()
    if tier == "low" and not force:
        return VerifierResult("SKIPPED", "none", "low-tier skip")
    if _inject in ("VERIFIER_ACCEPT", "VERIFIER_REJECT"):
        fc = "none" if _inject == "VERIFIER_ACCEPT" else "entailment"
        return VerifierResult(_inject, fc, "injected", round((time.perf_counter() - t0) * 1000, 1))
    # deterministic local gates first (no network)
    local = _local_checks(draft, sources, retrieved_ids)
    if local.verdict == "VERIFIER_REJECT":
        return VerifierResult(local.verdict, local.failed_check, local.reason, round((time.perf_counter() - t0) * 1000, 1))
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("verifier_skip no GROQ_API_KEY tier=%s", tier)
        return VerifierResult("SKIPPED", "none", "no-key; local 2/3 passed", round((time.perf_counter() - t0) * 1000, 1))
    try:
        _lazy.require_extra("groq", "pip install groq or set GROQ_API_KEY (verifier optional)")
        from groq import Groq  # type: ignore[import-not-found]

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        prompt = verifier_prompt(question, sources, draft)
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        import json

        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        verdict = str(obj.get("verdict", "VERIFIER_REJECT"))
        if verdict not in ("VERIFIER_ACCEPT", "VERIFIER_REJECT"):
            verdict = "VERIFIER_REJECT"
        fc = str(obj.get("failed_check", "none" if verdict == "VERIFIER_ACCEPT" else "entailment"))
        ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("verifier tier=%s verdict=%s ms=%.1f", tier, verdict, ms)
        return VerifierResult(verdict, fc, str(obj.get("reason", ""))[:280], ms, {"prompt_version": VERIFIER_PROMPT_VERSION})
    except Exception as e:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.warning("verifier_error %s: %s", type(e).__name__, str(e)[:120])
        return VerifierResult("SKIPPED", "error", f"verifier-error:{type(e).__name__}", ms)
