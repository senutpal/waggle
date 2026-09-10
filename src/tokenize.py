from __future__ import annotations

"""Code-preserving tokenizer for BM25 sparse retrieval (S3.1).

Wraps ``src.normalize`` masking (``unescape -> mask -> strip_signature``)
then tokenizes with first-match-wins alternation:

- placeholders atomic: ``__email__ | __phone__ | ORD-12345 | <URL> | @USER``
- hashtag atomic + stripped variant: ``#tag`` -> ``[#tag, tag]``
- generic codes atomic: ``[A-Z]+-\\d+(?:-\\d+)*``
- alphanumerics keep digits: ``\\w*\\d\\w*``
- text words ``[a-z']+`` lowercased + light-stemmed (TEXT field only)
- emoji: single ``[^\\w\\s]`` codepoint (ZWJ sequences split — logged limitation)

Lowercase applies to TEXT tokens only. Codes / placeholders / emoji /
hashtags are never stemmed. ``<URL>`` / ``@USER`` positions are kept so
BM25 ``b`` length-norm sees true 20-40 tok tweet lengths.
"""

import re

from src.normalize import normalize_text

BM25_TOKEN_RE = re.compile(
    r"__email__|__phone__|ORD-12345|<URL>|@USER|#\w+|[A-Z]+-\d+(?:-\d+)*|\w*\d\w*|[a-z']+|[^\w\s]",
    re.IGNORECASE,
)

_CODE_LIKE = re.compile(
    r"(?:__email__|__phone__|ORD-12345|<URL>|@USER|#\w+|[A-Z]+-\d+(?:-\d+)*|\w*\d\w*)",
    re.IGNORECASE,
)


def stem_light(tok: str) -> str:
    """Minimal stdlib plural/past stemmer for TEXT tokens only."""
    t = tok
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("es"):
        return t[:-2] + "e"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    if len(t) > 5 and t.endswith("ing") and len(t[:-3]) >= 4:
        return t[:-3]
    if len(t) > 5 and t.endswith("ed") and len(t[:-2]) >= 4:
        return t[:-2]
    return t


def bm25_tokenize(text_norm: str) -> list[str]:
    """Tokenize already-normalized text (docs and queries identically)."""
    toks: list[str] = []
    for m in BM25_TOKEN_RE.finditer(text_norm or ""):
        raw = m.group(0)
        if not raw:
            continue
        if raw.startswith("#") and len(raw) > 1:
            low = raw.lower()
            toks.extend([low, low[1:]])
            continue
        if _CODE_LIKE.fullmatch(raw):
            toks.append(raw if raw.startswith(("<", "@", "_")) else raw)
            # codes keep surface form; text already lowercased upstream
            # for generic codes, normalize case to lower for recall
            if toks[-1] and not toks[-1].startswith(("<", "@", "_")):
                toks[-1] = toks[-1].lower()
            continue
        if re.fullmatch(r"[a-zA-Z']+", raw):
            low = raw.lower()
            if len(low) <= 2:
                toks.append(low)
            else:
                toks.append(stem_light(low))
            continue
        toks.append(raw)
    return toks


def tokenize(raw: str) -> list[str]:
    """Normalize then tokenize raw text (single entry point)."""
    return bm25_tokenize(normalize_text(raw or ""))


def tokenize_batch(texts: list[str]) -> list[list[str]]:
    return [tokenize(t) for t in texts]


def doc_thread_text(inbound_texts: list[str]) -> str:
    """Join inbound turns exactly like train/golden (`` | `` separator)."""
    return " | ".join(inbound_texts)
