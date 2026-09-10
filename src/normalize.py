from __future__ import annotations

import html
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
ORDER_RE = re.compile(r"\b(?:\d{3}-\d{7}-\d{7}|ORD-?\d{5,})\b", re.IGNORECASE)
SIG_RE = re.compile(r"\s\^[A-Za-z]{2,4}\s*$")
WS_RE = re.compile(r"\s+")
TOKEN_RE = r"__email__|__phone__|ORD-12345|<URL>|@USER|[\w']+|[^\w\s]"


def unescape(text: str) -> str:
    return html.unescape(text)


def mask(text: str) -> str:
    t = EMAIL_RE.sub("__email__", text)
    t = ORDER_RE.sub("ORD-12345", t)
    t = PHONE_RE.sub("__phone__", t)
    t = URL_RE.sub("<URL>", t)
    t = MENTION_RE.sub("@USER", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def strip_signature(text: str) -> str:
    return SIG_RE.sub("", text).strip()


def normalize_text(text: str) -> str:
    t = unescape(text)
    t = mask(t)
    t = strip_signature(t)
    return t


def normalize_key(text: str) -> str:
    import hashlib

    norm = WS_RE.sub(" ", text.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def exact_dedup(df, col: str = "text_norm"):
    keep_idx: list[int] = []
    seen: set[str] = set()
    removed = 0
    for i, v in enumerate(df[col].astype(str).tolist()):
        k = normalize_key(v)
        if k in seen:
            removed += 1
            continue
        seen.add(k)
        keep_idx.append(i)
    out = df.iloc[keep_idx].reset_index(drop=True)
    return out, {"n_in": int(len(df)), "n_out": int(len(out)), "removed": int(removed)}


def char5grams(text: str):
    low = text.lower()
    if len(low) < 5:
        return [low]
    return [low[i : i + 5] for i in range(len(low) - 4)]


def minhash_signature(text: str, num_perm: int = 128, seed: int = 42):
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm, seed=seed, scheme="affine32")
    for sh in char5grams(text):
        m.update(sh.encode("utf-8"))
    return m


def minhash_dedup(texts: list[str], threshold: float = 0.85, num_perm: int = 128, seed: int = 42):
    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs = [minhash_signature(t, num_perm=num_perm, seed=seed) for t in texts]
    keep: list[int] = []
    n_candidates = 0
    for i, sig in enumerate(sigs):
        hits = lsh.query(sig)
        if hits:
            n_candidates += 1
            continue
        lsh.insert(str(i), sig)
        keep.append(i)
    return keep, {"n_in": len(texts), "n_out": len(keep), "n_flagged": int(n_candidates), "b": lsh.b, "r": lsh.r}


def tfidf_cosine_verify(a: str, b: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
    m = vec.fit_transform([a, b])
    return float(cosine_similarity(m[0], m[1])[0, 0])


def lid_filter(texts: list[str], keep_gt: float = 0.65, model_path: str | None = None):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import fasttext

    path = model_path or "cache/lid.176.ftz"
    model = fasttext.load_model(path)
    keep: list[bool] = []
    probs: list[float] = []
    for t in texts:
        line = t.replace("\n", " ").strip()
        if not line:
            keep.append(False)
            probs.append(0.0)
            continue
        labels, scores = model.predict(line, k=1)
        label = str(labels[0])
        p = float(scores[0])
        probs.append(p)
        keep.append(label == "__label__en" and p > keep_gt)
    return keep, probs


def scan_pii(texts: list[str]) -> dict[str, int]:
    n_email = sum(1 for t in texts if EMAIL_RE.search(t))
    n_phone = sum(1 for t in texts if PHONE_RE.search(t))
    return {"email_hits": int(n_email), "phone_hits": int(n_phone)}
