from __future__ import annotations

"""Baseline intent classifiers: T0 (regex/majority) + T1 (TF-IDF + LogReg).

T1 spec: ``TfidfVectorizer(word1-2 + char_wb3-5, 20k, min_df=5)`` +
``LogisticRegression(C=4, class_weight=balanced)`` (10-way). Implemented as
a ``FeatureUnion`` of a word (1,2) and a char_wb (3,5) vectorizer capped at
10k features each so the union is exactly 20k. A code-preserving
token pattern keeps ``__email__ / ORD-12345 / <URL> / @USER`` atomic.
"""

from src.golden import TAXONOMY, pseudo_intent
from src.normalize import TOKEN_RE

OTHER = "other"

MAJORITY_FALLBACK = "complaint-feedback"


def t0_predict(text: str) -> str:
    """Single regex-priority prediction (same priority as stratifier)."""
    intent, _ = pseudo_intent(text or "")
    return intent


def majority_label(labels: list[str]) -> str:
    from collections import Counter

    if not labels:
        return MAJORITY_FALLBACK
    return Counter(labels).most_common(1)[0][0]


def build_t1_pipeline(max_features: int = 20_000, min_df: int = 5, c: float = 4.0, max_iter: int = 500):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    half = max_features // 2
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=TOKEN_RE,
        lowercase=True,
        sublinear_tf=True,
        min_df=min_df,
        max_features=half,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        min_df=min_df,
        max_features=max_features - half,
    )
    union = FeatureUnion([("word", word), ("char", char)])
    # lbfgs: converges ~5x faster than saga at this scale on a single
    # thread and is deterministic; saga(max_iter=2000) blew the <15min
    # budget. C=4 / balanced per spec.
    clf = LogisticRegression(
        C=c,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=42,
    )
    return Pipeline([("features", union), ("clf", clf)])


def accuracy_macro_f1(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    labs = labels or TAXONOMY
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labs, average="macro", zero_division=0)),
    }


def group_train_test_split(
    thread_ids: list[str], test_size: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Deterministic group split by thread (no turn-level leakage)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    uniq = sorted(set(thread_ids))
    perm = rng.permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_size)))
    test_groups = {uniq[i] for i in perm[:n_test]}
    te = [i for i, g in enumerate(thread_ids) if g in test_groups]
    tr = [i for i, g in enumerate(thread_ids) if g not in test_groups]
    return tr, te
