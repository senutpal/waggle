from __future__ import annotations

import hashlib
from collections import deque
from typing import Any

from src.logging_utils import get_logger

logger = get_logger(__name__)

EMPTY_TOKENS = {"", "nan", "none", "null", "nat"}


def _clean_id(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in EMPTY_TOKENS:
        return None
    return s


def _split_children(value: Any) -> list[str]:
    if value is None:
        return []
    s = str(value).strip()
    if s.lower() in EMPTY_TOKENS:
        return []
    return [c.strip() for c in s.split(",") if c.strip() and c.strip().lower() not in EMPTY_TOKENS]


def build_indexes(df) -> dict[str, Any]:
    by_id: dict[str, int] = {}
    children: dict[str, list[str]] = {}
    parent: dict[str, str | None] = {}
    for row in df.itertuples():
        tid = _clean_id(row.tweet_id)
        if tid is None:
            continue
        if tid in by_id:
            continue
        by_id[tid] = int(row.Index)
        children[tid] = _split_children(row.response_tweet_id)
        parent[tid] = _clean_id(row.in_response_to_tweet_id)
    rev: dict[str, list[str]] = {tid: [] for tid in by_id}
    for tid, kids in children.items():
        for kid in kids:
            if kid in rev:
                rev[kid].append(tid)
    return {"by_id": by_id, "children": children, "parent": parent, "rev": rev}


def _neighbors(u: str, children, parent, rev, by_id) -> list[str]:
    nbrs: list[str] = []
    nbrs.extend(children.get(u, []))
    pu = parent.get(u)
    if pu is not None and pu in by_id:
        nbrs.append(pu)
    nbrs.extend(rev.get(u, []))
    return nbrs


def reconstruct_threads(
    df,
    gap_hours: float = 24.0,
    max_span_hours: float = 48.0,
    max_nominal_turns: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "n_input": int(len(df)),
        "missing_parent": 0,
        "parent_not_inbound_pool": 0,
        "cycles_guarded": 0,
        "fanout_threads": 0,
        "fanout_total": 0,
        "splits_24h": 0,
        "multi_company_dropped": 0,
        "singletons_dropped": 0,
        "long_flagged": 0,
        "over48_flagged": 0,
        "first_not_inbound_dropped": 0,
        "link_time_inversions": 0,
        "threads_emitted": 0,
        "avg_turns": 0.0,
    }
    df = df.copy()
    df["tweet_id"] = df["tweet_id"].astype(str).str.strip()
    df = df.dropna(subset=["created_at"])
    df = df.sort_values("created_at", kind="mergesort").reset_index(drop=True)
    created = dict(zip(df["tweet_id"].astype(str), df["created_at"]))
    inbound_m = dict(zip(df["tweet_id"].astype(str), df["inbound"].astype(bool)))
    author_m = dict(zip(df["tweet_id"].astype(str), df["author_id"].astype(str)))
    idx = build_indexes(df)
    by_id, children, parent, rev = idx["by_id"], idx["children"], idx["parent"], idx["rev"]
    for tid, par in parent.items():
        if par is not None and par not in by_id:
            stats["missing_parent"] += 1
    roots = sorted([t for t, p in parent.items() if p is None], key=lambda t: created[t])
    ordered_all = sorted(by_id, key=lambda t: created[t])
    visited: set[str] = set()
    components: list[list[str]] = []

    def bfs_collect(seed: str) -> list[str]:
        comp: list[str] = []
        in_comp: set[str] = set()
        queue: deque[str] = deque([seed])
        visited.add(seed)
        while queue:
            u = queue.popleft()
            comp.append(u)
            in_comp.add(u)
            for v in _neighbors(u, children, parent, rev, by_id):
                if v not in by_id:
                    continue
                if v in visited:
                    if v in in_comp:
                        stats["cycles_guarded"] += 1
                    continue
                visited.add(v)
                queue.append(v)
        return comp

    for r in roots:
        if r not in visited:
            components.append(bfs_collect(r))
    for tid in ordered_all:
        if tid not in visited:
            components.append(bfs_collect(tid))
    threads: list[dict[str, Any]] = []
    for comp in components:
        companies = {author_m[t] for t in comp if not inbound_m[t]}
        if len(companies) == 0:
            if len(comp) < 2:
                stats["singletons_dropped"] += 1
            continue
        if len(companies) > 1:
            stats["multi_company_dropped"] += 1
            continue
        company = sorted(companies)[0]
        if len(comp) < 2:
            stats["singletons_dropped"] += 1
            continue
        turns = sorted(comp, key=lambda t: created[t])
        for a, b in zip(turns, turns[1:]):
            if created[b] < created[a]:
                raise AssertionError("time inversion in thread")
        for t in turns:
            par = parent.get(t)
            if par in by_id and created[par] > created[t]:
                stats["link_time_inversions"] += 1
        in_set = set(turns)
        fanout_here = sum(1 for t in turns if sum(1 for c in children.get(t, []) if c in in_set) > 1)
        if fanout_here:
            stats["fanout_threads"] += 1
            stats["fanout_total"] += fanout_here
            turns = _linearize_earliest_child(turns, children, parent, created)
        pieces = _split_on_gap(turns, created, gap_hours)
        stats["splits_24h"] += max(0, len(pieces) - 1)
        for piece in pieces:
            span_h = (created[piece[-1]] - created[piece[0]]).total_seconds() / 3600.0
            over48 = span_h >= max_span_hours
            if over48:
                stats["over48_flagged"] += 1
            long_flag = len(piece) > max_nominal_turns
            if long_flag:
                stats["long_flagged"] += 1
            if not inbound_m[piece[0]]:
                stats["first_not_inbound_dropped"] += 1
                continue
            thread_id = hashlib.sha1("|".join(sorted(piece)).encode("utf-8")).hexdigest()[:16]
            threads.append(
                {
                    "thread_id": thread_id,
                    "company": company,
                    "turns": piece,
                    "n_turns": len(piece),
                    "span_h": round(span_h, 3),
                    "long_flag": bool(long_flag),
                    "over48": bool(over48),
                    "start": created[piece[0]].isoformat(),
                    "end": created[piece[-1]].isoformat(),
                }
            )
    threads.sort(key=lambda t: (t["start"], t["thread_id"]))
    stats["threads_emitted"] = len(threads)
    stats["fanout_rate"] = (stats["fanout_threads"] / len(threads)) if threads else 0.0
    if threads:
        stats["avg_turns"] = round(sum(t["n_turns"] for t in threads) / len(threads), 3)
    logger.info("threads=%d avg_turns=%s fanout_rate=%.4f", len(threads), stats["avg_turns"], stats["fanout_rate"])
    return threads, stats


def _linearize_earliest_child(turns, children, parent, created) -> list[str]:
    in_set = set(turns)
    root_candidates = [t for t in turns if parent.get(t) not in in_set]
    root = min(root_candidates, key=lambda t: created[t]) if root_candidates else turns[0]
    path: list[str] = []
    seen: set[str] = set()
    cur: str | None = root
    while cur is not None and cur in in_set and cur not in seen:
        seen.add(cur)
        path.append(cur)
        kids = sorted([c for c in children.get(cur, []) if c in in_set], key=lambda x: created[x])
        cur = kids[0] if kids else None
    return path


def _split_on_gap(turns: list[str], created: dict, gap_hours: float) -> list[list[str]]:
    import pandas as pd

    pieces: list[list[str]] = [[turns[0]]]
    gap = pd.Timedelta(hours=float(gap_hours))
    for prev, cur in zip(turns, turns[1:]):
        if created[cur] - created[prev] > gap:
            pieces.append([cur])
        else:
            pieces[-1].append(cur)
    return pieces


def threads_to_frame(threads: list[dict], df):
    import pandas as pd

    rows_by_id = {str(r.tweet_id).strip(): r for r in df.itertuples()}
    records: list[dict] = []
    for th in threads:
        for pos, tid in enumerate(th["turns"]):
            r = rows_by_id[tid]
            records.append(
                {
                    "thread_id": th["thread_id"],
                    "company": th["company"],
                    "pos": pos,
                    "n_turns": th["n_turns"],
                    "long_flag": th["long_flag"],
                    "tweet_id": tid,
                    "author_id": str(r.author_id),
                    "inbound": bool(r.inbound),
                    "created_at": r.created_at,
                    "text": str(r.text),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame = frame.sort_values(["created_at", "thread_id", "pos"], kind="mergesort").reset_index(drop=True)
    return frame
