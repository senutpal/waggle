from __future__ import annotations

import pandas as pd

from src.threads import reconstruct_threads


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def test_linearize_fanout_earliest_child():
    df = _df(
        [
            {"tweet_id": "1", "author_id": "c1", "inbound": True, "created_at": "2017-10-01T00:00:00Z", "text": "a", "response_tweet_id": "2,3", "in_response_to_tweet_id": ""},
            {"tweet_id": "2", "author_id": "AmazonHelp", "inbound": False, "created_at": "2017-10-01T01:00:00Z", "text": "b", "response_tweet_id": "", "in_response_to_tweet_id": "1"},
            {"tweet_id": "3", "author_id": "AmazonHelp", "inbound": False, "created_at": "2017-10-01T02:00:00Z", "text": "c", "response_tweet_id": "", "in_response_to_tweet_id": "1"},
        ]
    )
    threads, stats = reconstruct_threads(df)
    assert len(threads) == 1
    assert threads[0]["turns"] == ["1", "2"]
    assert stats["fanout_threads"] == 1


def test_cycle_guard_terminates():
    df = _df(
        [
            {"tweet_id": "1", "author_id": "c1", "inbound": True, "created_at": "2017-10-01T00:00:00Z", "text": "a", "response_tweet_id": "2", "in_response_to_tweet_id": "2"},
            {"tweet_id": "2", "author_id": "AmazonHelp", "inbound": False, "created_at": "2017-10-01T01:00:00Z", "text": "b", "response_tweet_id": "1", "in_response_to_tweet_id": "1"},
        ]
    )
    threads, _ = reconstruct_threads(df)
    assert len(threads) == 1


def test_split_on_24h_gap_and_long_flag():
    df = _df(
        [
            {"tweet_id": "1", "author_id": "c1", "inbound": True, "created_at": "2017-10-01T00:00:00Z", "text": "a", "response_tweet_id": "2", "in_response_to_tweet_id": ""},
            {"tweet_id": "2", "author_id": "AmazonHelp", "inbound": False, "created_at": "2017-10-01T01:00:00Z", "text": "b", "response_tweet_id": "3", "in_response_to_tweet_id": "1"},
            {"tweet_id": "3", "author_id": "c1", "inbound": True, "created_at": "2017-10-05T00:00:00Z", "text": "c", "response_tweet_id": "4", "in_response_to_tweet_id": "2"},
            {"tweet_id": "4", "author_id": "AmazonHelp", "inbound": False, "created_at": "2017-10-05T01:00:00Z", "text": "d", "response_tweet_id": "", "in_response_to_tweet_id": "3"},
        ]
    )
    threads, stats = reconstruct_threads(df)
    assert len(threads) == 2
    assert stats["splits_24h"] == 1


def test_long_flag_and_sorted_no_inversion():
    rows = []
    for i in range(10):
        rows.append(
            {
                "tweet_id": str(i + 1),
                "author_id": "c1" if i % 2 == 0 else "AmazonHelp",
                "inbound": i % 2 == 0,
                "created_at": f"2017-10-01T{i:02d}:00:00Z",
                "text": f"t{i}",
                "response_tweet_id": str(i + 2) if i < 9 else "",
                "in_response_to_tweet_id": str(i) if i > 0 else "",
            }
        )
    threads, _ = reconstruct_threads(_df(rows))
    assert len(threads) == 1
    assert threads[0]["long_flag"] is True
    assert threads[0]["n_turns"] == 10


def test_drop_multi_company_and_singleton():
    df = _df(
        [
            {"tweet_id": "1", "author_id": "c1", "inbound": True, "created_at": "2017-10-01T00:00:00Z", "text": "a", "response_tweet_id": "2", "in_response_to_tweet_id": ""},
            {"tweet_id": "2", "author_id": "Delta", "inbound": False, "created_at": "2017-10-01T01:00:00Z", "text": "b", "response_tweet_id": "3", "in_response_to_tweet_id": "1"},
            {"tweet_id": "3", "author_id": "AmazonHelp", "inbound": False, "created_at": "2017-10-01T02:00:00Z", "text": "c", "response_tweet_id": "", "in_response_to_tweet_id": "2"},
            {"tweet_id": "9", "author_id": "c9", "inbound": True, "created_at": "2017-10-01T00:00:00Z", "text": "solo", "response_tweet_id": "", "in_response_to_tweet_id": ""},
        ]
    )
    threads, stats = reconstruct_threads(df)
    assert threads == []
    assert stats["multi_company_dropped"] == 1
    assert stats["singletons_dropped"] == 1
