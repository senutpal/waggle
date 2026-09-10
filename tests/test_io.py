from __future__ import annotations

import io

import pandas as pd
import pytest

from src.io import TWCS_COLUMNS, normalize_inbound, parse_created_at, read_twcs


def _frame() -> pd.DataFrame:
    buf = io.StringIO(
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        "1,AmazonHelp,False,Tue Oct 10 15:16:08 +0000 2017,hi @USER <URL> ^JD,,2\n"
        "2,105835,True,Tue Oct 10 15:10:08 +0000 2017,hello,1,\n"
    )
    return pd.read_csv(buf, dtype=str, keep_default_na=False)


def test_twcs_columns_and_dtypes(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        "119239,105835,True,Tue Oct 10 15:16:08 +0000 2017,hi,,\n",
        encoding="utf-8",
    )
    df = read_twcs(p)
    assert list(df.columns) == TWCS_COLUMNS
    assert df["tweet_id"].iloc[0] == "119239"
    assert ".0" not in df["tweet_id"].iloc[0]


def test_normalize_inbound_bool_and_string():
    assert normalize_inbound(pd.Series([True, False])).tolist() == [True, False]
    out = normalize_inbound(pd.Series(["True", "False", "true"]))
    assert out.tolist() == [True, False, True]
    with pytest.raises(ValueError):
        normalize_inbound(pd.Series(["yes"]))


def test_parse_created_at_utc():
    s = parse_created_at(pd.Series(["Tue Oct 10 15:16:08 +0000 2017", "bad"]))
    assert str(s.iloc[0].tzinfo) == "UTC"
    assert pd.isna(s.iloc[1])
