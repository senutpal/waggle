from __future__ import annotations


def temporal_split(df, cut: str = "2017-10-15T00:00:00Z", gap_hours: int = 24):
    import pandas as pd

    cut_ts = pd.Timestamp(cut, tz="UTC")
    train = df[pd.to_datetime(df["created_at"], utc=True) < cut_ts].copy()
    golden = df[pd.to_datetime(df["created_at"], utc=True) >= cut_ts].copy()
    return train, golden


def thread_split(threads: list[dict], cut: str = "2017-10-15T00:00:00Z", gap_hours: int = 24):
    import pandas as pd

    cut_ts = pd.Timestamp(cut, tz="UTC")
    gap = pd.Timedelta(hours=gap_hours)
    train: list[dict] = []
    golden: list[dict] = []
    quarantine: list[dict] = []
    for th in threads:
        start = pd.Timestamp(th["start"], tz="UTC")
        end = pd.Timestamp(th["end"], tz="UTC")
        if start < cut_ts:
            if end >= cut_ts or end + gap >= cut_ts:
                quarantine.append(th)
            else:
                train.append(th)
        else:
            golden.append(th)
    return train, golden, quarantine


def assert_gap(train_max, golden_min, gap_hours: int = 24) -> float:
    import pandas as pd

    a = pd.to_datetime(train_max, utc=True)
    b = pd.to_datetime(golden_min, utc=True)
    delta_h = (b - a).total_seconds() / 3600.0
    if not (a + pd.Timedelta(hours=gap_hours) < b):
        raise AssertionError(f"gap violated: max_train={a} min_golden={b} delta_h={delta_h} < {gap_hours}")
    return delta_h


def max_min_created(df_train, df_golden, col: str = "created_at"):
    import pandas as pd

    t = pd.to_datetime(df_train[col], utc=True)
    g = pd.to_datetime(df_golden[col], utc=True)
    return t.max(), g.min()
