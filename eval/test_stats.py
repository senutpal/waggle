from __future__ import annotations


def test_wilson_golden():
    from eval.stats import wilson

    assert wilson(0, 0) == [0.0, 0.0]
    lo, hi = wilson(145, 200)
    assert abs(lo - 0.660) < 0.005 and abs(hi - 0.782) < 0.005


def test_paired_delta_ci_repro():
    from eval.stats import paired_delta_ci

    a = [1, 0, 1, 1, 0, 1]
    b = [1, 1, 1, 1, 0, 1]
    p1, c1 = paired_delta_ci(a, b, seed=123, n_boot=500)
    p2, c2 = paired_delta_ci(a, b, seed=123, n_boot=500)
    assert p1 == p2 and c1 == c2
    assert c1[0] <= p1 <= c1[1]


def test_mcnemar_exact_boundary():
    from eval.stats import mcnemar

    a = [0] * 12 + [1] * 12
    b = [1] * 12 + [0] * 12
    r = mcnemar(a, b)  # b+c=24 -> exact
    assert r["method"] == "exact"
    a2 = [0] * 13 + [1] * 13
    b2 = [1] * 13 + [0] * 13
    r2 = mcnemar(a2, b2)  # 26 -> chi2
    assert r2["method"] == "chi2"
    assert mcnemar([1, 1], [1, 1])["p"] == 1.0


def test_bca_matches_percentile_symmetric():
    from eval.stats import bca_interval

    import numpy as np

    rng = np.random.default_rng(123)
    boots = rng.normal(0.5, 0.05, 2000)
    jack = rng.normal(0.5, 0.05, 50)
    lo, hi = bca_interval(0.5, boots, jack)
    assert lo < 0.5 < hi


def test_wilcoxon_all_zero():
    from eval.stats import wilcoxon_signed_rank

    r = wilcoxon_signed_rank([3, 3, 3], [3, 3, 3])
    assert r["p"] == 1.0


def test_verdict_blocks_small_win():
    from eval.stats import verdict

    assert verdict(0.02, [0.005, 0.035], 0.03, False) == "inconclusive (<MDE)"
    assert verdict(0.05, [0.015, 0.09], 0.0129, True) == "inconclusive (overlapping CIs)"
    assert verdict(0.10, [0.03, 0.17], 0.01, False) == "win"


def test_format_renders_example():
    from eval.stats import format_point_ci

    assert format_point_ci(0.725, 0.660, 0.782) == "72.5% [66.0,78.2]"


def test_holm_blocks_cherry_pick():
    from eval.stats import holm

    out = holm([0.0129, 0.077, 0.38, 0.5, 0.6, 0.7, 0.8])
    assert all(not d["reject"] for d in out)
