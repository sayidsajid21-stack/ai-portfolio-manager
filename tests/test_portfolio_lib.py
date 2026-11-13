import numpy as np
import pandas as pd
import pytest

from portfolio_lib import (
    apply_bond_floor, apply_news_adjustment, cap_single_asset_weight,
    build_features, make_lagged_observations,
    equal_weight_backtest, policy_backtest, MOMENTUM_WINDOWS,
)


def test_bond_floor_rescales_when_below_floor():
    equity_idx = [0, 1]
    bond_idx = [2, 3]
    raw = np.array([0.4, 0.4, 0.1, 0.1])  # 20% bonds, below the 25% floor
    out = apply_bond_floor(raw, equity_idx, bond_idx, floor=0.25)
    assert out[bond_idx].sum() == pytest.approx(0.25, abs=1e-6)
    assert out[equity_idx].sum() == pytest.approx(0.75, abs=1e-6)
    # internal proportions within each sleeve are preserved
    assert out[0] == pytest.approx(out[1])
    assert out[2] == pytest.approx(out[3])


def test_bond_floor_leaves_weights_alone_when_already_above_floor():
    equity_idx = [0, 1]
    bond_idx = [2, 3]
    raw = np.array([0.3, 0.3, 0.2, 0.2])  # 40% bonds, already above floor
    out = apply_bond_floor(raw, equity_idx, bond_idx, floor=0.25)
    np.testing.assert_allclose(out, raw)


def test_bond_floor_handles_zero_bond_weight():
    equity_idx = [0, 1]
    bond_idx = [2, 3]
    raw = np.array([0.5, 0.5, 0.0, 0.0])
    out = apply_bond_floor(raw, equity_idx, bond_idx, floor=0.25)
    assert out[bond_idx].sum() == pytest.approx(0.25, abs=1e-6)
    assert out[equity_idx].sum() == pytest.approx(0.75, abs=1e-6)


def test_weights_always_sum_to_one():
    equity_idx = list(range(5))
    bond_idx = list(range(5, 8))
    rng = np.random.default_rng(0)
    for _ in range(50):
        raw = rng.dirichlet(np.ones(8))
        out = apply_bond_floor(raw, equity_idx, bond_idx, floor=0.25)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)
        assert out[bond_idx].sum() >= 0.25 - 1e-6


def _synthetic_prices(n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "AAA": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days)),
        "BBB": 50 * np.cumprod(1 + rng.normal(0.0001, 0.005, n_days)),
    }, index=dates)


def test_build_features_uses_real_fundamentals_not_random():
    prices = _synthetic_prices()
    fundamentals = {
        "AAA": {"pe": 18.0, "p_to_iv": 0.9, "div": 0.02},
        "BBB": {"pe": 25.0, "p_to_iv": 1.1, "div": 0.0},
    }
    feats = build_features(prices, fundamentals, ["AAA", "BBB"])
    # fundamentals columns should be exactly the provided constant, not random
    assert (feats["AAA_pe"] == 18.0).all()
    assert (feats["BBB_pe"] == 25.0).all()
    # returns column should match pct_change of prices, for the rows kept
    expected_ret = prices["AAA"].pct_change()
    np.testing.assert_allclose(feats["AAA_ret"].values, expected_ret.loc[feats.index].values)


def test_build_features_accepts_point_in_time_fundamentals_dataframe():
    prices = _synthetic_prices(n_days=200)
    # AAA gets real point-in-time fundamentals (a DataFrame, varying
    # across dates); BBB gets the old constant-snapshot dict, to confirm
    # build_features supports both modes side by side.
    pit_fundamentals = pd.DataFrame({
        "pe": np.linspace(15.0, 25.0, len(prices)),
        "p_to_iv": np.linspace(0.8, 1.2, len(prices)),
        "div": np.linspace(0.0, 0.03, len(prices)),
    }, index=prices.index)
    fundamentals = {
        "AAA": pit_fundamentals,
        "BBB": {"pe": 25.0, "p_to_iv": 1.1, "div": 0.0},
    }
    feats = build_features(prices, fundamentals, ["AAA", "BBB"])

    # AAA's pe column should vary across rows (point-in-time), not be
    # constant like the old broadcast behavior.
    assert feats["AAA_pe"].nunique() > 1
    # BBB's pe column should still be the old constant-broadcast behavior.
    assert (feats["BBB_pe"] == 25.0).all()
    # AAA's values should match the (reindexed) point-in-time series.
    expected_aaa_pe = pit_fundamentals["pe"].reindex(feats.index)
    np.testing.assert_allclose(feats["AAA_pe"].values, expected_aaa_pe.values)


def test_build_features_includes_momentum_and_drops_insufficient_history():
    prices = _synthetic_prices(n_days=200)
    fundamentals = {"AAA": {}, "BBB": {}}
    feats = build_features(prices, fundamentals, ["AAA", "BBB"])
    # momentum columns exist for every window
    for name in MOMENTUM_WINDOWS:
        assert f"AAA_{name}" in feats.columns
        assert f"BBB_{name}" in feats.columns
    # the longest lookback window determines how many leading rows are dropped
    longest = max(MOMENTUM_WINDOWS.values())
    assert len(feats) == 200 - longest
    # no NaNs should remain
    assert not feats.isna().any().any()
    # momentum values should match a manual pct_change calculation
    expected_mom_6m = prices["AAA"].pct_change(126)
    np.testing.assert_allclose(feats["AAA_mom_6m"].values, expected_mom_6m.loc[feats.index].values)


def test_lagged_observation_does_not_contain_same_day_return():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    full = pd.DataFrame({
        "AAA_ret": [0.01, 0.02, -0.01, 0.03, -0.02, 0.01],
        "AAA_pe": [20.0] * 6,
    }, index=dates)
    obs_df, reward_df = make_lagged_observations(full)
    # obs row i should equal the *previous* row of full_features
    for i in range(len(obs_df)):
        original_idx = full.index.get_loc(obs_df.index[i])
        np.testing.assert_allclose(obs_df.iloc[i].values, full.iloc[original_idx - 1].values)
        np.testing.assert_allclose(reward_df.iloc[i].values, full.iloc[original_idx].values)
    # the observation for a day must differ from that day's own reward row
    # whenever returns actually change day to day (sanity check it's really lagged)
    assert not obs_df["AAA_ret"].equals(reward_df["AAA_ret"])


def test_equal_weight_backtest_respects_bond_floor():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    tickers = ["EQ1", "EQ2", "BD1"]
    reward_df = pd.DataFrame({
        "EQ1_ret": [0.01, 0.02, -0.01],
        "EQ1_pe": [20.0, 20.0, 20.0],
        "EQ1_p_to_iv": [1.0, 1.0, 1.0],
        "EQ1_div": [0.0, 0.0, 0.0],
        "EQ2_ret": [0.02, -0.01, 0.01],
        "EQ2_pe": [20.0, 20.0, 20.0],
        "EQ2_p_to_iv": [1.0, 1.0, 1.0],
        "EQ2_div": [0.0, 0.0, 0.0],
        "BD1_ret": [0.001, 0.001, 0.001],
        "BD1_pe": [0.0, 0.0, 0.0],
        "BD1_p_to_iv": [1.0, 1.0, 1.0],
        "BD1_div": [0.04, 0.04, 0.04],
    }, index=dates)
    result = equal_weight_backtest(reward_df, tickers, ["EQ1", "EQ2"], ["BD1"], floor=0.25)
    # Hand-computed: weights are EQ1=EQ2=0.375, BD1=0.25
    expected_daily = [
        0.375 * 0.01 + 0.375 * 0.02 + 0.25 * 0.001,
        0.375 * 0.02 + 0.375 * -0.01 + 0.25 * 0.001,
        0.375 * -0.01 + 0.375 * 0.01 + 0.25 * 0.001,
    ]
    expected_cumulative = np.prod([1 + r for r in expected_daily]) - 1
    assert result == pytest.approx(expected_cumulative, abs=1e-9)


class _FakeNewsScore:
    """Mimics news_sentiment.NewsScore's relevant attribute without
    importing pydantic here, to keep this test file's dependencies
    minimal."""
    def __init__(self, sentiment):
        self.sentiment = sentiment


def test_apply_news_adjustment_no_change_without_scores():
    equity_idx = [0, 1]
    weights = np.array([0.4, 0.4, 0.2])
    out = apply_news_adjustment(weights, equity_idx, ["AAA", "BBB"], news_sentiments={})
    np.testing.assert_allclose(out, weights)


def test_apply_news_adjustment_boosts_positive_sentiment_ticker():
    equity_idx = [0, 1]
    weights = np.array([0.4, 0.4, 0.2])
    out = apply_news_adjustment(
        weights, equity_idx, ["AAA", "BBB"],
        news_sentiments={"AAA": 1.0},  # maximally positive
        max_adjustment_fraction=0.03,
    )
    # AAA's weight should increase relative to BBB's (which got no adjustment).
    assert out[0] > out[1]
    # bond weight (index 2) must be completely untouched
    assert out[2] == pytest.approx(0.2)
    # total equity weight preserved
    assert out[0] + out[1] == pytest.approx(0.8, abs=1e-9)
    # overall weights still sum to 1
    assert out.sum() == pytest.approx(1.0, abs=1e-9)


def test_apply_news_adjustment_penalizes_negative_sentiment_ticker():
    equity_idx = [0, 1]
    weights = np.array([0.4, 0.4, 0.2])
    out = apply_news_adjustment(
        weights, equity_idx, ["AAA", "BBB"],
        news_sentiments={"AAA": -1.0},  # maximally negative
    )
    assert out[0] < out[1]
    assert out[2] == pytest.approx(0.2)
    assert out.sum() == pytest.approx(1.0, abs=1e-9)


def test_apply_news_adjustment_accepts_object_with_sentiment_attribute():
    equity_idx = [0, 1]
    weights = np.array([0.4, 0.4, 0.2])
    out_with_object = apply_news_adjustment(
        weights, equity_idx, ["AAA", "BBB"],
        news_sentiments={"AAA": _FakeNewsScore(sentiment=0.5)},
    )
    out_with_float = apply_news_adjustment(
        weights, equity_idx, ["AAA", "BBB"],
        news_sentiments={"AAA": 0.5},
    )
    np.testing.assert_allclose(out_with_object, out_with_float)


def test_apply_news_adjustment_clips_out_of_range_sentiment():
    equity_idx = [0, 1]
    weights = np.array([0.4, 0.4, 0.2])
    out_clipped = apply_news_adjustment(weights, equity_idx, ["AAA", "BBB"], news_sentiments={"AAA": 5.0})
    out_max = apply_news_adjustment(weights, equity_idx, ["AAA", "BBB"], news_sentiments={"AAA": 1.0})
    np.testing.assert_allclose(out_clipped, out_max)


def test_cap_single_asset_weight_no_change_when_under_cap():
    weights = np.array([0.3, 0.3, 0.3, 0.1])  # the 4th index is a bond, untouched anyway
    idx_list = [0, 1, 2]
    out = cap_single_asset_weight(weights, idx_list, max_weight=0.5)
    np.testing.assert_allclose(out, weights)


def test_cap_single_asset_weight_clips_and_redistributes():
    # 5 equities (indices 0-4) summing to 0.90, so a 0.20 cap is feasible
    # (max achievable is 5 * 0.20 = 1.0 >= 0.90); a bond at index 5 is
    # outside idx_list and must stay untouched.
    weights = np.array([0.70, 0.05, 0.05, 0.05, 0.05, 0.10])
    idx_list = [0, 1, 2, 3, 4]
    out = cap_single_asset_weight(weights, idx_list, max_weight=0.20)
    assert out[0] == pytest.approx(0.20, abs=1e-6)
    # excess (0.50) should be redistributed equally among indices 1-4,
    # which started equal
    assert out[1] == pytest.approx(out[2]) == pytest.approx(out[3]) == pytest.approx(out[4])
    # sleeve total preserved exactly
    assert out[idx_list].sum() == pytest.approx(0.90, abs=1e-6)
    # bond weight (outside idx_list) completely untouched
    assert out[5] == pytest.approx(0.10)


def test_cap_single_asset_weight_handles_cascading_caps():
    # Capping the biggest weight and redistributing its excess can push
    # a second asset over the cap too — the function must keep iterating
    # (and must not hand any more excess to an asset already clipped to
    # exactly the cap) until everything in idx_list is at or under it.
    # With 3 assets and cap 0.40, max achievable is 1.2 >= 1.0, feasible.
    weights = np.array([0.80, 0.15, 0.05])
    idx_list = [0, 1, 2]
    out = cap_single_asset_weight(weights, idx_list, max_weight=0.40)
    assert (out[idx_list] <= 0.40 + 1e-6).all()
    assert out.sum() == pytest.approx(1.0, abs=1e-6)
    # Hand-traced expected result: [0.40, 0.40, 0.20]
    np.testing.assert_allclose(out, [0.40, 0.40, 0.20], atol=1e-6)


def test_cap_single_asset_weight_preserves_sleeve_total_exactly():
    rng = np.random.default_rng(3)
    idx_list = list(range(5))  # last index (bond) excluded from the cap
    # max_weight=0.25 with 5 assets is always feasible regardless of the
    # draw (5 * 0.25 = 1.25 >= any possible sleeve total of <= 1.0), so
    # the sleeve-total-preservation invariant always holds here — unlike
    # the cascading-caps test above, which deliberately explores a case
    # where a smaller cap is still feasible but requires iteration.
    for _ in range(20):
        raw = rng.dirichlet(np.ones(6))
        out = cap_single_asset_weight(raw, idx_list, max_weight=0.25)
        assert out[idx_list].sum() == pytest.approx(raw[idx_list].sum(), abs=1e-6)
        assert out[5] == pytest.approx(raw[5])  # untouched
        assert (out[idx_list] <= 0.25 + 1e-6).all()
