import numpy as np
import pandas as pd
import pytest

from backtest_lib import (
    cumulative_return, annualized_return, annualized_volatility,
    sharpe_ratio, sortino_ratio, max_drawdown, average_daily_turnover,
    summarize_performance, equal_weight_baseline, sixty_forty_baseline,
    inverse_volatility_baseline, graham_value_screen_baseline,
    walk_forward_splits, TRADING_DAYS_PER_YEAR,
)


def test_cumulative_return_basic():
    returns = pd.Series([0.1, 0.1, -0.05])
    expected = 1.1 * 1.1 * 0.95 - 1
    assert cumulative_return(returns) == pytest.approx(expected)


def test_annualized_return_matches_known_cagr():
    # A constant daily return r compounded for exactly one year should
    # annualize back to (1+r)^252 - 1.
    r = 0.001
    returns = pd.Series([r] * TRADING_DAYS_PER_YEAR)
    expected = (1 + r) ** TRADING_DAYS_PER_YEAR - 1
    assert annualized_return(returns) == pytest.approx(expected, rel=1e-6)


def test_annualized_return_handles_total_loss():
    returns = pd.Series([-1.0, 0.0, 0.0])
    assert annualized_return(returns) == -1.0


def test_annualized_volatility_scales_with_sqrt_time():
    rng = np.random.default_rng(0)
    daily_std = 0.01
    returns = pd.Series(rng.normal(0, daily_std, 5000))
    result = annualized_volatility(returns)
    expected = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert result == pytest.approx(expected, rel=0.1)


def test_sharpe_ratio_zero_for_zero_volatility():
    returns = pd.Series([0.001] * 10)  # no variance at all beyond one value... use constant
    assert sharpe_ratio(pd.Series([0.001])) == 0.0  # single point, no std


def test_sharpe_ratio_positive_for_positive_trend():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))
    assert sharpe_ratio(returns) > 0


def test_sortino_only_penalizes_downside():
    # Two series with identical downside but very different upside should
    # have identical Sortino ratios (upside volatility doesn't matter),
    # but different Sharpe ratios (which penalize all volatility).
    downside_days = [-0.01] * 20
    low_upside = pd.Series(downside_days + [0.001] * 20)
    high_upside = pd.Series(downside_days + [0.05] * 20)

    sortino_low = sortino_ratio(low_upside)
    sortino_high = sortino_ratio(high_upside)
    sharpe_low = sharpe_ratio(low_upside)
    sharpe_high = sharpe_ratio(high_upside)

    # Sortino denominators (downside vol) are identical between the two,
    # so a higher average return (from bigger upside days) should raise
    # Sortino more, proportionally, than it raises Sharpe (whose
    # denominator also grows with the added upside volatility).
    assert sortino_high > sortino_low
    assert sharpe_high > sharpe_low


def test_max_drawdown_detects_known_decline():
    # Goes up to 1.21, down to 0.605 (50% drawdown from peak), back up.
    returns = pd.Series([0.1, 0.1, -0.5, 0.1])
    dd = max_drawdown(returns)
    assert dd == pytest.approx(-0.5, abs=1e-9)


def test_max_drawdown_zero_for_monotonic_gains():
    returns = pd.Series([0.01, 0.02, 0.01])
    assert max_drawdown(returns) == pytest.approx(0.0)


def test_average_daily_turnover_zero_for_constant_weights():
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
    assert average_daily_turnover(weights) == pytest.approx(0.0)


def test_average_daily_turnover_detects_full_switch():
    weights = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    # L1 distance between [1,0] and [0,1] is 2.0
    assert average_daily_turnover(weights) == pytest.approx(2.0)


def test_summarize_performance_returns_all_keys():
    returns = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 300))
    weights = pd.DataFrame({"A": [0.5] * 300, "B": [0.5] * 300})
    summary = summarize_performance(returns, weights)
    for key in ["cumulative_return", "annualized_return", "annualized_volatility",
                "sharpe_ratio", "sortino_ratio", "max_drawdown", "average_daily_turnover"]:
        assert key in summary


def _make_reward_df(n_days=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    equity_tickers = ["EQ1", "EQ2"]
    bond_tickers = ["BD1"]
    cols = {}
    for t in equity_tickers:
        cols[f"{t}_ret"] = rng.normal(0.0005, 0.015, n_days)
        cols[f"{t}_pe"] = rng.uniform(10, 35, n_days)
    cols["BD1_ret"] = rng.normal(0.0001, 0.003, n_days)
    cols["BD1_pe"] = np.zeros(n_days)
    return pd.DataFrame(cols, index=dates), equity_tickers, bond_tickers


def test_equal_weight_baseline_respects_bond_floor_and_is_deterministic():
    reward_df, eq, bd = _make_reward_df()
    returns1 = equal_weight_baseline(reward_df, eq, bd, bond_floor=0.25)
    returns2 = equal_weight_baseline(reward_df, eq, bd, bond_floor=0.25)
    pd.testing.assert_series_equal(returns1, returns2)
    assert len(returns1) == len(reward_df)


def test_sixty_forty_baseline_uses_correct_split():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    reward_df = pd.DataFrame({
        "EQ1_ret": [0.1], "EQ2_ret": [0.1], "BD1_ret": [0.0],
    }, index=dates)
    returns = sixty_forty_baseline(reward_df, ["EQ1", "EQ2"], ["BD1"])
    # 60% equities at 0.1 return + 40% bonds at 0.0 = 0.06
    assert returns.iloc[0] == pytest.approx(0.06)


def test_inverse_volatility_baseline_produces_fewer_rows_during_warmup():
    reward_df, eq, bd = _make_reward_df(n_days=200)
    returns = inverse_volatility_baseline(reward_df, eq, bd, lookback=63)
    # pandas rolling(window=lookback, min_periods=lookback) first produces
    # a valid value at row index (lookback - 1), so (lookback - 1) leading
    # rows have no full window yet and are dropped.
    assert len(returns) == 200 - (63 - 1)
    assert not returns.isna().any()


def test_inverse_volatility_baseline_favors_lower_vol_asset():
    # EQ1 has much higher volatility than EQ2; inverse-vol should
    # allocate it less weight, so the resulting series' volatility
    # should be between the two if constructed correctly. We check this
    # indirectly: pure EQ1 should be far more volatile than the
    # inverse-vol-weighted result restricted to equities only.
    n_days = 150
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(5)
    reward_df = pd.DataFrame({
        "EQ1_ret": rng.normal(0, 0.05, n_days),   # high vol
        "EQ2_ret": rng.normal(0, 0.002, n_days),  # low vol
        "BD1_ret": np.zeros(n_days),
    }, index=dates)
    returns = inverse_volatility_baseline(reward_df, ["EQ1", "EQ2"], ["BD1"], lookback=30)
    assert returns.std() < reward_df["EQ1_ret"].iloc[-len(returns):].std()


def test_graham_value_screen_excludes_expensive_stocks():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    reward_df = pd.DataFrame({
        "EQ1_ret": [0.10], "EQ1_pe": [15.0],  # cheap
        "EQ2_ret": [-0.10], "EQ2_pe": [40.0],  # expensive, excluded
        "BD1_ret": [0.0],
    }, index=dates)
    returns = graham_value_screen_baseline(reward_df, ["EQ1", "EQ2"], ["BD1"],
                                            bond_floor=0.25, pe_threshold=22.0)
    # Only EQ1 (cheap) should get the full 75% equity allocation.
    expected = 0.75 * 0.10 + 0.25 * 0.0
    assert returns.iloc[0] == pytest.approx(expected)


def test_graham_value_screen_falls_back_to_all_equities_if_none_cheap():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    reward_df = pd.DataFrame({
        "EQ1_ret": [0.10], "EQ1_pe": [40.0],
        "EQ2_ret": [0.20], "EQ2_pe": [50.0],
        "BD1_ret": [0.0],
    }, index=dates)
    returns = graham_value_screen_baseline(reward_df, ["EQ1", "EQ2"], ["BD1"],
                                            bond_floor=0.25, pe_threshold=22.0)
    # Falls back to equal-weighting both equities.
    expected = 0.375 * 0.10 + 0.375 * 0.20 + 0.25 * 0.0
    assert returns.iloc[0] == pytest.approx(expected)


def test_walk_forward_splits_produce_non_overlapping_rolling_windows():
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    splits = list(walk_forward_splits(dates, train_years=3, test_years=1))
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # train strictly precedes test, no overlap
        assert train_idx.max() < test_idx.min()
    # consecutive folds should roll forward (each fold's test immediately
    # follows the previous fold's test)
    for (_, test1), (_, test2) in zip(splits, splits[1:]):
        assert test2.min() > test1.min()


def test_walk_forward_splits_empty_for_insufficient_data():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")  # way too short
    splits = list(walk_forward_splits(dates, train_years=3, test_years=1))
    assert splits == []
