"""
Walk-forward backtesting: performance metrics and static baseline
strategies, used to check whether the RL model is actually worth running
before treating its output as a real recommendation.

Pure logic, no network calls or model training here — this is what's
unit tested. The orchestration that actually trains a model per
walk-forward fold lives in walk_forward_backtest.py, since that's
expensive and depends on stable_baselines3/network data.
"""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def cumulative_return(daily_returns: pd.Series) -> float:
    return float(np.prod(1 + daily_returns) - 1)


def annualized_return(daily_returns: pd.Series) -> float:
    """CAGR implied by a series of daily returns."""
    n_days = len(daily_returns)
    if n_days == 0:
        return 0.0
    total_growth = np.prod(1 + daily_returns)
    years = n_days / TRADING_DAYS_PER_YEAR
    if years == 0 or total_growth <= 0:
        return -1.0
    return float(total_growth ** (1 / years) - 1)


def annualized_volatility(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio. risk_free_rate is annualized."""
    vol = annualized_volatility(daily_returns)
    if vol == 0:
        return 0.0
    excess = annualized_return(daily_returns) - risk_free_rate
    return float(excess / vol)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio — like Sharpe but only penalizes downside
    volatility, which fits a long-term-investing lens better (upside
    swings aren't "risk" in the same sense as drawdowns)."""
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_vol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_vol == 0:
        return 0.0
    excess = annualized_return(daily_returns) - risk_free_rate
    return float(excess / downside_vol)


def max_drawdown(daily_returns: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.23
    for a 23% drawdown)."""
    if len(daily_returns) == 0:
        return 0.0
    wealth = (1 + daily_returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1
    return float(drawdown.min())


def average_daily_turnover(weights_df: pd.DataFrame) -> float:
    """Mean L1 distance between consecutive days' weight vectors — 0 if
    weights never change, up to 2.0 if the whole portfolio is replaced
    every day."""
    if len(weights_df) < 2:
        return 0.0
    diffs = weights_df.diff().abs().sum(axis=1).iloc[1:]
    return float(diffs.mean())


def summarize_performance(daily_returns: pd.Series, weights_df: pd.DataFrame = None) -> dict:
    """One dict of all the headline metrics for a return series (and
    optionally its weight history, for turnover)."""
    summary = {
        "cumulative_return": cumulative_return(daily_returns),
        "annualized_return": annualized_return(daily_returns),
        "annualized_volatility": annualized_volatility(daily_returns),
        "sharpe_ratio": sharpe_ratio(daily_returns),
        "sortino_ratio": sortino_ratio(daily_returns),
        "max_drawdown": max_drawdown(daily_returns),
    }
    if weights_df is not None:
        summary["average_daily_turnover"] = average_daily_turnover(weights_df)
    return summary


# ---------------------------------------------------------------------------
# Static baseline strategies
# ---------------------------------------------------------------------------

def _tickers_returns(reward_df: pd.DataFrame, tickers):
    idx_ret = [reward_df.columns.get_loc(f"{t}_ret") for t in tickers]
    return reward_df.iloc[:, idx_ret]


def equal_weight_baseline(reward_df: pd.DataFrame, equity_tickers, bond_tickers, bond_floor: float = 0.25) -> pd.Series:
    """Equal weight within each sleeve, bonds pinned at the floor."""
    all_tickers = equity_tickers + bond_tickers
    weights = np.zeros(len(all_tickers))
    weights[:len(equity_tickers)] = (1 - bond_floor) / len(equity_tickers)
    weights[len(equity_tickers):] = bond_floor / len(bond_tickers)
    returns = _tickers_returns(reward_df, all_tickers).values
    return pd.Series(returns @ weights, index=reward_df.index)


def sixty_forty_baseline(reward_df: pd.DataFrame, equity_tickers, bond_tickers) -> pd.Series:
    """The classic 60% equities / 40% bonds benchmark, equal-weighted
    within each sleeve."""
    all_tickers = equity_tickers + bond_tickers
    weights = np.zeros(len(all_tickers))
    weights[:len(equity_tickers)] = 0.60 / len(equity_tickers)
    weights[len(equity_tickers):] = 0.40 / len(bond_tickers)
    returns = _tickers_returns(reward_df, all_tickers).values
    return pd.Series(returns @ weights, index=reward_df.index)


def inverse_volatility_baseline(reward_df: pd.DataFrame, equity_tickers, bond_tickers,
                                 lookback: int = 63, bond_floor: float = 0.25) -> pd.Series:
    """Weight each asset inversely to its trailing realized volatility
    (risk parity style), rescaled to respect the bond floor. Uses only
    trailing data at each date — no lookahead."""
    all_tickers = equity_tickers + bond_tickers
    returns = _tickers_returns(reward_df, all_tickers)
    rolling_vol = returns.rolling(lookback, min_periods=lookback).std()

    # Floor volatility at a tiny epsilon rather than skipping the date
    # outright when one asset happens to have exactly zero rolling
    # volatility (e.g. a very low-volatility short-term bond fund) — a
    # single flat asset shouldn't blank out the whole day's allocation.
    epsilon = 1e-8

    daily_returns = []
    index = []
    for i in range(len(reward_df)):
        vol_row = rolling_vol.iloc[i]
        if vol_row.isna().any():
            continue  # not enough history yet for this date
        inv_vol = 1.0 / np.maximum(vol_row.values, epsilon)
        raw_weights = inv_vol / inv_vol.sum()

        equity_idx = list(range(len(equity_tickers)))
        bond_idx = list(range(len(equity_tickers), len(all_tickers)))
        total_bond = raw_weights[bond_idx].sum()
        if total_bond < bond_floor:
            raw_weights[bond_idx] *= bond_floor / total_bond
            raw_weights[equity_idx] *= (1 - bond_floor) / raw_weights[equity_idx].sum()

        day_return = returns.iloc[i].values @ raw_weights
        daily_returns.append(day_return)
        index.append(reward_df.index[i])

    return pd.Series(daily_returns, index=index)


def graham_value_screen_baseline(reward_df: pd.DataFrame, equity_tickers, bond_tickers,
                                  bond_floor: float = 0.25, pe_threshold: float = 22.0) -> pd.Series:
    """
    A simple Graham-style value screen: equal-weight equities whose
    *current* P/E is below `pe_threshold` (cheap/reasonably-valued
    stocks only), with any excluded equity weight redistributed to the
    remaining cheap equities. If every equity is "expensive" on a given
    day, falls back to equal-weighting all equities that day rather than
    holding zero equities.
    """
    all_tickers = equity_tickers + bond_tickers
    equity_idx = list(range(len(equity_tickers)))
    bond_idx = list(range(len(equity_tickers), len(all_tickers)))
    returns = _tickers_returns(reward_df, all_tickers)
    pe_cols = [f"{t}_pe" for t in equity_tickers]

    daily_returns = []
    for i in range(len(reward_df)):
        pe_row = reward_df.iloc[i][pe_cols].values
        cheap_mask = pe_row < pe_threshold
        if not cheap_mask.any():
            cheap_mask = np.ones_like(cheap_mask, dtype=bool)  # fall back to all

        weights = np.zeros(len(all_tickers))
        n_cheap = cheap_mask.sum()
        weights[:len(equity_tickers)][cheap_mask] = (1 - bond_floor) / n_cheap
        weights[bond_idx] = bond_floor / len(bond_tickers)

        day_return = returns.iloc[i].values @ weights
        daily_returns.append(day_return)

    return pd.Series(daily_returns, index=reward_df.index)


# ---------------------------------------------------------------------------
# Walk-forward splitting
# ---------------------------------------------------------------------------

def walk_forward_splits(dates: pd.DatetimeIndex, train_years: int = 3, test_years: int = 1):
    """
    Yield (train_index, test_index) pairs for walk-forward validation:
    train on a rolling `train_years`-year window, test on the following
    `test_years`-year window, then roll forward by `test_years` and
    repeat until the data runs out. Both returned values are
    pandas.DatetimeIndex slices of `dates`.
    """
    if len(dates) == 0:
        return

    start = dates[0]
    final = dates[-1]
    train_end = start + pd.DateOffset(years=train_years)

    while True:
        test_end = train_end + pd.DateOffset(years=test_years)
        train_mask = (dates >= start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)

        if test_mask.sum() == 0:
            break

        yield dates[train_mask], dates[test_mask]

        if test_end >= final:
            break
        train_end = test_end
