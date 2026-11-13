"""
Walk-forward backtest: trains a fresh model on each rolling window and
scores it against four static baselines on the following held-out window,
repeating across the whole price history. This is the Phase 2 tool that
answers "is this strategy actually worth running?" rather than the single
train/holdout split rebalance.py uses to decide whether to overwrite the
saved model.

Run manually (python walk_forward_backtest.py) — this is a validation
tool for development, not part of the monthly production workflow, since
training several models per run is slow.

Network note: like rebalance.py, this fetches real yfinance/SEC EDGAR
data. It is not part of CI and has not been run against live data from
this sandbox (network access is blocked here) — see README.
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from portfolio_lib import (
    ALL_TICKERS, EQUITY_TICKERS, BOND_TICKERS, BOND_FLOOR,
    build_features, make_lagged_observations, BroadMarketArena,
    apply_bond_floor, softmax_weights,
)
from backtest_lib import (
    summarize_performance, equal_weight_baseline, sixty_forty_baseline,
    inverse_volatility_baseline, graham_value_screen_baseline,
    walk_forward_splits,
)
import rebalance

TRAIN_TIMESTEPS_PER_FOLD = 60000


def policy_returns_and_weights(model, obs_df: pd.DataFrame, reward_df: pd.DataFrame,
                                equity_tickers, bond_tickers, floor: float = BOND_FLOOR):
    """Like portfolio_lib.policy_backtest, but returns the full daily
    return series and weight history (for turnover/drawdown metrics)
    instead of just the cumulative return."""
    n_equity = len(equity_tickers)
    equity_idx = list(range(n_equity))
    bond_idx = list(range(n_equity, n_equity + len(bond_tickers)))
    all_tickers = equity_tickers + bond_tickers
    idx_ret = [reward_df.columns.get_loc(f"{t}_ret") for t in all_tickers]

    returns, weight_rows = [], []
    for i in range(len(obs_df)):
        obs = obs_df.iloc[i].values.astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        raw_weights = softmax_weights(action)
        weights = apply_bond_floor(raw_weights, equity_idx, bond_idx, floor)
        returns.append(reward_df.iloc[i, idx_ret].values @ weights)
        weight_rows.append(weights)

    returns_series = pd.Series(returns, index=obs_df.index)
    weights_df = pd.DataFrame(weight_rows, index=obs_df.index, columns=all_tickers)
    return returns_series, weights_df


def run_fold(train_obs, train_reward, test_obs, test_reward):
    env = BroadMarketArena(train_obs, train_reward, EQUITY_TICKERS, BOND_TICKERS)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
    model.learn(total_timesteps=TRAIN_TIMESTEPS_PER_FOLD)

    model_returns, model_weights = policy_returns_and_weights(
        model, test_obs, test_reward, EQUITY_TICKERS, BOND_TICKERS
    )

    results = {"model": summarize_performance(model_returns, model_weights)}
    for name, baseline_fn in [
        ("equal_weight", lambda: equal_weight_baseline(test_reward, EQUITY_TICKERS, BOND_TICKERS)),
        ("sixty_forty", lambda: sixty_forty_baseline(test_reward, EQUITY_TICKERS, BOND_TICKERS)),
        ("inverse_volatility", lambda: inverse_volatility_baseline(test_reward, EQUITY_TICKERS, BOND_TICKERS)),
        ("graham_value_screen", lambda: graham_value_screen_baseline(test_reward, EQUITY_TICKERS, BOND_TICKERS)),
    ]:
        baseline_returns = baseline_fn()
        results[name] = summarize_performance(baseline_returns)

    return results


def print_fold_report(fold_idx, test_start, test_end, results):
    print(f"\n=== Fold {fold_idx}: test period {test_start.date()} to {test_end.date()} ===")
    header = f"{'strategy':<22} | {'CAGR':>8} | {'Sharpe':>7} | {'Sortino':>7} | {'MaxDD':>8} | {'Turnover':>8}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        turnover = metrics.get("average_daily_turnover")
        turnover_str = f"{turnover:.3f}" if turnover is not None else "   n/a"
        print(f"{name:<22} | {metrics['annualized_return']:>7.2%} | "
              f"{metrics['sharpe_ratio']:>7.2f} | {metrics['sortino_ratio']:>7.2f} | "
              f"{metrics['max_drawdown']:>7.2%} | {turnover_str:>8}")


def main():
    print("Fetching price and fundamentals data for the walk-forward backtest...")
    price_close = rebalance.fetch_prices(ALL_TICKERS)
    fundamentals = rebalance.fetch_fundamentals(EQUITY_TICKERS, BOND_TICKERS, price_close)
    full_features = build_features(price_close, fundamentals, ALL_TICKERS)
    obs_df, reward_df = make_lagged_observations(full_features)

    all_fold_results = []
    for fold_idx, (train_dates, test_dates) in enumerate(walk_forward_splits(obs_df.index), start=1):
        train_obs, train_reward = obs_df.loc[train_dates], reward_df.loc[train_dates]
        test_obs, test_reward = obs_df.loc[test_dates], reward_df.loc[test_dates]
        if len(train_obs) < 100 or len(test_obs) < 20:
            continue  # not enough data in this fold to mean much

        results = run_fold(train_obs, train_reward, test_obs, test_reward)
        print_fold_report(fold_idx, test_dates.min(), test_dates.max(), results)
        all_fold_results.append(results)

    if not all_fold_results:
        print("Not enough price history for even one walk-forward fold.")
        return

    print("\n=== Summary across all folds (mean annualized return) ===")
    strategies = all_fold_results[0].keys()
    for name in strategies:
        mean_return = np.mean([fold[name]["annualized_return"] for fold in all_fold_results])
        mean_sharpe = np.mean([fold[name]["sharpe_ratio"] for fold in all_fold_results])
        print(f"{name:<22} | mean CAGR: {mean_return:>7.2%} | mean Sharpe: {mean_sharpe:>6.2f}")

    model_mean_return = np.mean([fold["model"]["annualized_return"] for fold in all_fold_results])
    sixty_forty_mean_return = np.mean([fold["sixty_forty"]["annualized_return"] for fold in all_fold_results])
    if model_mean_return > sixty_forty_mean_return:
        print("\nResult: the RL model beat the 60/40 baseline on average across folds.")
    else:
        print("\nResult: the RL model did NOT beat the 60/40 baseline on average across folds — "
              "reconsider before treating its output as a real recommendation.")


if __name__ == "__main__":
    main()
