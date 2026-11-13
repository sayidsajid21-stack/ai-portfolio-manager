"""
Monthly training/allocation run for the AI portfolio manager.

This targets long-term investing, not day trading: it's meant to run on a
monthly cadence (see .github/workflows/monthly_rebalance.yml), using
longer-horizon momentum features and a turnover penalty (both in
portfolio_lib.py) that reward the model for holding positions rather than
trading often.

Fixes applied vs. the original version (see project plan, Phase 0):
  1. No more random fundamentals — real current snapshots from yfinance.
  2. No lookahead — the model's observation for day t is lagged by one
     day; it never sees the return it is being scored on until after it
     commits to weights.
  3. The final recommendation uses the most recent available day, not
     day 0 of the training window.
  4. The model is loaded and continued (not recreated from scratch) when
     a saved model exists, and the saved file is only overwritten if the
     new model backtests at least as well as the old one AND beats the
     equal-weight baseline on a held-out slice.
  5. The bond-floor rule lives in one function (portfolio_lib.apply_bond_floor),
     used everywhere weights are computed.
  6. yfinance end date now includes today.
"""
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

from portfolio_lib import (
    ALL_TICKERS, EQUITY_TICKERS, BOND_TICKERS, BOND_FLOOR,
    build_features, make_lagged_observations, BroadMarketArena,
    apply_bond_floor, softmax_weights, equal_weight_backtest, policy_backtest,
)

MODEL_PATH = "graham_value_ai"
MODEL_FILE = f"{MODEL_PATH}.zip"
METADATA_PATH = "model_metadata.json"
HOLDOUT_DAYS = 60  # last N days held out to validate before overwriting the saved model


def fetch_fundamentals(tickers):
    """
    Pull a current fundamentals snapshot per ticker from yfinance.

    This is real data, not random noise, but it is a single point-in-time
    snapshot broadcast across the whole training window (see
    portfolio_lib.build_features docstring) — a known simplification until
    Phase 1 adds point-in-time history from SEC EDGAR.
    """
    fundamentals = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:
            print(f"  Warning: could not fetch fundamentals for {ticker}: {e}")
            info = {}

        pe = info.get('trailingPE') or info.get('forwardPE') or 20.0
        p_to_iv = info.get('priceToBook') or 1.0  # proxy for price-to-intrinsic-value
        div = info.get('dividendYield') or 0.0
        # yfinance has, at various versions, returned dividendYield as
        # either a fraction (0.02) or a percentage (2.0) — normalize.
        if div > 1.0:
            div = div / 100.0

        fundamentals[ticker] = {
            'pe': float(np.clip(pe, 1.0, 100.0)),
            'p_to_iv': float(np.clip(p_to_iv, 0.1, 10.0)),
            'div': float(np.clip(div, 0.0, 0.15)),
        }
    return fundamentals


def fetch_prices(tickers, years=5):
    end_date = datetime.today() + timedelta(days=1)  # yfinance end is exclusive; include today
    start_date = end_date - timedelta(days=years * 365)
    print(f"Fetching prices from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    return data['Close']


def load_or_create_model(env):
    if os.path.exists(MODEL_FILE):
        print(f"Loading existing model from {MODEL_FILE} and continuing training...")
        model = PPO.load(MODEL_PATH, env=env)
        return model, True
    print("No existing model found — creating a new one.")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
    return model, False


def main():
    print("1. Fetching real price and fundamentals data...")
    price_close = fetch_prices(ALL_TICKERS)
    fundamentals = fetch_fundamentals(ALL_TICKERS)

    print("2. Building features (real fundamentals, no synthetic noise)...")
    full_features = build_features(price_close, fundamentals, ALL_TICKERS)

    print("3. Lagging observations by one day to remove lookahead...")
    obs_df, reward_df = make_lagged_observations(full_features)

    split = max(len(obs_df) - HOLDOUT_DAYS, int(len(obs_df) * 0.8))
    train_obs, train_reward = obs_df.iloc[:split], reward_df.iloc[:split]
    holdout_obs, holdout_reward = obs_df.iloc[split:], reward_df.iloc[split:]
    print(f"   Train: {len(train_obs)} days, Holdout: {len(holdout_obs)} days")

    env = BroadMarketArena(train_obs, train_reward, EQUITY_TICKERS, BOND_TICKERS)

    print("4. Loading/creating model and training...")
    model, continued = load_or_create_model(env)
    model.learn(total_timesteps=60000, reset_num_timesteps=not continued)
    print("Training complete.")

    print("5. Validating on held-out data before saving...")
    baseline_return = equal_weight_backtest(holdout_reward, ALL_TICKERS, EQUITY_TICKERS, BOND_TICKERS)
    new_model_return = policy_backtest(model, holdout_obs, holdout_reward, EQUITY_TICKERS, BOND_TICKERS)

    old_model_return = None
    if continued and os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            old_meta = json.load(f)
        old_model_return = old_meta.get("holdout_return")

    print(f"   Equal-weight baseline holdout return: {baseline_return:.4%}")
    print(f"   New model holdout return:              {new_model_return:.4%}")
    if old_model_return is not None:
        print(f"   Previously saved model holdout return: {old_model_return:.4%}")

    beats_baseline = new_model_return > baseline_return
    beats_old = old_model_return is None or new_model_return >= old_model_return

    if beats_baseline and beats_old:
        model.save(MODEL_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump({
                "last_trained": datetime.today().strftime("%Y-%m-%d"),
                "holdout_return": new_model_return,
                "baseline_return": baseline_return,
            }, f, indent=2)
        print("Model IMPROVED or matched previous — saved.")
    else:
        print("Model did NOT beat the baseline/previous model — not saved. Keeping prior model file.")

    print("\n6. Generating today's recommendation from the most recent available day...")
    latest_obs = obs_df.iloc[-1].values.astype(np.float32)
    action, _ = model.predict(latest_obs, deterministic=True)
    raw_weights = softmax_weights(action)

    equity_idx = list(range(len(EQUITY_TICKERS)))
    bond_idx = list(range(len(EQUITY_TICKERS), len(ALL_TICKERS)))
    final_weights = apply_bond_floor(raw_weights, equity_idx, bond_idx, BOND_FLOOR)

    print(f"\n=== AI PORTFOLIO RECOMMENDATION (as of {obs_df.index[-1].date()}) ===")
    print(f"{'Asset':<10} | {'Type':<10} | {'Weight'}")
    print("-" * 35)
    total_eq, total_bd = 0.0, 0.0
    for i, ticker in enumerate(ALL_TICKERS):
        weight = final_weights[i]
        if i in bond_idx:
            total_bd += weight
            print(f"{ticker:<10} | Bond       | {weight*100:05.2f}%")
        else:
            total_eq += weight
            print(f"{ticker:<10} | Equity     | {weight*100:05.2f}%")
    print("-" * 35)
    print(f"TOTAL EQUITY: {total_eq*100:.2f}%")
    print(f"TOTAL BONDS:  {total_bd*100:.2f}%")


if __name__ == "__main__":
    main()
