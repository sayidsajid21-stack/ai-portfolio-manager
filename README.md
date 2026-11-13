# AI Portfolio Manager

A reinforcement-learning (PPO) agent for **long-term investing**, not day
trading. It allocates a portfolio across 16 large US stocks and 3 bond
ETFs, with a hard rule that at least 25% stays in bonds, and is rebalanced
monthly rather than daily — trading less often, using multi-month trend
features, and penalizing unnecessary turnover so it doesn't chase
short-term noise.

## How it runs

- `.github/workflows/monthly_rebalance.yml` runs at 22:00 UTC on the 1st of
  each month (or on demand via "Run workflow"): fetches data,
  trains/updates the model, validates it on held-out data, and prints a
  recommendation.
- `.github/workflows/ci.yml` runs the unit tests on every push/PR — it does
  **not** retrain the model.

## Project layout

- `portfolio_lib.py` — feature engineering, the trading environment, and the
  bond-floor/turnover rules. Pure logic, no network calls — this is what's
  unit tested.
- `rebalance.py` — the monthly entry point: fetches data, trains, validates,
  saves, and prints the recommendation.
- `tests/` — unit tests for `portfolio_lib.py` that run without network access
  or GPU, using synthetic data.
- `model_metadata.json` — small text file tracking when the model was last
  retrained and its holdout backtest score. Committed to git (unlike the
  model binary itself).
- `graham_value_ai.zip` — the trained model. **Not committed to git** — it's
  published as a workflow artifact and as an asset on the `latest-model`
  GitHub release, so retraining doesn't bloat repo history with a binary
  diff every run.

## Why monthly, not daily

This project originally retrained and reallocated every weekday, which is a
day-trading cadence — it chases daily noise and racks up trading costs with
no clear benefit for a long-term holding strategy. Three things changed to
match a long-term orientation:

- **Cadence**: the workflow runs monthly instead of daily (`cron: '0 22 1 *
  *'`). Training itself still uses daily price data internally (monthly
  samples alone would be far too few data points for PPO to learn from), but
  the model is only asked to produce a new, actionable recommendation once a
  month.
- **Features**: `portfolio_lib.MOMENTUM_WINDOWS` adds 3-month and 6-month
  trailing return features alongside the 1-day return, so the model's
  observation reflects the medium-term trend, not just yesterday's move.
- **Turnover penalty**: `portfolio_lib.TURNOVER_COST_RATE` penalizes the
  training reward whenever the model's weights change from the previous
  step, so it's rewarded for holding stable positions rather than trading
  often. Separately, `TRANSACTION_COST_BPS` deducts a realistic trading-cost
  estimate from the backtests in `policy_backtest`/`equal_weight_backtest`,
  so a model that happens to trade a lot doesn't look artificially better
  than it would after real-world costs.

## Known limitations (tracked, not yet fixed)

- **Fundamentals are a single current snapshot, broadcast across the whole
  training history.** yfinance only exposes today's P/E, price-to-book, and
  dividend yield — not what those were on each past date. The model is
  trained as if today's valuation always applied. The project plan's Phase 1
  replaces this with real point-in-time fundamentals from SEC EDGAR.
- **No news/sentiment input yet** (planned Phase 4).
- **Recommendations only — no live or paper trading wired up** (planned
  Phase 6, via Alpaca's paper-trading API).
- **No proper walk-forward backtest against baselines yet** beyond the
  single train/holdout split used to gate whether a freshly trained model
  replaces the saved one (planned Phase 2).

## Fixed in this round (Phase 0)

1. Fundamentals are now real yfinance snapshots instead of `np.random` noise.
2. The model's observation for day *t* is lagged by one day, so it can no
   longer see the same day's return it's being scored on during training
   (previously a lookahead/leakage bug).
3. The recommendation is generated from the most recent available day,
   not day 0 of the training window.
4. The model is loaded and continues training from the previous run's
   weights (rather than being recreated from scratch every run), and the
   saved file is only overwritten if the new model beats both the
   equal-weight baseline and the previously saved model on a held-out slice.
5. The 25%-bonds-minimum rule now lives in one function
   (`portfolio_lib.apply_bond_floor`), used by both the training environment
   and the final allocation output — previously duplicated and inconsistent.
6. Workflow fixes: stopped retraining on every push (now schedule/manual
   only); added `permissions: contents: write`; added a separate CI workflow
   that runs tests on push/PR; stopped committing the model binary to git
   every run (now a release asset + workflow artifact instead).

## Long-term orientation (follow-up round)

This project is meant for long-term investing, not day trading — see "Why
monthly, not daily" above for the cadence, feature, and turnover-penalty
changes that followed from that.
