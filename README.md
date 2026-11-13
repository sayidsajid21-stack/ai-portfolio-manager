# AI Portfolio Manager

A reinforcement-learning (PPO) agent that allocates a daily portfolio across
16 large US stocks and 3 bond ETFs, with a hard rule that at least 25% stays
in bonds.

## How it runs

- `.github/workflows/daily_trade.yml` runs on weekdays at 22:00 UTC (or on
  demand via "Run workflow"): fetches data, trains/updates the model,
  validates it on held-out data, and prints a recommendation.
- `.github/workflows/ci.yml` runs the unit tests on every push/PR — it does
  **not** retrain the model.

## Project layout

- `portfolio_lib.py` — feature engineering, the trading environment, and the
  bond-floor rule. Pure logic, no network calls — this is what's unit tested.
- `daily_updater.py` — the daily entry point: fetches data, trains, validates,
  saves, and prints the recommendation.
- `tests/` — unit tests for `portfolio_lib.py` that run without network access
  or GPU, using synthetic data.
- `model_metadata.json` — small text file tracking when the model was last
  retrained and its holdout backtest score. Committed to git (unlike the
  model binary itself).
- `graham_value_ai.zip` — the trained model. **Not committed to git** — it's
  published as a workflow artifact and as an asset on the `latest-model`
  GitHub release, so daily retraining doesn't bloat repo history with a
  binary diff every day.

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
3. The daily recommendation is generated from the most recent available day,
   not day 0 of the training window.
4. The model is loaded and continues training from the previous day's
   weights (rather than being recreated from scratch every run), and the
   saved file is only overwritten if the new model beats both the
   equal-weight baseline and the previously saved model on a held-out slice.
5. The 25%-bonds-minimum rule now lives in one function
   (`portfolio_lib.apply_bond_floor`), used by both the training environment
   and the final allocation output — previously duplicated and inconsistent.
6. Workflow fixes: stopped retraining on every push (now schedule/manual
   only); added `permissions: contents: write`; added a separate CI workflow
   that runs tests on push/PR; stopped committing the model binary to git
   every day (now a release asset + workflow artifact instead).
