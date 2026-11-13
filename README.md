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
- `edgar_fundamentals.py` — fetches point-in-time fundamentals (P/E,
  price-to-book, dividend yield) for equities from SEC EDGAR's free XBRL
  API. See "Point-in-time fundamentals" below.
- `backtest_lib.py` — performance metrics (CAGR, Sharpe, Sortino, max
  drawdown, turnover), four static baseline strategies, and walk-forward
  date splitting. Pure logic, no network calls — unit tested.
- `walk_forward_backtest.py` — run manually (`python
  walk_forward_backtest.py`) to check whether the RL model is actually
  worth running: trains a fresh model on each rolling window and scores
  it against the baselines on the following held-out window. See
  "Walk-forward backtesting" below.
- `rebalance.py` — the monthly entry point: fetches data, trains, validates,
  saves, and prints the recommendation.
- `tests/` — unit tests that run without network access or GPU, using
  synthetic data and a fake SEC response fixture.
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

## Point-in-time fundamentals (Phase 1)

Equities now get real point-in-time fundamentals history from SEC EDGAR's
free XBRL company-facts API, via `edgar_fundamentals.py`, instead of a
single current snapshot broadcast across the whole training history. Each
historical date gets the P/E, price-to-book, and dividend yield that were
actually knowable *on that date* — computed from each filing's `filed`
date (when it became public), not its `end` date (the fiscal period it
covers), so there's no lookahead.

- **Setup**: SEC's Fair Access policy requires a descriptive User-Agent
  with a contact email on every request. Set a repository variable named
  `EDGAR_USER_AGENT` (Settings -> Secrets and variables -> Actions ->
  Variables) to something like `"ai-portfolio-manager you@example.com"`.
  It'll work with a generic placeholder if you skip this, but SEC asks
  that you don't.
- **Fallback**: bond ETFs have no SEC equity filings, so they always use
  the Phase 0 constant-snapshot approach (via yfinance) — this is fine
  since the bond-floor rule doesn't use fundamentals at all, and only
  equities get the Graham-style P/E/intrinsic-value penalty. If EDGAR
  lookup fails for a specific equity (rate limit, no CIK found, missing
  tag), that ticker also falls back to the constant-snapshot approach
  rather than failing the whole run.
- **Known simplifications** (documented in `edgar_fundamentals.py`): EPS
  uses the annual (10-K) figure only, updated once a year, rather than a
  rolling trailing-twelve-month figure from quarterly 10-Qs; dividend
  yield annualizes the most recently declared quarterly dividend (×4)
  rather than summing the actual trailing four quarters.

## Walk-forward backtesting (Phase 2)

`walk_forward_backtest.py` answers the question the monthly workflow
doesn't: is this strategy actually any good, over several multi-year
train/test windows, against real alternatives — not just "did this
month's model beat last month's model" (which is all `rebalance.py`'s
save-gate checks).

- **Method**: rolling 3-year train / 1-year test windows
  (`backtest_lib.walk_forward_splits`), rolled forward a year at a time
  across the whole price history. A fresh model is trained per fold —
  this is slow (several PPO training runs), so it's a manual/dev tool,
  not part of the monthly production workflow.
- **Baselines**: equal weight, the classic 60/40 equities/bonds split,
  inverse-volatility (risk parity style), and a simple Graham-style value
  screen (equal-weight only the equities whose current P/E is below a
  threshold). All four respect the same 25%-bonds floor the RL model is
  held to.
- **Metrics**: CAGR, Sharpe, Sortino (downside-only, which fits a
  long-term lens better than Sharpe's penalizing-all-volatility), max
  drawdown, and average daily turnover.
- **Verdict**: the script's summary states plainly whether the model beat
  the 60/40 baseline on average across folds — a strategy that doesn't
  shouldn't be treated as a real recommendation.

## Known limitations (tracked, not yet fixed)

- **No news/sentiment input yet** (planned Phase 4).
- **Recommendations only — no live or paper trading wired up** (planned
  Phase 6, via Alpaca's paper-trading API).

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
