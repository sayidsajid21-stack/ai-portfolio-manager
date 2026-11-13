"""
Core logic for the AI portfolio manager: feature engineering, the trading
environment, and the shared bond-floor rule.

Kept separate from rebalance.py so it can be unit-tested without network
access or a full PPO training run.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

EQUITY_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech/Comm
    'JNJ', 'UNH', 'LLY',                      # Healthcare
    'JPM', 'V', 'BAC',                        # Financials
    'XOM', 'CVX',                             # Energy
    'WMT', 'PG', 'KO'                         # Consumer Defensive
]

BOND_TICKERS = [
    'AGG',  # US Aggregate Bond ETF
    'TLT',  # Long Term Treasury
    'SHV'   # Short Treasury / Cash Equivalent
]

ALL_TICKERS = EQUITY_TICKERS + BOND_TICKERS

BOND_FLOOR = 0.25  # bonds must be at least this fraction of the portfolio

# This system targets long-term investing, not day trading: it should hold
# positions and only change them when there's a real reason to, not chase
# daily noise. TURNOVER_COST_RATE penalizes the model for changing weights
# day to day (modeling real-world trading costs — spread, slippage, and for
# a taxable account, short-term capital gains), so the trained policy is
# rewarded for being patient rather than for trading often. It's expressed
# in the same reward units as `portfolio_return * 100` in BroadMarketArena —
# 1.0 total turnover (fully rebalancing the whole portfolio) costs roughly
# as much reward as a 1% daily move, which is deliberately a strong
# deterrent against frequent trading.
TURNOVER_COST_RATE = 1.0

# Trading-day lookback windows for longer-horizon trend features. Day
# trading cares about yesterday's return; long-term investing cares more
# about the trend over the last few months.
MOMENTUM_WINDOWS = {'mom_3m': 63, 'mom_6m': 126}

# A realistic estimate of round-trip trading cost (spread + slippage) as a
# fraction of the traded amount, used in the backtests below to get a fair
# after-cost comparison. This is a dollar-cost estimate, unlike
# TURNOVER_COST_RATE above, which is a training reward-shaping constant,
# not a cost estimate — the two serve different purposes and are
# deliberately not the same number.
TRANSACTION_COST_BPS = 0.0005  # 5 basis points per unit of turnover


def apply_bond_floor(raw_weights: np.ndarray, equity_idx, bond_idx, floor: float = BOND_FLOOR) -> np.ndarray:
    """
    Enforce a minimum bond allocation by rescaling the equity and bond
    sub-portfolios, each keeping its own internal proportions.

    This is the single place that implements the "at least `floor` in
    bonds" rule. Both the training environment and the final allocation
    report must call this instead of duplicating the rescaling logic.
    """
    weights = raw_weights.copy()
    total_bond = weights[bond_idx].sum()
    total_equity = weights[equity_idx].sum()

    if total_bond < floor and total_bond > 0:
        bond_multiplier = floor / total_bond
        equity_multiplier = (1.0 - floor) / total_equity
        weights[bond_idx] *= bond_multiplier
        weights[equity_idx] *= equity_multiplier
    elif total_bond == 0:
        # Degenerate case: model put zero weight on every bond. Force the
        # floor directly rather than dividing by zero.
        weights[bond_idx] = floor / len(bond_idx)
        weights[equity_idx] *= (1.0 - floor) / total_equity

    return weights


def softmax_weights(action: np.ndarray) -> np.ndarray:
    exp_action = np.exp(action - np.max(action))
    return exp_action / np.sum(exp_action)


def build_features(price_close: pd.DataFrame, fundamentals: dict, all_tickers) -> pd.DataFrame:
    """
    Build the per-day feature table used as model input.

    `price_close` is a DataFrame of adjusted close prices indexed by date,
    one column per ticker in `all_tickers`.

    `fundamentals` maps ticker -> *either*:
      - a dict with keys 'pe', 'p_to_iv', 'div', each a single constant
        snapshot value, broadcast across every row (the Phase 0
        behavior — used when no point-in-time history is available for
        that ticker, e.g. a bond ETF with no SEC filings); or
      - a pandas DataFrame with columns 'pe', 'p_to_iv', 'div' indexed by
        date, giving the real point-in-time value that actually applied
        on each date (see edgar_fundamentals.py, Phase 1). This is the
        preferred mode for equities: it replaces today's valuation being
        smeared backward across the whole training history with the P/E
        and intrinsic-value ratio that was actually knowable on each past
        date.

    Returns a DataFrame with columns `{ticker}_ret`, `{ticker}_pe`,
    `{ticker}_p_to_iv`, `{ticker}_div`, `{ticker}_mom_3m`, `{ticker}_mom_6m`
    for every ticker. The momentum columns are trailing 3-month and
    6-month returns (`MOMENTUM_WINDOWS`) — long-term investing cares about
    the trend over the last few months, not just yesterday's move. Rows
    near the start of the price history that don't yet have a full
    6-month lookback are dropped, since those windows would be undefined.
    """
    returns = price_close.pct_change().dropna()
    momentum = {
        name: price_close.pct_change(window) for name, window in MOMENTUM_WINDOWS.items()
    }

    feature_dfs = []
    for ticker in all_tickers:
        f = fundamentals.get(ticker, {})
        if isinstance(f, pd.DataFrame):
            # Point-in-time mode: reindex onto the returns index (the
            # DataFrame may cover a wider date range than the usable
            # return rows) and forward-fill small gaps from, e.g.,
            # differing holiday calendars between the fundamentals and
            # price sources.
            aligned = f.reindex(returns.index).ffill()
            pe_col = aligned['pe']
            p_to_iv_col = aligned['p_to_iv']
            div_col = aligned['div']
        else:
            pe_col = pd.Series(f.get('pe', 20.0), index=returns.index)
            p_to_iv_col = pd.Series(f.get('p_to_iv', 1.0), index=returns.index)
            div_col = pd.Series(f.get('div', 0.0), index=returns.index)

        cols = {
            f'{ticker}_ret': returns[ticker],
            f'{ticker}_pe': pe_col,
            f'{ticker}_p_to_iv': p_to_iv_col,
            f'{ticker}_div': div_col,
        }
        for name in MOMENTUM_WINDOWS:
            cols[f'{ticker}_{name}'] = momentum[name][ticker]
        feature_dfs.append(pd.DataFrame(cols))

    full = pd.concat(feature_dfs, axis=1)
    # Drop the leading rows where the longest momentum window isn't full yet.
    return full.dropna()


def make_lagged_observations(full_features: pd.DataFrame) -> pd.DataFrame:
    """
    Shift the feature table forward by one day so that the observation
    used to choose day t's weights only contains information that was
    actually known at the close of day t-1.

    Without this shift, the environment hands the model the same day's
    realized return as part of its observation before asking it to
    allocate for that day — the model can "see the future" during
    training and learns to exploit it, which cannot work on live data.

    The reward for day t still uses day t's real return (from
    `full_features`, unshifted) — only the *observation* is lagged. The
    first row has no prior day to lag from and is dropped.
    """
    lagged = full_features.shift(1)
    # Keep rows where the lagged observation is defined, dropping the
    # first row.
    valid_index = lagged.dropna().index
    return lagged.loc[valid_index], full_features.loc[valid_index]


class BroadMarketArena(gym.Env):
    """
    A portfolio-allocation environment with a 25% minimum bond
    constraint.

    `obs_df` gives the observation at each step (already lagged — it must
    not contain information from the day being traded). `reward_df` gives
    the real returns/fundamentals for that same day, used only to compute
    the reward after the action is taken.

    `turnover_cost_rate` penalizes changing weights day to day (see
    TURNOVER_COST_RATE), so the trained policy favors holding positions
    over frequent rebalancing — this system is for long-term investing,
    not day trading.
    """

    def __init__(self, obs_df: pd.DataFrame, reward_df: pd.DataFrame, equity_tickers, bond_tickers,
                 turnover_cost_rate: float = TURNOVER_COST_RATE):
        super().__init__()
        self.turnover_cost_rate = turnover_cost_rate
        assert len(obs_df) == len(reward_df)
        assert list(obs_df.columns) == list(reward_df.columns)

        self.obs_data = obs_df.values
        self.reward_data = reward_df.values
        self.all_tickers = equity_tickers + bond_tickers
        self.n_assets = len(self.all_tickers)
        self.n_steps = len(self.obs_data)

        self.equity_idx = list(range(len(equity_tickers)))
        self.bond_idx = list(range(len(equity_tickers), self.n_assets))

        self.idx_ret = [reward_df.columns.get_loc(f"{t}_ret") for t in self.all_tickers]
        self.idx_pe = [reward_df.columns.get_loc(f"{t}_pe") for t in self.all_tickers]
        self.idx_p_to_iv = [reward_df.columns.get_loc(f"{t}_p_to_iv") for t in self.all_tickers]
        self.idx_div = [reward_df.columns.get_loc(f"{t}_div") for t in self.all_tickers]

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_data.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_weights = None
        obs = self.obs_data[self.current_step].astype(np.float32)
        return obs, {}

    def step(self, action):
        raw_weights = softmax_weights(action)
        final_weights = apply_bond_floor(raw_weights, self.equity_idx, self.bond_idx)

        daily_returns = self.reward_data[self.current_step, self.idx_ret]
        pe_ratios = self.reward_data[self.current_step, self.idx_pe]
        p_to_iv = self.reward_data[self.current_step, self.idx_p_to_iv]
        div_yields = self.reward_data[self.current_step, self.idx_div]

        portfolio_return = np.sum(final_weights * daily_returns)

        graham_penalty = 0.0
        dividend_bonus = 0.0
        for i in self.equity_idx:
            weight = final_weights[i]
            if weight > 0:
                if pe_ratios[i] > 22.0:
                    graham_penalty += (pe_ratios[i] - 22.0) * weight * 0.0005
                if p_to_iv[i] > 1.0:
                    graham_penalty += ((p_to_iv[i] - 1.0) ** 2) * weight * 0.01
                dividend_bonus += div_yields[i] * weight * 0.0001

        # Penalize changing weights from the previous day so the model
        # favors holding positions over frequent rebalancing (see
        # TURNOVER_COST_RATE docstring). Turnover is the L1 distance
        # between today's and yesterday's weights; 2.0 is the max possible
        # (moving 100% of the portfolio to entirely different assets).
        if self.prev_weights is None:
            turnover_cost = 0.0  # first allocation isn't a "change"
        else:
            turnover = np.abs(final_weights - self.prev_weights).sum()
            turnover_cost = turnover * self.turnover_cost_rate
        self.prev_weights = final_weights

        net_reward = (portfolio_return * 100) - graham_penalty + dividend_bonus - turnover_cost

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        obs = (
            self.obs_data[self.current_step].astype(np.float32)
            if not done
            else np.zeros(self.observation_space.shape[0], dtype=np.float32)
        )

        return obs, float(net_reward), done, False, {"weights": final_weights, "portfolio_return": portfolio_return}


def equal_weight_backtest(reward_df: pd.DataFrame, all_tickers, equity_tickers, bond_tickers, floor: float = BOND_FLOOR) -> float:
    """
    Cumulative return of a static baseline: equal weight within equities
    and within bonds, rescaled to respect the same bond floor the RL
    model is required to respect. Used to check the trained model is
    actually worth deploying before it overwrites the saved model file.
    """
    n_equity = len(equity_tickers)
    n_bond = len(bond_tickers)
    equity_idx = list(range(n_equity))
    bond_idx = list(range(n_equity, n_equity + n_bond))

    # Equal weight within each sleeve, with bonds pinned at exactly the
    # floor and equities taking the rest — this already sums to 1, so it
    # is constructed directly rather than routed through apply_bond_floor
    # (which only rescales an existing, possibly differently-scaled,
    # weight vector).
    weights = np.zeros(len(all_tickers))
    weights[equity_idx] = (1.0 - floor) / n_equity
    weights[bond_idx] = floor / n_bond

    idx_ret = [reward_df.columns.get_loc(f"{t}_ret") for t in all_tickers]
    returns = reward_df.iloc[:, idx_ret].values
    daily_portfolio_returns = returns @ weights
    cumulative = np.prod(1 + daily_portfolio_returns) - 1
    return float(cumulative)


def policy_backtest(model, obs_df: pd.DataFrame, reward_df: pd.DataFrame, equity_tickers, bond_tickers,
                     floor: float = BOND_FLOOR, transaction_cost_bps: float = TRANSACTION_COST_BPS) -> float:
    """
    Cumulative, after-cost return of a trained model's (deterministic)
    policy over a held-out slice, applying the same bond floor. Used to
    compare a freshly trained model against both the baseline and the
    previously saved model before deciding whether to overwrite the saved
    file.

    A realistic trading cost (`transaction_cost_bps` per unit of
    turnover) is deducted each day the model changes its weights, so a
    model that trades often doesn't look better than it would in
    practice — this system targets long-term, low-turnover investing.
    """
    n_equity = len(equity_tickers)
    equity_idx = list(range(n_equity))
    bond_idx = list(range(n_equity, n_equity + len(bond_tickers)))
    all_tickers = equity_tickers + bond_tickers
    idx_ret = [reward_df.columns.get_loc(f"{t}_ret") for t in all_tickers]

    cumulative = 1.0
    prev_weights = None
    for i in range(len(obs_df)):
        obs = obs_df.iloc[i].values.astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        raw_weights = softmax_weights(action)
        weights = apply_bond_floor(raw_weights, equity_idx, bond_idx, floor)
        day_return = reward_df.iloc[i, idx_ret].values @ weights
        cost = 0.0 if prev_weights is None else np.abs(weights - prev_weights).sum() * transaction_cost_bps
        prev_weights = weights
        cumulative *= (1 + day_return - cost)
    return float(cumulative - 1)
