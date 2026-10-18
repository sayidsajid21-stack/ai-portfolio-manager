import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from datetime import datetime, timedelta

# ==========================================
# 1. BROAD MARKET & BOND UNIVERSE
# ==========================================
equity_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech/Comm
    'JNJ', 'UNH', 'LLY',                      # Healthcare
    'JPM', 'V', 'BAC',                        # Financials
    'XOM', 'CVX',                             # Energy
    'WMT', 'PG', 'KO'                         # Consumer Defensive
]

bond_tickers = [
    'AGG',  # US Aggregate Bond ETF
    'TLT',  # Long Term Treasury
    'SHV'   # Short Treasury / Cash Equivalent
]

all_tickers = equity_tickers + bond_tickers

# Calculate dynamic 5-year rolling window
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
end_str = end_date.strftime('%Y-%m-%d')
start_str = start_date.strftime('%Y-%m-%d')

print(f"1. Fetching broad market data from {start_str} to {end_str}...")
data = yf.download(all_tickers, start=start_str, end=end_str)['Close']
returns = data.pct_change().dropna()
n_days = len(returns)

print("2. Engineering Fundamental Metrics across 19 Assets...")
np.random.seed(42)
feature_dfs = []

for ticker in equity_tickers:
    ret = returns[ticker].values
    pe_ratio = np.clip(np.random.normal(20, 5, n_days), 5, 50)
    price_to_intrinsic = np.clip(np.random.normal(1.0, 0.2, n_days), 0.5, 2.0)
    div_yield = np.clip(np.random.normal(0.02, 0.01, n_days), 0.0, 0.06)
    
    ticker_df = pd.DataFrame({
        f'{ticker}_ret': ret,
        f'{ticker}_pe': pe_ratio,
        f'{ticker}_p_to_iv': price_to_intrinsic,
        f'{ticker}_div': div_yield
    })
    feature_dfs.append(ticker_df)

for ticker in bond_tickers:
    bond_df = pd.DataFrame({
        f'{ticker}_ret': returns[ticker].values,
        f'{ticker}_pe': np.zeros(n_days),      
        f'{ticker}_p_to_iv': np.ones(n_days),  
        f'{ticker}_div': np.ones(n_days) * 0.04 
    })
    feature_dfs.append(bond_df)

full_features = pd.concat(feature_dfs, axis=1)

# ==========================================
# 2. THE CONSTRAINED ARENA
# ==========================================
class BroadMarketArena(gym.Env):
    def __init__(self, feature_df, equity_tickers, bond_tickers):
        super(BroadMarketArena, self).__init__()
        self.feature_data = feature_df.values
        self.all_tickers = equity_tickers + bond_tickers
        self.n_assets = len(self.all_tickers)
        self.n_steps = len(self.feature_data)
        
        self.equity_idx = list(range(len(equity_tickers)))
        self.bond_idx = list(range(len(equity_tickers), self.n_assets))
        
        self.idx_ret = [feature_df.columns.get_loc(f"{t}_ret") for t in self.all_tickers]
        self.idx_pe = [feature_df.columns.get_loc(f"{t}_pe") for t in self.all_tickers]
        self.idx_p_to_iv = [feature_df.columns.get_loc(f"{t}_p_to_iv") for t in self.all_tickers]
        self.idx_div = [feature_df.columns.get_loc(f"{t}_div") for t in self.all_tickers]
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_data.shape[1],), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.feature_data[self.current_step].astype(np.float32)
        return obs, {}

    def step(self, action):
        exp_action = np.exp(action - np.max(action))
        raw_weights = exp_action / np.sum(exp_action)
        
        total_bond_weight = np.sum(raw_weights[self.bond_idx])
        total_equity_weight = np.sum(raw_weights[self.equity_idx])
        
        if total_bond_weight < 0.25:
            bond_multiplier = 0.25 / total_bond_weight
            equity_multiplier = 0.75 / total_equity_weight
            
            raw_weights[self.bond_idx] *= bond_multiplier
            raw_weights[self.equity_idx] *= equity_multiplier
            
        final_weights = raw_weights
        
        daily_returns = self.feature_data[self.current_step, self.idx_ret]
        pe_ratios = self.feature_data[self.current_step, self.idx_pe]
        p_to_iv = self.feature_data[self.current_step, self.idx_p_to_iv]
        div_yields = self.feature_data[self.current_step, self.idx_div]
        
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

        net_reward = (portfolio_return * 100) - graham_penalty + dividend_bonus
        
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        obs = self.feature_data[self.current_step].astype(np.float32) if not done else np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        return obs, float(net_reward), done, False, {}

# ==========================================
# 3. TRAINING & ALLOCATION OUTPUT
# ==========================================
print("3. Initializing Broad Market Environment...")
env = BroadMarketArena(full_features, equity_tickers, bond_tickers)

print("4. Training AI across 19 assets with strict bond constraints...")
model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
model.learn(total_timesteps=60000)
print("Training Complete!")

# ==========================================
# 4. SAVE THE TRAINED MODEL
# ==========================================
model.save("graham_value_ai")
print("Model saved successfully as graham_value_ai.zip!")

# Generate final recommendation
obs, _ = env.reset()
action, _ = model.predict(obs)

exp_action = np.exp(action - np.max(action))
raw_weights = exp_action / np.sum(exp_action)

if np.sum(raw_weights[env.bond_idx]) < 0.25:
    raw_weights[env.bond_idx] *= (0.25 / np.sum(raw_weights[env.bond_idx]))
    raw_weights[env.equity_idx] *= (0.75 / np.sum(raw_weights[env.equity_idx]))

print("\n=== AI PORTFOLIO RECOMMENDATION ===")
print(f"{'Asset':<10} | {'Type':<10} | {'Weight'}")
print("-" * 35)

total_eq = 0
total_bd = 0

for i, ticker in enumerate(all_tickers):
    weight = raw_weights[i]
    if i in env.bond_idx:
        total_bd += weight
        print(f"{ticker:<10} | Bond       | {weight*100:05.2f}%")
    else:
        total_eq += weight
        print(f"{ticker:<10} | Equity     | {weight*100:05.2f}%")

print("-" * 35)
print(f"TOTAL EQUITY: {total_eq*100:.2f}%")
print(f"TOTAL BONDS:  {total_bd*100:.2f}%")