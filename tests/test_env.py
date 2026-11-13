import numpy as np
import pandas as pd

from portfolio_lib import BroadMarketArena


def make_env(n_days=10):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    cols = {}
    for t in ["EQ1", "EQ2"]:
        cols[f"{t}_ret"] = np.full(n_days, 0.01)
        cols[f"{t}_pe"] = np.full(n_days, 20.0)
        cols[f"{t}_p_to_iv"] = np.full(n_days, 1.0)
        cols[f"{t}_div"] = np.full(n_days, 0.0)
    for t in ["BD1"]:
        cols[f"{t}_ret"] = np.full(n_days, 0.001)
        cols[f"{t}_pe"] = np.full(n_days, 0.0)
        cols[f"{t}_p_to_iv"] = np.full(n_days, 1.0)
        cols[f"{t}_div"] = np.full(n_days, 0.04)
    df = pd.DataFrame(cols, index=dates)
    return BroadMarketArena(df, df, ["EQ1", "EQ2"], ["BD1"])


def test_env_step_enforces_bond_floor_in_info():
    env = make_env()
    obs, _ = env.reset()
    # action strongly favors equities
    action = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    weights = info["weights"]
    assert weights[2] >= 0.25 - 1e-6  # bond index is 2
    assert weights.sum() == 1.0 or abs(weights.sum() - 1.0) < 1e-6


def test_env_episode_terminates():
    env = make_env(n_days=5)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        assert steps <= 10  # safety bound against infinite loop
    assert done


def test_env_observation_shape_matches_feature_columns():
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape[0] == 12  # 2 equities + 1 bond, 4 features each


def test_env_charges_no_turnover_cost_on_first_step():
    # With no prior weights, the first allocation can't be a "change", so
    # there should be no turnover penalty baked into the first reward.
    env = make_env()
    obs, _ = env.reset()
    action = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    obs, reward_first, done, truncated, info = env.step(action)

    # A second env that starts already holding exactly these weights would
    # see the same portfolio_return/graham/dividend terms but no turnover
    # cost either, since nothing changed — so repeating the identical
    # action on the same env should not cost more than the first step did
    # on this dimension (turnover versus "no prior weights" is 0 either way).
    obs2, reward_repeat, done2, truncated2, info2 = env.step(action)
    assert np.allclose(info["weights"], info2["weights"])
    # Reward should be very close (same weights, same constant features);
    # the only thing that could differ is the turnover cost, which should
    # be ~0 since the weights didn't change between these two steps.
    assert abs(reward_first - reward_repeat) < 1e-3


def test_env_penalizes_changing_weights():
    env = make_env()
    obs, _ = env.reset()
    # First allocation: heavy equities.
    env.step(np.array([1.0, 1.0, -1.0], dtype=np.float32))
    # Second step: flip hard to the opposite allocation — a large change
    # in weights should incur a turnover cost that repeating the same
    # action (no change) would not.
    env_a = make_env()
    env_a.reset()
    env_a.step(np.array([1.0, 1.0, -1.0], dtype=np.float32))
    _, reward_hold, *_ = env_a.step(np.array([1.0, 1.0, -1.0], dtype=np.float32))

    env_b = make_env()
    env_b.reset()
    env_b.step(np.array([1.0, 1.0, -1.0], dtype=np.float32))
    _, reward_switch, *_ = env_b.step(np.array([-1.0, -1.0, 1.0], dtype=np.float32))

    assert reward_switch < reward_hold  # switching costs more than holding
