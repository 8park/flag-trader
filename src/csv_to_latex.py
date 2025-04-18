import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

INIT_CASH = 10_000.0

class MarketEnv(gym.Env):
    """Trading environment for FLAG-TRADER pilot."""
    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path, index_col=0)
        self.n_steps = len(self.df)
        self.action_space = spaces.Discrete(3)  # Buy/Sell/Hold
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.reset()

    def _get_obs(self):
        row = self.df.iloc[self.t]
        return np.array([row["Close"], row["Volume"], row["RSI"], self.cash, self.shares], dtype=np.float32)

    def step(self, action: int):
        price = self.df.iloc[self.t]["Close"]
        if action == 0:  # Buy all
            max_shares = int(self.cash // price)
            self.shares += max_shares
            self.cash -= max_shares * price
        elif action == 1:  # Sell all
            self.cash += self.shares * price
            self.shares = 0
        self.t += 1
        done = self.t >= self.n_steps - 1
        next_price = self.df.iloc[self.t]["Close"] if not done else price
        net_worth = self.cash + self.shares * next_price
        reward = net_worth - self.init_net_worth
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = INIT_CASH
        self.shares = 0
        self.t = 0
        self.init_net_worth = INIT_CASH
        return self._get_obs(), {}
