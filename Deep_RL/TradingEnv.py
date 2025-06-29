import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CustumEnv(gym.Env):
    def __init__(self, df, render_mode='human'):
        super().__init__()
        self.prices = df['Close'].values
        self.features = df.values
        self.index = df.index
        self.fees_rate = 0.0002
        self.max_steps = len(self.prices) - 1

        self.balance = 20
        self.current_step = 0
        self.profit = 0
        self.no_shares = 0
        self.avg_buy_price = 0
        self.unrealized_pnl = 0

        self.cnt_buy = 0
        self.cnt_hold = 0
        
        self.state = []
        self.action_space = spaces.Discrete(3)  # Actions: 0 - hold, 1 - buy, 2 - sell
        self.observation_space = spaces.Box(
            low=np.full(self.features.shape[1] + 1, -np.inf, dtype=np.float32),
            high=np.full(self.features.shape[1] + 1, np.inf, dtype=np.float32),
            dtype=np.float32
        )
        self.seq_len = 1
        self.render_mode = render_mode

        self.reset()

    def reset(self, *, seed=42):
        super().reset(seed=seed)
        if seed is not None:
          np.random.seed(seed)
          random.seed(seed)

        self.balance = 20
        self.current_step = 0
        self.no_shares = 0
        self.profit = 0
        self.avg_buy_price = 0
        self.unrealized_pnl = 0

        self.cnt_buy = 0
        self.cnt_hold = 0

        self.state = []

        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.append(self.features[self.current_step], [self.no_shares])
        # self.state.append(obs)
        # self.state = self.state[-self.seq_len:]
        # padded_state = np.pad(self.state, ((self.seq_len - len(self.state), 0), (0, 0)), mode='constant')
        # return np.array(padded_state, dtype=np.float32)
        return obs.astype(np.float32)


    def step(self, action):
        done = False
        current_price = self.prices[self.current_step]

        reward = 0.0  
        pnl = 0.0

        if action == 1 :  # buy
            if self.no_shares < 20:
                self.no_shares += 1
                self.profit -= current_price * (self.fees_rate)
                self.avg_buy_price = (self.avg_buy_price * (self.no_shares - 1) + current_price * (1 + self.fees_rate)) / self.no_shares
                self.balance -= current_price * (1 + self.fees_rate)

        elif action == 2 :  #  sell
            if self.no_shares > 0:
                self.no_shares -= 1
                pnl = current_price * (1 - self.fees_rate) - self.avg_buy_price
                self.profit+=pnl
                self.balance += current_price * (1 - self.fees_rate)
                if self.no_shares == 0:
                    self.avg_buy_price = 0

        self.unrealized_pnl = self.no_shares * (current_price * (1 - self.fees_rate) - self.avg_buy_price)

        reward +=  (pnl + 0.1*self.unrealized_pnl)*10

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, no of shares: {self.no_shares}, "
            f"profit: {self.profit + self.unrealized_pnl:.2f} (realized: {self.profit:.2f}, unrealized: {self.unrealized_pnl:.2f})")


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CustumEnv2(gym.Env):
    def __init__(self, df, render_mode='human'):
        super().__init__()
        self.prices = df['Close'].values
        self.features = df.values
        self.index = df.index
        self.fees_rate = 0.0002
        self.max_steps = len(self.prices) - 1

        self.action_space = spaces.Discrete(3)  # 0 - hold, 1 - buy, 2 - sell
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.features.shape[1],),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.reset()

    def reset(self, *, seed=42):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_step = 0
        self.profit = 0
        self.position = 0  # 0 - neutral, 1 - long, 2 - short
        self.enty_price = 0

        return self._get_obs(), {}

    def map_action(self, action):
        action_map = {
            (0, 0): (0, 0),  # hold
            (0, 1): (1, 1),  # open long
            (0, 2): (2, 2),  # open short
            (1, 0): (1, 0),  # hold long
            (1, 1): (1, 0),  # hold long
            (1, 2): (0, 3),  # close long
            (2, 0): (2, 0),  # hold short
            (2, 1): (0, 4),  # close short
            (2, 2): (2, 0)   # hold short
        }
        return action_map.get((self.position, action), (0, 0))

    def _get_obs(self):
        return self.features[self.current_step].astype(np.float32)

    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_obs(), 0, True, False, {}

        curr_price = self.prices[self.current_step]
        next_price = self.prices[self.current_step + 1]
        price_diff = next_price - curr_price

        reward = 0
        if action == 1:
            reward = 1 if price_diff > 0 else -1
        elif action == 2:
            reward = 1 if price_diff < 0 else -1

        self.position, action_taken = self.map_action(action)

        if action_taken == 1:  # open long
            self.profit -= curr_price * self.fees_rate
            self.enty_price = curr_price * (1 + self.fees_rate)

        elif action_taken == 3:  # close long
            self.profit += (curr_price * (1 - self.fees_rate) - self.enty_price)

        elif action_taken == 2:  # open short
            self.profit -= curr_price * self.fees_rate
            self.enty_price = curr_price * (1 - self.fees_rate)

        elif action_taken == 4:  # close short
            self.profit += (self.enty_price - curr_price * (1 - self.fees_rate))

        self.current_step += 1
        obs = self._get_obs()
        done = self.current_step >= self.max_steps

        return obs, reward, done, False, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Profit: {self.profit:.2f}")



import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CustumEnv3(gym.Env):
    def __init__(self, df, prices ,render_mode='human'):
        super().__init__()
        self.prices = prices
        self.features = df.values
        self.index = df.index
        self.fees_rate = 0.0002
        self.max_steps = len(prices) - 1

        self.pos_reward = 0
        self.neg_reward = 0 
        self.current_step  = 0

        self.action_space = spaces.Discrete(3)  # 0 - hold, 1 - buy, 2 - sell
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.features.shape[1],),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.reset()

    def reset(self, *, seed=42):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.pos_reward = 0
        self.neg_reward = 0 
        self.current_step = 0
        # self.current_step  = np.random.randint(0,self.max_steps)

        return self._get_obs(), {}

    def _get_obs(self):
        return self.features[self.current_step].astype(np.float32)

    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_obs(), 0, True, False, {}

        curr_price = self.prices[self.current_step]

        reward = 0

        
        if action == 0 :  # BUY 2%
            tp = curr_price * (1 + 0.001)
            sl = curr_price * (1 - 0.005)
            reward = - 0.2
            for i in range(self.current_step + 1, min(self.current_step + 720, self.max_steps)):
                if self.prices[i] <= sl:
                    break
                if self.prices[i] >= tp:
                    reward = 1
                    break 
        elif action == 1:  # SELL 2%
            tp = curr_price * (1 - 0.001)
            sl = curr_price * (1 + 0.005)
            reward = - 0.2
            for i in range(self.current_step + 1, min(self.current_step + 720, self.max_steps)):
                if self.prices[i] >= sl:
                    break
                if self.prices[i] <= tp:
                    reward = 1
                    break  

        if reward == 1:
            self.pos_reward += 1
        elif reward == -0.2 :
            self.neg_reward += 1
        self.current_step += 1
        obs = self._get_obs()
        done = self.current_step >= self.max_steps

        return obs, reward, done, False, {}

    def render(self, mode="human"):
        accuracy = self.pos_reward/(self.pos_reward + self.neg_reward +1e-8)
        print(f"Step: {self.current_step}, accuracy: {accuracy:.2f},positive: {self.pos_reward},negative : {self.neg_reward}")




import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CustumEnv4(gym.Env):
    def __init__(self, df, prices, atr ,render_mode='human'):
        super().__init__()
        self.prices = prices
        self.atr = atr
        self.features = df.values
        self.index = df.index
        self.max_steps = len(prices) - 1

        self.win = 0
        self.loss = 0 
        self.current_step  = 0
        self.total_reward = 0
        self.cnt_hold = 0

        self.action_space = spaces.Discrete(3)  # 0 - hold, 1 - buy, 2 - sell
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.features.shape[1],),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.reset()

    def reset(self, *, seed=42):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.pos_reward = 0
        self.neg_reward = 0 
        self.current_step  = np.random.randint(0,self.max_steps-50)
        # self.current_step = 0
        self.total_reward = 0
        self.cnt_hold = 0

        return self.get_obs(), {}

    def get_obs(self):
        return self.features[self.current_step].astype(np.float32)

    def step(self, action):
        if self.current_step >= self.max_steps - 50:
            return self.get_obs(), 0, True, False, {}

        reward = 0
        self.cnt_hold += (1 if action == 2 else 0)

        step_lim = 

        if action == 0:  # BUY
            entry_price = self.curr_price
            max_price = entry_price

            for i in range(self.current_step + 1, step_lim):
                price = self.prices[i]
                atr_val = self.atr[i]
                max_price = max(max_price, price)
                trailing_sl = max_price - 3 * atr_val

                if price <= trailing_sl:
                    exit_price = price
                    reward = (exit_price - entry_price) / entry_price * 10
                    break

            self.cnt_hold = 0

        elif action == 1:  # SELL
            entry_price = curr_price
            min_price = entry_price

            for i in range(self.current_step + 1, step_lim):
                price = self.prices[i]
                atr_val = self.atr[i]
                min_price = min(min_price, price)
                trailing_sl = min_price + 3 * atr_val

                if price >= trailing_sl:
                    exit_price = price
                    reward = (entry_price - exit_price) / entry_price * 10
                    break

            self.cnt_hold = 0

        elif action == 2:  # HOLD
            if self.cnt_hold > 30:
                reward = -0.1  # Penalize excessive inaction


        self.total_reward += reward

        if reward > 0 :
            self.win += 1
        elif reward < - 0.1:
            self.loss += 1
            reward *= 5 
        
        # Increment step after processing action
        self.current_step += 1
        obs = self.get_obs()
        done = self.current_step >= self.max_steps - 50

        return obs, reward, done, False, {}

    def render(self, mode="human"):
        accuracy = self.win/(self.win + self.loss +1e-8)
        print(f"Step: {self.current_step}, accuracy: {accuracy:.2f},positive: {self.win},negative : {self.loss},total reward :{self.total_reward:.2f}")
