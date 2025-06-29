SEED = 42
import numpy as np
import random
np.random.seed(SEED)
random.seed(SEED)

import torch
torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

import pandas as pd
df = pd.read_csv("./intermediate/ETHUSDT_1h_indicators.csv")
df.set_index('Date', inplace=True)
prices = df['Close'].values
atr = df['atr'].values

from utils import min_max_scal

df_scaled , _ = min_max_scal(df)

split_idx = int(0.9* len(df_scaled))
train_df = df_scaled.iloc[:split_idx]
prices_train = prices[:split_idx]
atr_train =  atr[:split_idx]

val_df = df_scaled.iloc[split_idx:]
prices_val = prices[split_idx:]
print('Shape of Train df',train_df.shape)
print('Shape of Val df',val_df.shape)

from TradingEnv import CustumEnv4
train_env = CustumEnv4(train_df,prices = prices_train,atr= atr_train)
# val_env = CustumEnv3(val_df,prices)

from stable_baselines3 import PPO

policy_kwargs = dict(
    net_arch=dict(
        pi=[32, 16, 8], 
        vf=[32, 16, 8]  
    )
)

agent = PPO(
    "MlpPolicy",
    train_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,
    ent_coef=0.01,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95
)

agent.learn(total_timesteps=20 * train_env.max_steps)
agent.save('./dqn')



