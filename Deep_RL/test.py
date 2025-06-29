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

split_idx = int(0.7* len(df_scaled))
train_df = df_scaled.iloc[:split_idx]
prices_train = prices[:split_idx]
atr_train =  atr[:split_idx]

val_df = df_scaled.iloc[split_idx:]
prices_val = prices[split_idx:]
atr_val =  atr[:split_idx]
print('Shape of Train df',train_df.shape)
print('Shape of Val df',val_df.shape)


from TradingEnv import CustumEnv4
test_env = CustumEnv4(
        df=train_df,
        prices=prices_train,
        atr=atr_val
    )

from stable_baselines3 import PPO

test_agent = PPO.load('./dqn')
test_agent.set_env(test_env)

from utils import visualize_actions

obs = test_agent.env.reset()
done = False
Date_actions = []

action, _ = test_agent.predict(obs, deterministic=True)
action = int(action.item())
Date_actions.append([test_env.index[test_env.current_step], test_env.prices[test_env.current_step], action,test_env.win - test_env.loss])

while True:
    obs, reward, done, _, _ = test_env.step(action)
    action, _ = test_agent.predict(obs, deterministic=True)
    action = int(action.item())
    Date_actions.append([test_env.index[test_env.current_step], test_env.prices[test_env.current_step], action, test_env.win - test_env.loss])

    if done:
        test_agent.env.render(mode="human")
        break

Date_actions = pd.DataFrame(Date_actions, columns=['Date','Close', 'Action','Profit'])
Date_actions.set_index('Date', inplace=True)
Date_actions.to_csv('./intermediate/Date_actions.csv')

visualize_actions(Date_actions)

actions_Pridicted = Date_actions['Action'].values
print(Date_actions['Action'].value_counts())