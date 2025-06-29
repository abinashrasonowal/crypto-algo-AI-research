
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn

class HuberDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.SmoothL1Loss()


from stable_baselines3 import DQN
import numpy as np

class CustomDQN(DQN):
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if deterministic or self.exploration_rate == 0.0:
            return super().predict(observation, state, episode_start, deterministic=True)

        if np.random.rand() < self.exploration_rate:
            valid_actions = list(range(self.action_space.n))
            
            shares = observation[0][-1]

            if shares >= 20 and 1 in valid_actions:
                valid_actions.remove(1)
            
            if shares <=0 and 2 in valid_actions:
                valid_actions.remove(2)

            action = np.array([np.random.choice(valid_actions)])
            return action, state
        else:
            return super().predict(observation, state, episode_start, deterministic=False)
