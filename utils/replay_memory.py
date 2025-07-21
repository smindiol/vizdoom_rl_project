import random
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_img, action, reward, next_obs_img, done, obs_vars=None, next_obs_vars=None):
        if obs_vars is not None and next_obs_vars is not None:
            self.buffer.append((obs_img, obs_vars, action, reward, next_obs_img, next_obs_vars, done))
        else:
            self.buffer.append((obs_img, action, reward, next_obs_img, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        if len(batch[0]) == 7:
            obs_img, obs_vars, actions, rewards, next_obs_img, next_obs_vars, dones = zip(*batch)
            return (np.array(obs_img),
                    np.array(obs_vars),
                    np.array(actions),
                    np.array(rewards),
                    np.array(next_obs_img),
                    np.array(next_obs_vars),
                    np.array(dones))
        else:
            obs_img, actions, rewards, next_obs_img, dones = zip(*batch)
            return (np.array(obs_img),
                    np.array(actions),
                    np.array(rewards),
                    np.array(next_obs_img),
                    np.array(dones))

    def __len__(self):
        return len(self.buffer)
