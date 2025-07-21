# utils/replay_memory.py

import random
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """
        Guarda una transición. 
        Puede recibir:
        - (obs_img, action, reward, next_obs_img, done)     ← red simple
        - (obs_img, obs_vars, action, reward, next_obs_img, next_obs_vars, done) ← red con game_vars
        """
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        if len(batch[0]) == 5:
            # Red que usa solo imágenes
            obs_img, actions, rewards, next_obs_img, dones = zip(*batch)
            return (
                np.array(obs_img),
                np.array(actions),
                np.array(rewards),
                np.array(next_obs_img),
                np.array(dones),
            )
        elif len(batch[0]) == 7:
            # Red que usa imágenes + variables del juego
            obs_img, obs_vars, actions, rewards, next_obs_img, next_obs_vars, dones = zip(*batch)
            return (
                np.array(obs_img),
                np.array(obs_vars),
                np.array(actions),
                np.array(rewards),
                np.array(next_obs_img),
                np.array(next_obs_vars),
                np.array(dones),
            )
        else:
            raise ValueError("Formato de transición desconocido en el buffer.")

    def __len__(self):
        return len(self.buffer)
