# env/vizdoom_env.py

from vizdoom import DoomGame
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import cv2
import os

class VizDoomGym(Env): 
    def __init__(self, render=False): 
        super().__init__()
        self.game = DoomGame()
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "defend_the_center.cfg")
        self.game.load_config(config_path)
        
        self.game.set_window_visible(render)
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8) 
        self.action_space = Discrete(3)
        
    def step(self, action):
        actions = np.identity(3)
        reward = self.game.make_action(actions[action], 4) 
        
        if self.game.get_state(): 
            state = self.grayscale(self.game.get_state().screen_buffer)
            info = {"info": self.game.get_state().game_variables[0]}
        else: 
            state = np.zeros(self.observation_space.shape)
            info = {"info": 0}
        
        done = self.game.is_episode_finished()
        return state, reward, done, info 
    
    def reset(self): 
        self.game.new_episode()
        return self.grayscale(self.game.get_state().screen_buffer)
    
    def render(self, mode='human'): 
        pass
    
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (100,160,1))
    
    def close(self): 
        self.game.close()
