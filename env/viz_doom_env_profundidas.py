import os
import cv2
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from vizdoom import DoomGame, ScreenFormat

class VizDoomGym(Env):
    def __init__(self, render=False, config_path=None):
        super().__init__()
        self.game = DoomGame()
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "defend_the_center.cfg")
        self.game.load_config(config_path)
        # Activar buffers necesarios
        #self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_window_visible(render)
        self.game.init()

        # Observación con 2 canales: edges + depth
        self.observation_shape = (100, 160, 2)
        self.observation_space = Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action, frame_skip=4):
        actions = np.identity(self.action_space.n)
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(frame_skip):
            reward = self.game.make_action(actions[action], 1)
            total_reward += reward
            done = self.game.is_episode_finished()
            if done:
                break

        if self.game.get_state():
            state = self.get_observation(self.game.get_state())
            info = {"info": self.game.get_state().game_variables[0]}
        else:
            state = np.zeros(self.observation_shape, dtype=np.uint8)
            info = {"info": 0}

        return state, total_reward, done, info

    def reset(self):
        self.game.new_episode()
        while self.game.get_state() is None:
            self.game.advance_action()
        return self.get_observation(self.game.get_state())

    def get_observation(self, state):
        rgb = state.screen_buffer
        rgb = np.moveaxis(rgb, 0, -1)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.resize(gray, (160, 100)), 100, 250)
        depth = state.depth_buffer
        depth_norm = cv2.resize(depth, (160, 100))
        depth_norm = ((depth_norm - np.min(depth_norm)) / (np.ptp(depth_norm) + 1e-5) * 255).astype(np.uint8)
        return np.stack([edges, depth_norm], axis=-1)
    def render(self, mode='human'):
        pass
    def close(self):
        print(" Cerrando entorno VizDoom")
        self.game.close()
    def get_game_vars(self):
        state = self.game.get_state()
        return state.game_variables if state else [0, 0]