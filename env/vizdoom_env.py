from vizdoom import DoomGame
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import cv2
import os

class VizDoomGym(Env):
    def __init__(self, render=False, config_path=None):
        super().__init__()
        self.game = DoomGame()
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "defend_the_center.cfg")
        self.game.load_config(config_path)

        self.game.set_window_visible(render)
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

        print(" Entorno VizDoom inicializado")
        print(f" Observación esperada: {self.observation_space.shape}")

    def step(self, action, frame_skip=1):
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
            state = self.grayscale(self.game.get_state().screen_buffer)
            info = {"info": self.game.get_state().game_variables[0]}
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = {"info": 0}

        return state, total_reward, done, info

    def reset(self):
        self.game.new_episode()
        state = self.grayscale(self.game.get_state().screen_buffer)
        print(" Episodio reiniciado")
        return state

    def render(self, mode='human'):
        pass

    def grayscale(self, observation):
        # Convertir a escala de grises
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        # Filtro Canny para detección de bordes
        edges = cv2.Canny(resized, 100, 200)
        # Retornar imagen como (100, 160, 1)
        return np.reshape(edges, (100, 160, 1))

    def close(self):
        print(" Cerrando entorno VizDoom")
        self.game.close()
