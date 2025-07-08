# eval.py

import torch
import numpy as np
import time
from env.vizdoom_env import VizDoomGym
from models.dqn import DQN

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar entorno con render activado
env = VizDoomGym(render=True)
input_shape = (1, 100, 160)
n_actions = env.action_space.n

# Cargar red y pesos
policy_net = DQN(input_shape, n_actions).to(device)
policy_net.load_state_dict(torch.load("checkpoints/dqn_best.pth", map_location=device))
policy_net.eval()

def preprocess(obs):
    return torch.tensor(np.moveaxis(obs, -1, 0), dtype=torch.float32).unsqueeze(0).to(device)

# Jugar episodio
obs = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = preprocess(obs)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
        action = q_values.argmax().item()

    obs, reward, done, _ = env.step(action)
    total_reward += reward

    time.sleep(0.04)  # Controlar velocidad de ejecución

env.close()
print(f"Recompensa total del episodio: {total_reward:.2f}")
