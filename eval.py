# eval.py
import torch
import numpy as np
import time
from env.vizdoom_env import VizDoomGym
import yaml
 
# Cargar configuración
with open("config/config_defend_the_center.yaml", "r") as f:
    cfg = yaml.safe_load(f)
# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Cargar entorno con render activado
scenario_path = cfg["env"]["scenario_path"]
env = VizDoomGym(render=True, config_path=scenario_path)
# Parámetros
input_shape = tuple(cfg["env"]["input_shape"])
n_actions = cfg["env"]["actions"]
model_type = cfg.get("model", {}).get("type", "dqn")
# Selección de modelo
if model_type == "dqn":
    from models.dqn_GRU import DQN as SelectedModel
elif model_type == "dqn_att":
    from models.dqn_att import DQNWithAttention  as SelectedModel
else:
    raise ValueError(f"Modelo no reconocido: {model_type}")

# Cargar red y pesos
policy_net = SelectedModel(input_shape, n_actions).to(device)
import os
model_best_path = os.path.join(cfg["training"]["checkpoint_path"], "dqn_best.pth")
policy_net.load_state_dict(torch.load(model_best_path, map_location=device))
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
