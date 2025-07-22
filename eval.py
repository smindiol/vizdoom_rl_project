# eval.py
import torch
import numpy as np
import time
import os
import yaml
import argparse

# Argumentos por terminal
parser = argparse.ArgumentParser()
parser.add_argument(
    "config_name",
    nargs="?",
    default="config_defend_the_center_profundidad.yaml",
    help="Archivo YAML de configuración (por defecto: config_defend_the_center.yaml)"
)
args = parser.parse_args()
config_path = os.path.join("config", args.config_name)

# Cargar configuración
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Parámetros generales
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = tuple(cfg["env"]["input_shape"])
n_actions = cfg["env"]["actions"]
model_type = cfg.get("model", {}).get("type", "dqn")
num_game_vars = cfg.get("model", {}).get("num_game_vars", 0)

# Seleccionar entorno
if input_shape[0] == 2:
    from env.viz_doom_env_profundidas import VizDoomGym
else:
    from env.vizdoom_env import VizDoomGym
scenario_path = cfg["env"]["scenario_path"]
env = VizDoomGym(render=True, config_path=scenario_path)

# Selección del modelo
if model_type == "dqn_GRU":
    from models.dqn_GRU import RecurrentDQNWithAttention as SelectedModel
    policy_net = SelectedModel(input_shape, num_game_vars, n_actions).to(device)
elif model_type == "dqn_att":
    from models.dqn_att import DQNWithAttention as SelectedModel
    policy_net = SelectedModel(input_shape, n_actions).to(device)
else:
    raise ValueError(f"Modelo no reconocido: {model_type}")

# Cargar pesos del modelo
model_best_path = os.path.join(cfg["training"]["checkpoint_path"], "dqn_best.pth")
policy_net.load_state_dict(torch.load(model_best_path, map_location=device))
policy_net.eval()

# Preprocesamiento de observación
def preprocess(obs):
    obs_tensor = torch.tensor(np.moveaxis(obs, -1, 0), dtype=torch.float32).unsqueeze(0)
    return obs_tensor.to(device)

# Jugar un episodio
obs = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = preprocess(obs)

    with torch.no_grad():
        if model_type == "dqn_GRU":
            hidden = policy_net.init_hidden(1)
            game_vars_tensor = torch.zeros((1, num_game_vars), device=device)  # inicialización por defecto
            q_values, _ = policy_net(state_tensor, game_vars_tensor, hidden)
        elif model_type == "dqn_att":
            q_values = policy_net(state_tensor)
        action = q_values.argmax().item()

    obs, reward, done, _ = env.step(action)
    total_reward += reward
    time.sleep(0.04)

env.close()
print(f"Recompensa total del episodio: {total_reward:.2f}")
