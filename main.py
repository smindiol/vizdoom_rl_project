import yaml
import torch
import threading
from trainer import DQNTrainer
import argparse
import os


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
 
#  Crear entorno
scenario_path = cfg["env"]["scenario_path"]
input_shape = tuple(cfg["env"]["input_shape"])
if input_shape[0] == 2:
    from env.viz_doom_env_profundidas import VizDoomGym
else:
    from env.vizdoom_env import VizDoomGym

env = VizDoomGym(render=cfg["env"]["render"], config_path=scenario_path)

#  Parámetros
n_actions = cfg["env"]["actions"]
#  Preparar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Dispositivo actual: {device}")

model_type = cfg.get("model", {}).get("type", "dqn")
num_game_vars = cfg.get("model", {}).get("num_game_vars", 0)

# Selección de clase
if model_type == "dqn_GRU":
    from models.dqn_GRU import RecurrentDQNWithAttention as SelectedModel
    policy_net = SelectedModel(input_shape, num_game_vars, n_actions).to(device)
    target_net = SelectedModel(input_shape, num_game_vars, n_actions).to(device)
elif model_type == "dqn_att":
    from models.dqn_att import DQNWithAttention as SelectedModel
    policy_net = SelectedModel(input_shape, n_actions).to(device)
    target_net = SelectedModel(input_shape, n_actions).to(device)
else:
    raise ValueError(f"Modelo no reconocido: {model_type}")

target_net.eval()
#  Verificación y diagnóstico
print(f" policy_net en: {next(policy_net.parameters()).device}")
print(f" target_net en: {next(target_net.parameters()).device}")
param_device = next(policy_net.parameters()).device
#assert param_device.type == device.type, f" policy_net está en {param_device}, esperado en {device}"
#  Preparar trainer
trainer = DQNTrainer(env, policy_net, target_net, cfg)

trainer.save_plot()
trainer.train()
