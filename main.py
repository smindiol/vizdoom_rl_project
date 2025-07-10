import yaml
import torch
from env.vizdoom_env import VizDoomGym
from trainer import DQNTrainer
 
# Cargar configuración
with open("config/config_defend_the_center.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Crear entorno
scenario_path = cfg["env"]["scenario_path"]
env = VizDoomGym(render=cfg["env"]["render"], config_path=scenario_path)

# Parámetros
input_shape = tuple(cfg["env"]["input_shape"])
n_actions = cfg["env"]["actions"]
model_type = cfg.get("model", {}).get("type", "dqn")

# Selección de modelo
if model_type == "dqn":
    from models.dqn import DQN as SelectedModel
elif model_type == "dqn_att":
    from models.dqn_att import DQNWithAttention as SelectedModel
else:
    raise ValueError(f"Modelo no reconocido: {model_type}")

# Preparar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = SelectedModel(input_shape, n_actions).to(device)
target_net = SelectedModel(input_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Entrenamiento
trainer = DQNTrainer(env, policy_net, target_net, cfg)
# Entrenamiento con manejo de Ctrl+C
# ...código anterior...

# Entrenamiento con manejo robusto de Ctrl+C y cierre seguro
try:
    trainer.train()
except KeyboardInterrupt:
    print("\n🚨 Entrenamiento detenido por el usuario. Guardando todo...")
finally:
    trainer.env.close()
    trainer.save_plot()
    print("🟢 Progreso y entorno cerrados correctamente. ¡Hasta luego!")