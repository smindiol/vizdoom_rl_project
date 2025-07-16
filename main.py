import yaml
import torch
import threading
from env.vizdoom_env import VizDoomGym
from trainer import DQNTrainer

# 📦 Cargar configuración
with open("config/config_defend_the_center.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 🎮 Crear entorno
scenario_path = cfg["env"]["scenario_path"]
env = VizDoomGym(render=cfg["env"]["render"], config_path=scenario_path)

# ⚙️ Parámetros
input_shape = tuple(cfg["env"]["input_shape"])
n_actions = cfg["env"]["actions"]
model_type = cfg.get("model", {}).get("type", "dqn")

# 🧠 Selección de modelo
if model_type == "dqn":
    from models.dqn import DQN as SelectedModel
elif model_type == "dqn_att":
    from models.dqn_att import DQNWithAttention as SelectedModel
else:
    raise ValueError(f"Modelo no reconocido: {model_type}")

# 💻 Preparar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Dispositivo actual: {device}")

policy_net = SelectedModel(input_shape, n_actions).to(device)
target_net = SelectedModel(input_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 🔍 Verificación y diagnóstico
print(f"📦 policy_net en: {next(policy_net.parameters()).device}")
print(f"📦 target_net en: {next(target_net.parameters()).device}")
param_device = next(policy_net.parameters()).device
assert param_device.type == device.type, f"❌ policy_net está en {param_device}, esperado en {device}"

# 🏋️‍♂️ Preparar trainer
trainer = DQNTrainer(env, policy_net, target_net, cfg)

# 🔁 Función de escape con ENTER
def keyboard_watcher():
    input("⛔ Presiona ENTER en cualquier momento para detener el entrenamiento...\n")
    raise KeyboardInterrupt

# 🧵 Lanzar watcher
watcher_thread = threading.Thread(target=keyboard_watcher, daemon=True)
watcher_thread.start()

# 🚀 Entrenamiento con manejo seguro
try:
    trainer.train()
except KeyboardInterrupt:
    print("\n🚨 Entrenamiento detenido por el usuario. Guardando todo...")
finally:
    trainer.env.close()
    trainer.save_plot()
    print("🟢 Progreso y entorno cerrados correctamente. ¡Hasta luego!")
