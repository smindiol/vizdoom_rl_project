import sys
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading
import signal
import io
import yaml
import torch
from env.vizdoom_env import VizDoomGym
from trainer import DQNTrainer

# ==== Configuración ====
with open("config/config_defend_the_center.yaml", "r") as f:
    cfg = yaml.safe_load(f)

scenario_path = cfg["env"]["scenario_path"]
env = VizDoomGym(render=cfg["env"]["render"], config_path=scenario_path)

input_shape = tuple(cfg["env"]["input_shape"])
n_actions = cfg["env"]["actions"]
model_type = cfg.get("model", {}).get("type", "dqn")

# ==== Selección de modelo ====
if model_type == "dqn":
    from models.dqn import DQN as SelectedModel
elif model_type == "dqn_att":
    from models.dqn_att import DQNWithAttention as SelectedModel
else:
    raise ValueError(f"Modelo no reconocido: {model_type}")

# ==== Crear redes ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Dispositivo actual: {device}")
policy_net = SelectedModel(input_shape, n_actions).to(device)
target_net = SelectedModel(input_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# ==== Inicializar trainer ====
trainer = DQNTrainer(env, policy_net, target_net, cfg)

# ========== Interfaz gráfica ==========
stop_flag = False
training_thread = None

class TextRedirector(io.StringIO):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def write(self, string):
        self.widget.insert(tk.END, string)
        self.widget.see(tk.END)

    def flush(self):
        pass

def start_training():
    import __main__
    __main__.stop_flag = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("🛑 Entrenamiento detenido.")
    finally:
        trainer.env.close()
        trainer.save_plot()
        print("✅ Progreso guardado y entorno cerrado.")

def stop_training():
    import __main__
    __main__.stop_flag = True

def run_gui():
    global training_thread

    root = tk.Tk()
    root.title("Entrenamiento RL - VizDoom")

    text_area = ScrolledText(root, height=30, width=100)
    text_area.pack(padx=10, pady=10)

    stop_button = tk.Button(root, text=" Detener entrenamiento", command=stop_training, bg="red", fg="white")
    stop_button.pack(pady=5)

    sys.stdout = TextRedirector(text_area)

    training_thread = threading.Thread(target=start_training)
    training_thread.start()

    root.mainloop()

# === Iniciar GUI ===
run_gui()
