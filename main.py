# main.py

import yaml
import torch
from env.vizdoom_env import VizDoomGym
from models.dqn import DQN
from trainer import DQNTrainer

with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

env = VizDoomGym(render=cfg["env"]["render"])
input_shape = tuple(cfg["env"]["input_shape"])
n_actions = cfg["env"]["actions"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(input_shape, n_actions).to(device)
target_net = DQN(input_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

trainer = DQNTrainer(env, policy_net, target_net, cfg)
trainer.train()
