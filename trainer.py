# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from pathlib import Path
from utils.replay_memory import ReplayMemory
import matplotlib.pyplot as plt


class DQNTrainer:
    def __init__(self, env, policy_net, target_net, config):
        self.rewards_log = []
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.cfg = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayMemory(self.cfg["training"]["memory_size"])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg["training"]["learning_rate"])
        self.epsilon = self.cfg["training"]["epsilon_start"]
        self.best_reward = -float("inf")
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def preprocess(self, obs):
        return torch.tensor(np.moveaxis(obs, -1, 0), dtype=torch.float32).unsqueeze(0).to(self.device)

    def save_plot(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.rewards_log)
        plt.title("Recompensa por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa total")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reward_curve.png")
        plt.close()
        print("Gráfica de recompensas guardada como reward_curve.png")

    def train(self):
        for episode in range(self.cfg["training"]["episodes"]):
            obs = self.env.reset()
            total_reward = 0

            for _ in range(self.cfg["training"]["max_steps"]):
                if random.random() < self.epsilon:
                    action = random.randint(0, self.cfg["env"]["actions"] - 1)
                else:
                    with torch.no_grad():
                        state_tensor = self.preprocess(obs)
                        q_values = self.policy_net(state_tensor)
                        action = q_values.argmax().item()

                next_obs, reward, done, _ = self.env.step(action)
                total_reward += reward

                self.memory.push(obs, action, reward, next_obs, done)
                obs = next_obs

                if len(self.memory) > self.cfg["training"]["batch_size"]:
                    self.optimize_model()

                if done:
                    break

            self.update_target_network(episode)
            self.save_checkpoints(episode, total_reward)
            self.epsilon = max(self.cfg["training"]["epsilon_end"], self.epsilon * self.cfg["training"]["epsilon_decay"])
            self.rewards_log.append(total_reward)
            print(f"Ep {episode+1}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        self.env.close()
        self.save_plot()


    def optimize_model(self):
        batch = self.memory.sample(self.cfg["training"]["batch_size"])
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(np.moveaxis(states, -1, 1), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.moveaxis(next_states, -1, 1), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.cfg["training"]["gamma"] * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, episode):
        if (episode + 1) % self.cfg["training"]["target_update_freq"] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoints(self, episode, total_reward):
        if (episode + 1) % self.cfg["training"]["checkpoint_freq"] == 0:
            torch.save(self.policy_net.state_dict(), self.checkpoint_dir / f"dqn_ep{episode+1}.pth")

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            torch.save(self.policy_net.state_dict(), self.checkpoint_dir / "dqn_best.pth")
            print(f"Nuevo mejor modelo guardado (reward = {self.best_reward:.2f})")
