import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import yaml
from pathlib import Path
from utils.replay_memory import ReplayMemory
import matplotlib.pyplot as plt


class DQNTrainer:
    def __init__(self, env, policy_net, target_net, config):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Usando dispositivo: {self.device}")

        # Verificaciones
        print(f" policy_net en: {next(self.policy_net.parameters()).device}")
        print(f" target_net en: {next(self.target_net.parameters()).device}")
        #assert next(self.policy_net.parameters()).device == self.device, " policy_net no está en el dispositivo esperado"
        #assert next(self.target_net.parameters()).device == self.device, " target_net no está en el dispositivo esperado"

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg["training"]["learning_rate"])
        self.memory = ReplayMemory(self.cfg["training"]["memory_size"])

        self.checkpoint_dir = Path(self.cfg["training"]["checkpoint_path"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.rewards_log = []
        self.metrics_log = []
        self.epsilon = self.cfg["training"]["epsilon_start"]
        self.best_reward = -float("inf")
        self.start_episode = 0

        self.load_previous()

    def load_previous(self):
        meta_path = self.checkpoint_dir / "meta.yaml"
        rewards_path = self.checkpoint_dir / "rewards.npy"
        metrics_path = self.checkpoint_dir / "metrics.npy"

        models = sorted(self.checkpoint_dir.glob("dqn_ep*.pth"), key=lambda p: int(p.stem.split("_ep")[-1]))
        if models:
            latest_model_path = models[-1]
            last_episode = int(latest_model_path.stem.split("_ep")[-1])
            self.start_episode = last_episode + 1

            #  Asegurarse de cargar al dispositivo actual
            self.policy_net.load_state_dict(torch.load(latest_model_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f" Modelo cargado desde {latest_model_path}, comenzando desde episodio {self.start_episode}")

        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)
                self.epsilon = meta.get("epsilon", self.epsilon)
                self.best_reward = meta.get("best_reward", self.best_reward)
        else:
            print(" No se encontró meta.yaml. Entrenamiento iniciará desde cero.")

        if rewards_path.exists():
            self.rewards_log = list(np.load(rewards_path))
            print(f" Recompensas anteriores cargadas ({len(self.rewards_log)} episodios)")
        else:
            print(" No se encontraron recompensas anteriores.")

        if metrics_path.exists():
            self.metrics_log = list(np.load(metrics_path, allow_pickle=True))
            print(f" Métricas anteriores cargadas ({len(self.metrics_log)} episodios)")
        else:
            print(" No se encontraron métricas anteriores.")

    def preprocess(self, obs, game_vars=None):
        # (H, W, C) → (C, H, W) → [1, C, H, W] en float
        img_tensor = torch.from_numpy(np.moveaxis(obs, -1, 0)).float().unsqueeze(0).to(self.device)  # [1, C, H, W]
        if getattr(self.policy_net, "use_game_vars", False):
            vars_tensor = torch.tensor(game_vars, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 2]
            return img_tensor, vars_tensor
        else:
            return img_tensor

    def train(self):
        for episode in range(self.start_episode, self.cfg["training"]["episodes"]):
            obs = self.env.reset()
            if getattr(self.policy_net, "use_game_vars", False):
                 game_vars = self.env.get_game_vars()  # ejemplo: [health, ammo]
            total_reward = 0
            steps = 0

            for _ in range(self.cfg["training"]["max_steps"]):
                if random.random() < self.epsilon:
                    action = random.randint(0, self.cfg["env"]["actions"] - 1)
                else:
                    with torch.no_grad():
                        if getattr(self.policy_net, "use_game_vars", False):
                            input_tensor, vars_tensor = self.preprocess(obs, game_vars)
                            q_values = self.policy_net(input_tensor, vars_tensor)
                        else:
                            input_tensor = self.preprocess(obs)
                            q_values = self.policy_net(input_tensor)
                        action = q_values.argmax().item()

                next_obs, reward, done, _ = self.env.step(action)
                if getattr(self.policy_net, "use_game_vars", False):
                    next_vars = self.env.get_game_vars()
                    self.memory.push(obs, action, reward, next_obs, done, game_vars, next_vars)
                    game_vars = next_vars  # actualiza para el siguiente paso
                else:
                    self.memory.push(obs, action, reward, next_obs, done)
                total_reward += reward
                obs = next_obs
                steps += 1

                if len(self.memory) > self.cfg["training"]["batch_size"]:
                    self.optimize_model()

                if done:
                    break

            self.rewards_log.append(total_reward)
            self.metrics_log.append({"episode": episode + 1, "reward": total_reward, "steps": steps})
            self.epsilon = max(
                self.cfg["training"]["epsilon_end"],
                self.epsilon * self.cfg["training"]["epsilon_decay"]
            )
            self.update_target_network(episode)
            self.save_checkpoints(episode, total_reward)

            print(f" Ep {episode+1} | Reward: {total_reward:.2f} | Steps: {steps} | Epsilon: {self.epsilon:.3f}")

        self.env.close()
        self.save_plot()
        
    def optimize_model(self):
        batch = self.memory.sample(self.cfg["training"]["batch_size"])

        if getattr(self.policy_net, "use_game_vars", False):
            obs_imgs, obs_vars, actions, rewards, next_obs_imgs, next_obs_vars, dones = batch

            obs_imgs = torch.from_numpy(np.moveaxis(obs_imgs, -1, 1)).float().to(self.device)
            next_obs_imgs = torch.from_numpy(np.moveaxis(next_obs_imgs, -1, 1)).float().to(self.device)
            obs_vars = torch.from_numpy(obs_vars).float().to(self.device)
            next_obs_vars = torch.from_numpy(next_obs_vars).float().to(self.device)
            actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            dones = torch.from_numpy(dones).float().to(self.device)

            q_values, _ = self.policy_net(obs_imgs, obs_vars)
            q_values = q_values.gather(1, actions).squeeze(1)

            with torch.no_grad():
                next_q_values, _ = self.target_net(next_obs_imgs, next_obs_vars)
                max_next_q = next_q_values.max(1)[0]
                target = rewards + self.cfg["training"]["gamma"] * max_next_q * (1 - dones)

        else:
            states, actions, rewards, next_states, dones = batch

            states = torch.from_numpy(np.moveaxis(states, -1, 1)).float().to(self.device)
            next_states = torch.from_numpy(np.moveaxis(next_states, -1, 1)).float().to(self.device)
            actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            dones = torch.from_numpy(dones).float().to(self.device)

            q_values = self.policy_net(states).gather(1, actions).squeeze(1)

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
            print(f" Target network actualizada en el episodio {episode+1}")

    def save_checkpoints(self, episode, total_reward):
        if (episode + 1) % self.cfg["training"]["checkpoint_freq"] == 0:
            torch.save(self.policy_net.state_dict(), self.checkpoint_dir / f"dqn_ep{episode+1}.pth")

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            torch.save(self.policy_net.state_dict(), self.checkpoint_dir / "dqn_best.pth")
            print(f" Nuevo mejor modelo guardado (reward = {self.best_reward:.2f})")

        meta = {
            "episode": episode + 1,
            "epsilon": self.epsilon,
            "best_reward": self.best_reward
        }
        with open(self.checkpoint_dir / "meta.yaml", "w") as f:
            yaml.dump(meta, f)

        np.save(self.checkpoint_dir / "rewards.npy", np.array(self.rewards_log))
        np.save(self.checkpoint_dir / "metrics.npy", np.array(self.metrics_log, dtype=object))

    def save_plot(self):
        path = self.checkpoint_dir / "reward_curve.png"
        plt.figure(figsize=(10, 4))
        plt.plot(self.rewards_log)
        plt.title("Recompensa por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa total")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f" Curva guardada en {path}")