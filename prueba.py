import matplotlib.pyplot as plt
from env.vizdoom_env import VizDoomGym
from models.dqn_with_heatmap_vizdoom import YOLOHeatmapWrapper, ImageWithHeatmapProcessor
import torch

# 1. Inicializa el entorno y el heatmap processor
env = VizDoomGym(render=False)
obs = env.reset()  # [H, W, C]
device = "cuda" if torch.cuda.is_available() else "cpu"
heatmap_model = YOLOHeatmapWrapper()
processor = ImageWithHeatmapProcessor(heatmap_model, device)

num_frames = 5
for i in range(num_frames):
    # Acción aleatoria
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    combined = processor(obs)  # [1, 4, H, W]
    heatmap = combined[0, 3].cpu().numpy()  # Toma el canal del heatmap

    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Frame {i+1}")
    plt.subplot(1, 2, 1)
    plt.title("Imagen original")
    plt.imshow(obs)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.axis('off')

    plt.show()

    if done:
        obs = env.reset()