from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np

class YOLOHeatmapWrapper(nn.Module):
    def __init__(self, model_name="yolov8n.pt"):
        super().__init__()
        self.model = YOLO(model_name)
        self.model.eval()

    def forward(self, x):
        # x: [B, 3, H, W]
        B, _, H, W = x.shape
        heatmaps = []
        for i in range(B):
            img = x[i].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            results = self.model.predict(img, verbose=False)[0]

            heatmap = torch.zeros((H, W))
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = box.int()
                heatmap[y1:y2, x1:x2] += 1.0

            heatmap = torch.clamp(heatmap, 0, 1).unsqueeze(0)
            heatmaps.append(heatmap)

        return torch.stack(heatmaps, dim=0).to(x.device)

class ImageWithHeatmapProcessor:
    def __init__(self, heatmap_model, device):
        self.heatmap_model = heatmap_model.to(device)
        self.device = device

    def __call__(self, obs_np):
        # obs_np: [H, W, C] en NumPy (RGB)
        obs = torch.from_numpy(np.moveaxis(obs_np, -1, 0)).float().unsqueeze(0).to(self.device)  # [1, 3, H, W]
        with torch.no_grad():
            heatmap = self.heatmap_model(obs)  # [1, 1, H, W]
        combined = torch.cat([obs, heatmap], dim=1)  # [1, 4, H, W]
        return combined
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class DQNWithAttention(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNWithAttention, self).__init__()
        c, h, w = input_shape  # c = 4 (3 canales RGB + 1 heatmap)

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.attention = SpatialAttention(64)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy_input)
            conv_out = self.attention(conv_out)
            conv_output_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = self.attention(x)
        return self.fc(x.view(x.size(0), -1))
