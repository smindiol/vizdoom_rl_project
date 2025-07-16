import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))  # [B, 1, H, W]
        return x * attention


class DQNWithAttention(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNWithAttention, self).__init__()
        c, h, w = input_shape  # Ej: (1, 100, 160)

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

        #  dummy_input solo para calcular tamaño del vector FC, aún en CPU
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # En CPU
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
        #assert x.device == next(self.parameters()).device, " Input no está en el mismo device que el modelo"
        x = x / 255.0  # Normalización
        x = self.conv(x)
        x = self.attention(x)
        return self.fc(x.reshape(x.size(0), -1))
