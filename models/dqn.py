# models/dqn.py

import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape  # Ej: (1, 100, 160)

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Determinar tamaño de salida automáticamente
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output_size = self.conv(dummy_input).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0  # Normalizar imágenes
        x = self.conv(x)
        return self.fc(x.reshape(x.size(0), -1))

