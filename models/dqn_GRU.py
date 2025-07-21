import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: imagen [B, C, H, W]
        # game_vars: [B, 2]
        attention = torch.sigmoid(self.conv(x))  # [B, 1, H, W]
        return x * attention


class RecurrentDQNWithAttention(nn.Module):
    def __init__(self, input_shape, num_game_vars, num_actions, hidden_size=256):
        super(RecurrentDQNWithAttention, self).__init__()
        self.use_game_vars = True
        c, h, w = input_shape  # Por ejemplo: (2, 100, 160)

        # CNN + atención espacial
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

        # Calcular tamaño de salida del bloque convolucional
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.attention(self.conv(dummy))
            self.cnn_output_size = out.view(1, -1).size(1)

        # GRU input: CNN output + variables del juego (e.g. vida, munición)
        self.gru_input_size = self.cnn_output_size + num_game_vars
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=hidden_size, batch_first=True)

        # Capa final
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, image, game_vars, hidden_state=None):
        """
        image:      [B, C, H, W]
        game_vars:  [B, num_vars]
        hidden_state: [1, B, hidden_size] (opcional, para GRU)
        """
        B = image.size(0)
        x = image / 255.0

        x = self.conv(x)
        x = self.attention(x)
        x = x.reshape(B, -1)
        # Concatenar con game variables
        x = torch.cat([x, game_vars], dim=1)  # [B, cnn+vars]

        # GRU espera secuencias, así que añadimos dimensión temporal ficticia
        x = x.unsqueeze(1)  # [B, 1, D]
        output, h = self.gru(x, hidden_state)  # output: [B, 1, H]

        q_values = self.fc(output.squeeze(1))  # [B, num_actions]
        return q_values, h
