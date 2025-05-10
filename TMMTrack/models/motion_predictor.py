import torch
import torch.nn as nn

class MotionPredictor(nn.Module):
    def __init__(self, in_dim, hid_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 4)
        )

    def forward(self, feat):
        # feat: [B, T, N, D] -> use last frame
        x = feat[:,-1]  # [B, N, D]
        return self.fc(x)