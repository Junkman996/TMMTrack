import torch
import torch.nn as nn
from .tim import TIM
from .stim import STIM

# Trajectory Motion Modeling
class TMMModule(nn.Module):
    def __init__(self, hidden_dim=256, tim_layers=2, stim_layers=2):
        super().__init__()
        self.tim = TIM(hidden_dim, tim_layers)
        self.stim = STIM(hidden_dim, stim_layers)

    def forward(self, trajectories):
        # trajectories: [B, T, N, D]
        t_feat = self.tim(trajectories)   # [B, T, N, D]
        st_feat = self.stim(t_feat)       # [B, T, N, D]
        return st_feat