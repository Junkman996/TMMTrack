import torch
import torch.nn as nn

# Temporal Interaction Module
class TIM(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, traj):
        # traj: [B, T, N, D] -> flatten N as batch
        B, T, N, D = traj.shape
        x = traj.view(B*N, T, D).permute(1, 0, 2)
        out = self.encoder(x)
        return out.permute(1, 0, 2).view(B, T, N, D)