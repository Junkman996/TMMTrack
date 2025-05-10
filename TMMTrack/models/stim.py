import torch
import torch.nn as nn

# Spatio-Temporal Interaction Module
class STIM(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, feats):
        # feats: [B, T, N, D] -> flatten T as batch
        B, T, N, D = feats.shape
        x = feats.permute(0,2,1,3).reshape(B*T, N, D).permute(1,0,2)
        out = self.encoder(x)
        return out.permute(1,0,2).view(B, T, N, D)