import sys

import torch
sys.path.append("./PanoHead")
from PanoHead.training.networks_stylegan2 import FullyConnectedLayer


class HUGSDecoder(torch.nn.Module):
    def __init__(self, n_features, out_features: list, hidden_dim=128):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            FullyConnectedLayer(n_features, hidden_dim),
            torch.nn.GELU(),
            FullyConnectedLayer(hidden_dim, hidden_dim),
            torch.nn.GELU()
        )

        self.heads = []
        for f in out_features:
            self.heads.append(FullyConnectedLayer(hidden_dim, f).to("cuda"))

    def forward(self, sampled_features, xyz):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)
        x = torch.concat([x, xyz], dim=-1)
        x = self.backbone(x)

        outs = []
        for head in self.heads:
            outs.append(head(x))
        x = torch.concat(outs, dim=-1)
        x = x.view(N, M, -1)
        return x