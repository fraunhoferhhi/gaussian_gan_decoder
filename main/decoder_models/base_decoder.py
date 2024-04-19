import torch
from torch import nn

from torch_utils import persistence


@persistence.persistent_class
class Decoder(nn.Module):
    def __init__(self, n_features, out_features=3, hidden_dim=128):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_features, out_features=hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=out_features),
        )

    def forward(self, triplane_features, gaussian_features):
        triplane_features = triplane_features.mean(0)
        n, c = triplane_features.shape
        triplane_features = triplane_features.view(n, c)
        x = torch.concat([triplane_features, gaussian_features], dim=-1)
        x = self.backbone(x)
        return x

