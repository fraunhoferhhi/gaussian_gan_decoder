import torch
from torch.nn.functional import softplus


def sigma2opacity(sigma, gaussian_model):
    sigma = softplus(sigma - 1)
    sigma = sigma * 1.0 / 512
    alpha = 1 - torch.exp(-sigma)
    alpha = gaussian_model.inverse_opacity_activation(alpha)
    alpha[torch.isneginf(alpha)] = -100
    alpha[torch.isinf(alpha)] = 100
    return alpha

def rgb2gaussiancolor(rgb):
    return torch.clip(rgb[..., :3], 0, 1)
