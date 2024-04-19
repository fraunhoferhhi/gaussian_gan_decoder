import numpy as np
import torch
import random


def set_seeds(seed_value):
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed(seed_value)  # PyTorch CUDA (for GPU computations)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # PyTorch CuDNN optimizer
    torch.backends.cudnn.benchmark = False
