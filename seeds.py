import numpy as np
import random
import torch

seed_pool = [
    12, 37, 58, 93, 117, 142, 199, 224, 255, 312,
    345, 389, 415, 456, 489, 512, 578, 603, 644, 677,
    702, 745, 799, 820, 856, 901, 933, 974, 1003, 1044
]

def set_common_seeds(seed):

    if seed is None:
        return

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
