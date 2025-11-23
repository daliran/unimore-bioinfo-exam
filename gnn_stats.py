import numpy as np
import random
import torch
from scipy import stats

seed_pool = [
    12,
    37,
    58,
    93,
    117,
    142,
    199,
    224,
    255,
    312,
    345,
    389,
    415,
    456,
    489,
    512,
    578,
    603,
    644,
    677,
    702,
    745,
    799,
    820,
    856,
    901,
    933,
    974,
    1003,
    1044,
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


def mean_confidence_interval(data, confidence=0.95, dist="t"):
    a = np.array(data)
    n = len(a)
    mean = np.mean(a)
    sem = np.std(a, ddof=1) / np.sqrt(n)

    if dist == "t":
        lower, upper = stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
    elif dist == "norm":
        lower, upper = stats.norm.interval(confidence, loc=mean, scale=sem)
    else:
        raise ValueError("dist must be 't' or 'norm'")

    return mean, lower, upper
