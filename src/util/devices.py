import torch
from util import mini

def get_device():
    if torch.cuda.is_available():
        if mini.is_mini():
            raise Exception("Did not expect mini mode when CUDA is available")
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
