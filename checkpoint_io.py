import torch


def torch_load_ckpt(path: str, map_location):
    """Load full checkpoint dict; compatible across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
