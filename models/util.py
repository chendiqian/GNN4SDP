import torch


def upper_triangle_mask(n: int, device: torch.device):
    mask = torch.zeros(n, n, device=device, dtype=torch.bool)
    triu = torch.triu_indices(n, n, device=device)
    mask[triu[0], triu[1]] = True
    return mask
