import torch


def upper_triangle_mask(n: int, device: torch.device):
    mask = torch.zeros(n, n, device=device, dtype=torch.bool)
    triu = torch.triu_indices(n, n, device=device)
    mask[triu[0], triu[1]] = True
    return mask


def need_padding(batch: torch.Tensor):
    """
    If the batch have homogeneous sizes, then no padding
    """
    _, cnt = torch.unique_consecutive(batch, return_counts=True)
    return len(cnt.unique()) != 1
