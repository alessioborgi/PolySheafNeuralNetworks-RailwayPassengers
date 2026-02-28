"""Utilities for building node subset masks for per-group evaluation."""

import numpy as np
import torch


def build_node_subset_masks(average_p_counts):
    """Build boolean masks for node subsets based on average raw passenger counts.
    Args:
        average_p_counts: 1-D numpy array [N] of average raw passenger counts per station.
    Returns:
        dict of {name: bool tensor [N]} for 'all', 'top10', 'top100', 'bottom100'.
    """
    N = len(average_p_counts)
    sorted_idx = np.argsort(average_p_counts)
    masks = {}
    masks["all"] = torch.ones(N, dtype=torch.bool)
    for name, indices in [("top10", sorted_idx[-10:]),
                          ("top100", sorted_idx[-100:]),
                          ("bottom100", sorted_idx[:100])]:
        m = torch.zeros(N, dtype=torch.bool)
        m[indices] = True
        masks[name] = m
    return masks
