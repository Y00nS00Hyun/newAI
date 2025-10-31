from typing import Tuple

import torch


def apply_augmentation(
    inputs: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tensors unchanged until custom augmentations are implemented."""
    return inputs, lengths, labels
