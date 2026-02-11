import torch
from torch import nn

from typing import Literal


def get_norm(norm: Literal["batch", "group"], num_channels: int, num_groups: int = 1) -> nn.Module:
    """Create a normalization layer for the ResNet, either BatchNorm or GroupNorm

    Args:
        norm (Literal[&quot;batch&quot;, &quot;group&quot;]): The norm name
        num_channels (int): The number of channels
        num_groups (int, optional): The number of groups. Relevant only for GroupNorm. Defaults to 1.

    Raises:
        Exception: _description_

    Returns:
        nn.Module: _description_
    """
    if num_channels % num_groups != 0:
        raise Exception(f"Channels: {num_channels} % Groups: {num_groups} != 0")
    if norm == "batch":
        return nn.BatchNorm2d(num_features=num_channels)
    else:
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
