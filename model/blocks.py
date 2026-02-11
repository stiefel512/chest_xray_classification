import torch
from torch import nn
from model.utils import get_norm

from typing import Tuple, Optional, Literal


class ResidualBlock(nn.Module):
    """A Residual Block"""

    expansion = 1
    
    def __init__(self, 
                 in_planes: int, 
                 planes: int, 
                 stride: int | Tuple[int, int] = 1, 
                 downsample: Optional[nn.Module] = None, 
                 norm: Literal["batch", "group"] = "batch", 
                 num_groups: int = 32) -> None:
        """Constructor for a Residual Block

        Args:
            in_planes (int): Number of input channels
            planes (int): Number of output channels
            stride (int | Tuple[int, int], optional): Stride of the 1st Convolution. Defaults to 1.
            downsample (Optional[nn.Module], optional): Downsample Module for the Residual Connection. Defaults to None.
            norm (Literal[&quot;batch&quot;, &quot;group&quot;], optional): Which norm to choose. Defaults to "batch".
            num_groups (int, optional): Number of groups to be used in GroupNorm. Defaults to 32.
        """
        super(ResidualBlock, self).__init__()
                
        self.conv1 = nn.Conv2d(in_channels=in_planes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.norm1 = get_norm(norm, planes, num_groups)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels=planes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.norm2 = get_norm(norm, planes, num_groups)
        
        self.downsample = downsample if downsample is not None else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the ResidualBlock

        Args:
            x (torch.Tensor): The data to be processed

        Returns:
            torch.Tensor: The processed data
        """
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        residual = self.downsample(residual)
        
        return self.relu(x + residual)


class ResidualBottleneckBlock(nn.Module):
    """A Residual Bottleneck Block"""
    expansion = 4
    
    def __init__(self, 
                 in_planes: int, 
                 planes: int, 
                 stride: int | Tuple[int, int] = 1, 
                 downsample: Optional[nn.Module] = None, 
                 norm: Literal["batch", "group"] = "batch", 
                 num_groups: int = 32) -> None:
        """Constructor for a Residual Bottleneck Block
        
        Args:
            in_planes (int): The number of input channels
            planes (int): The number of interim channels
            stride (int | Tuple[int, int], optional): The stride of the 1st convolution. Defaults to 1.
            downsample (Optional[nn.Module], optional): The downsample module for the Residual Connection. Defaults to None.
            norm (Literal[&quot;batch&quot;, &quot;group&quot;], optional): The norm to use. Defaults to "batch".
            num_groups (int, optional): The number of groups for GroupNorm. Defaults to 32.
        """
        super(ResidualBottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_planes,
                               out_channels=planes,
                               kernel_size=1,
                               stride=stride,
                               padding=0,
                               bias=False)
        self.norm1 = get_norm(norm, planes, num_groups)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels=planes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.norm2 = get_norm(norm, planes, num_groups)
        
        self.conv3 = nn.Conv2d(in_channels=planes,
                               out_channels=planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.norm3 = get_norm(norm, planes * self.expansion, num_groups)
        
        self.downsample = downsample if downsample is not None else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass of the Residual Bottleneck Block

        Args:
            x (torch.Tensor): The data to be processed

        Returns:
            torch.Tensor: The processed data
        """
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        residual = self.downsample(residual)
        return self.relu(x + residual) 