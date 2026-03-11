from dataset import ChestXRayDataset
import torch
from torch import nn
from model.model import ResNet
from model.blocks import ResidualBlock, ResidualBottleneckBlock
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
import random
import numpy as np
import os
from typing import Literal
from pathlib import Path
from omegaconf import OmegaConf
from parse import parse
from losses.focal import BinaryFocalLoss
    

def get_device() -> torch.device:
    """Get the device on which to train and run the network

    Returns:
        torch.device: The device
    """
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def set_seed(seed: int) -> None:
    """Set all the different seeds, for experiment reproducibility

    Args:
        seed (int): The seed
    """
    # Set python's seed
    random.seed(seed)
    
    # Numpy's seed
    np.random.seed(seed)
    
    # PyTorch's seed
    torch.manual_seed(seed)
    
    # PyTorch cuda's seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # cuDNN/Backend determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    

class PerImageStandardize:
    """Standardize each image to zero mean and unit variance (per-image, not global)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + self.eps)


def build_transform(cfg: OmegaConf, train: bool) -> T.transforms:
    """Build the training transform

    Args:
        cfg (OmegaConf): The configuration

    Returns:
        T.transforms: The composed transform
    """
    transforms = []
    if train:
        augment_cfg = cfg.data.augmentations
        # Random Rotations
        if augment_cfg.rotations.enabled:
            transforms.append(T.RandomRotation(augment_cfg.rotations.max_angle, T.InterpolationMode.BILINEAR))
        # Random Horizontal Filp
        if augment_cfg.horizontal_flip.enabled:
            transforms.append(T.RandomHorizontalFlip(p=augment_cfg.horizontal_flip.p))
        # Random Contrast Jitter
        if augment_cfg.contrast_jitter.enabled:
            transforms.append(T.ColorJitter(contrast=augment_cfg.contrast_jitter.contrast))
        # Random Crop
        if augment_cfg.crop.enabled:
            transforms.append(T.RandomCrop(size=cfg.data.input_size[1:], padding=augment_cfg.crop.padding))

    transforms.append(T.Resize(cfg.data.input_size[1:]))
    transforms.append(T.ToTensor())

    normalization = cfg.data.get("normalization", "global")
    if normalization == "per_image":
        transforms.append(PerImageStandardize())
    else:
        transforms.append(T.Normalize((0.4913,), (0.2494,)))  # Training DS Statistics

    transforms = T.Compose(transforms)
    return transforms
    
    
def seed_worker(worker_id):
    """Seed DataLoader workers, to ensure consistent training order and augmentations across runs

    Args:
        worker_id (_type_): The ID of the worker to be seeded
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def build_dataloader(cfg: OmegaConf, stage: Literal["train", "val", "test"]) -> DataLoader:
    """Build the DataLoader objects for training and validation

    Args:
        cfg (OmegaConf): The Configuration

    Raises:
        Exception: Unsupported Dataset

    Returns:
        Tuple[DataLoader, DataLoader]: Training and Validation DataLoader objects
    """
    
    # Set the trainsforms:
    transforms = build_transform(cfg, stage == "train")

    # Get the csv path
    csv_path = Path(f"{cfg.data.data_dir}") / "splits" / (stage + ".csv")
    
    # Load the dataset
    ds = ChestXRayDataset(csv_path, Path(cfg.data.src_dir), transforms)    
    
    if stage == 'train' and cfg.data.weighted_sampling:
        # Get the Sampler
        sampler = WeightedRandomSampler(
            weights=ds.get_sample_weights().double(),
            num_samples=len(ds),
            replacement=True
        )
        
        # Ensure the same shuffle order and random augmentations per epoch
        g = torch.Generator()
        g.manual_seed(cfg.experiment.seed)
        dataloader = DataLoader(
            dataset = ds,
            batch_size = cfg.data.batch_size,
            # shuffle = True if stage == 'train' else False,
            sampler = sampler,
            num_workers = cfg.data.num_workers,
            worker_init_fn = seed_worker,
            generator = g,
            pin_memory = True if stage == 'train' else False
        )
    else:
        g = torch.Generator()
        g.manual_seed(cfg.experiment.seed)
        dataloader = DataLoader(
            dataset = ds,
            batch_size = cfg.data.batch_size,
            shuffle = True if stage == 'train' else False,
            num_workers = cfg.data.num_workers,
            worker_init_fn = seed_worker,
            generator = g,
            pin_memory = True if stage == 'train' else False
        )
    return dataloader


def build_model(cfg: OmegaConf) -> nn.Module:
    """Build the Model

    Args:
        cfg (OmegaConf): The configuration

    Raises:
        Exception: Incorrect Model Architecture
        Exception: Unsupported Dataset

    Returns:
        nn.Module: ResNet
    """
    arch = parse("resnet{configuration}", cfg.model.architecture)
    if arch is None:
        raise Exception("Incorrect Model Architecture")
    
    return ResNet(
        configuration = int(arch['configuration']),
        in_channels = cfg.data.input_size[0],
        num_classes = 1,
        norm = cfg.model.norm,
        num_groups = cfg.model.num_groups,
        base_channels = cfg.model.base_channels
    )
    
    
def kaiming_init(model: nn.Module) -> None:
    """Explicitly initialize the model weights with kaiming initialization

    Args:
        model (nn.Module): The model to initialize
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity='relu'
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_in',
                nonlinearity='relu'
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            
            
def zero_init_residual(model: nn.Module) -> None:
    """Zero-Init the Residual mappings, as described by He et al.

    Args:
        model (nn.Module): The model to initialize
    """
    for m in model.modules():
        if isinstance(m, ResidualBlock):
            if hasattr(m.norm2, "weight"):
                nn.init.zeros_(m.norm2.weight)
        elif isinstance(m, ResidualBottleneckBlock):
            if hasattr(m.norm3, "weight"):
                nn.init.zeros_(m.norm3.weight)
                

def build_optimizer(optimizer_conf: OmegaConf, model: nn.Module) -> torch.optim.Optimizer:
    """Build the Optimizer

    Args:
        optimizer_conf (OmegaConf): The configuration
        model (nn.Module): The model to be optimized

    Raises:
        Exception: Unsupported Optimizer

    Returns:
        torch.optim.Optimizer: SGD or Adam optimizer
    """
    
    # Decouple batch norm from weight decay for small DS
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if param.ndim == 1 or "bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    
    if optimizer_conf.type == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = optimizer_conf.lr,
            momentum = optimizer_conf.momentum,
            weight_decay = optimizer_conf.weight_decay
        )
    elif optimizer_conf.type == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = optimizer_conf.lr,
            weight_decay = optimizer_conf.weight_decay
        )
    elif optimizer_conf.type == 'AdamW':
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": optimizer_conf.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr = optimizer_conf.lr,
        )
    else:
        raise Exception("Optimizer type not supported")
    
    
def build_scheduler(cfg: OmegaConf, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the LR Scheduler

    Args:
        cfg (OmegaConf): The configuration
        optimizer (torch.optim.Optimizer): The optimizer

    Raises:
        Exception: Unsupported LR Scheduler

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The scheduler
    """
    if cfg.scheduler.type not in ['cosine', 'step']:
        return None
    
    if cfg.scheduler.warmup is not None and cfg.scheduler.warmup.epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=cfg.scheduler.warmup.start_factor,
            end_factor=1.0,
            total_iters=cfg.scheduler.warmup.epochs
        )
        remaining_epochs = cfg.training.max_epochs - cfg.scheduler.warmup.epochs
    else:
        warmup_scheduler = None
        remaining_epochs = cfg.training.max_epochs
    
    if cfg.scheduler.type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=remaining_epochs,
            eta_min=cfg.scheduler.cosine_schedule.eta_min
        )
    elif cfg.scheduler.type == 'step':
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, 
            step_size=cfg.scheduler.step_schedule.step_size,
            gamma=cfg.scheduler.step_schedule.gamma
        )
    else:
        return None
    
    if warmup_scheduler is None:
        return main_scheduler
    else:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[cfg.scheduler.warmup.epochs]
        )
        
def get_loss_fn(cfg: OmegaConf, train_ds: ChestXRayDataset) -> nn.Module:
    """Create the loss function

    Args:
        cfg (OmegaConf): The configuration
        train_ds (ChestXRayDataset): The data to be trained on

    Returns:
        nn.Module: CrossEntropyLoss object
    """
    if cfg.training.loss == 'bce':
        if cfg.training.weighted_loss:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_ds.get_pos_weight()]))
        else:
            return nn.BCEWithLogitsLoss()
    elif cfg.training.loss == 'focal':
        return BinaryFocalLoss(cfg.training.focal_gamma, cfg.training.focal_alpha, reduction='mean')
