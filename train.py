import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from omegaconf import OmegaConf

from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pprint

from training.utils import (
    get_device,
    set_seed,
    build_dataloader,
    build_model,
    kaiming_init,
    zero_init_residual,
    build_optimizer,
    build_finetune_optimizer,
    build_scheduler,
    get_loss_fn
)

from torchmetrics.classification import (
    BinaryAUROC, 
    BinaryROC, 
    BinaryPrecisionRecallCurve,
    BinaryConfusionMatrix,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics import AveragePrecision

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device, cfg: OmegaConf) -> Tuple[float, float]:
    """Train the model for one epoch

    Args:
        model (nn.Module): The model to be trained
        loader (DataLoader): The data on which to train
        optimizer (torch.optim.Optimizer): The optimizer to update the model weights
        loss_fn (nn.Module): The loss function
        device (torch.device): The device on which to run the training

    Returns:
        Tuple[float, float]: Average model loss and accuracy across the training epoch
    """
    model.train()
    if cfg.data.weighted_sampling or cfg.model.pretrained:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(loader):
        # Unpack the batch
        x, y = batch
        x, y = x.to(device), y.to(device).float()
        
        # Compute the predictions
        pred = model(x)
        
        # Compute the loss
        loss = loss_fn(pred, y.unsqueeze(1))
        train_loss += loss.item() * x.size(0)
        
        correct += ((torch.sigmoid(pred) > 0.5).long() == y.unsqueeze(1)).sum().item()
        total += y.size(0)
        # Backpropagation:
        # Clear the gradients
        optimizer.zero_grad()
        # Backprop
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        if batch_idx % cfg.training.log_every_n_steps == 0:
            loss, current = loss.item(), batch_idx * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(loader.dataset):>5d}]")
    
    avg_loss = train_loss / len(loader.dataset)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc
    
    
def validate(model: nn.Module, 
             loader: DataLoader, 
             loss_fn: nn.Module, 
             device: torch.device, 
             auroc_metric: BinaryAUROC, 
             ap_metric: AveragePrecision, 
             acc_metric: BinaryAccuracy) -> Tuple[float, float, float, float, float]:
    """Validate the model

    Args:
        model (nn.Module): The model
        loader (DataLoader): The validation data
        loss_fn (nn.Module): The loss function
        device (torch.device): The device on which to run the validation

    Returns:
        Tuple[float, float]: The model's average validation loss and accuracy for the epoch
    """
    
    model.eval()
    val_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():  # Disable Gradient Calculation
        for batch_idx, batch in enumerate(loader):
            # Unpack the batch
            x, y = batch
            x, y = x.to(device), y.to(device).float()
            all_labels.append(y)
            # Compute the predictions
            pred = model(x)

            # Compute the loss
            loss = loss_fn(pred, y.unsqueeze(1))
            
            val_loss += loss.item() * x.size(0)
            all_preds.append(pred)
              
    preds = torch.cat(all_preds, dim=0).squeeze()
    probs = torch.sigmoid(preds)
    
    print(preds.mean().item(), preds.std().item())
    
    labels = torch.cat(all_labels, dim=0)
    avg_loss = val_loss / len(loader.dataset)
    accuracy = 100 * acc_metric(probs, labels).cpu().item()
    auroc = auroc_metric(probs, labels).cpu().item()
    ap = ap_metric(probs, labels.long()).cpu().item()
    ppr = 100 * (probs > 0.5).long().sum().cpu().item() / len(loader.dataset)
            
    return avg_loss, accuracy, auroc, ap, ppr


def load_best_model(cfg: OmegaConf) -> nn.Module:
    model = build_model(cfg)
    state_dict = torch.load(f"{cfg.experiment.output_dir}/best.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    return model


def collect_probs_and_labels(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    model.to(device)
    
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # extract batch
            x, y = batch
            x, y = x.to(device), y.to(device)
            all_labels.append(y)
            preds = model(x)
            all_probs.append(torch.sigmoid(preds))

    return torch.cat(all_probs, dim=0).squeeze(), torch.cat(all_labels, dim=0)


def final_evaluation(model: nn.Module, loader: DataLoader, device: torch.device, cfg: OmegaConf):
    fig_dir = Path(cfg.experiment.output_dir) / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("Collecting Probabilities and Labels")
    probs, labels = collect_probs_and_labels(model, loader, device)
    
    # Initialize metrics
    acc_metric = BinaryAccuracy(threshold=cfg.evaluation.accuracy_threshold).to(device)
    auroc_metric = BinaryAUROC().to(device)
    roc_metric = BinaryROC().to(device)
    confmat_metric = BinaryConfusionMatrix().to(device)
    pr_curve_metric = BinaryPrecisionRecallCurve().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    ap_metric = AveragePrecision(task='binary').to(device)
    
    metrics = {}
    # Update metrics with accumulated data
    # Accuracy
    acc = acc_metric(probs, labels)
    metrics["accuracy"] = acc.cpu().item()
    
    # ROC-AUC Score
    auroc = auroc_metric(probs, labels)
    metrics["auroc"] = auroc.cpu().item()
    
    # ROC    
    fprs, tprs, thresholds = roc_metric(probs, labels)
    fig, ax = roc_metric.plot(score=True)
    ax.set_title("ROC Curve")
    fig.savefig(fig_dir / "ROC_AUC.png")
    target_fpr = 1.0 - cfg.evaluation.specificity
    closest_threshold_idx = torch.argmin(torch.abs(fprs - target_fpr))
    optimal_threshold = thresholds[closest_threshold_idx]
    
    binarized_predictions = (probs >= optimal_threshold).int()
    
    # Calculate metrics at this fixed threshold
    precision = precision_metric(labels, binarized_predictions).cpu().item()
    sensitivity = recall_metric(labels, binarized_predictions).cpu().item() # Sensitivity is the same as Recall
    metrics['opt_threshold'] = optimal_threshold.cpu().item()
    metrics['opt_precision'] = precision
    metrics['opt_sensitivity'] = sensitivity

    # Verify specificity
    tn, fp, fn, tp = confmat_metric(labels, binarized_predictions).ravel().cpu().numpy()
    calculated_specificity = tn / (tn + fp)
    metrics["confmat"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    print(f"Threshold for {cfg.evaluation.specificity*100}% specificity: {optimal_threshold.cpu().item()}")
    print(f"Calculated Specificity: {calculated_specificity}")
    print(f"Precision at this threshold: {precision}")
    print(f"Sensitivity (Recall) at this threshold: {sensitivity}")

    # Precision/Recall curve
    precision, recall, thresholds = pr_curve_metric(probs, labels)
    fig, ax = pr_curve_metric.plot(score=True)
    fig.savefig(fig_dir / 'PR.png')
    avg_precision = ap_metric(probs, labels).cpu().item()
    metrics["ap"] = avg_precision
    
    metrics_df = pd.json_normalize(metrics)
    metrics_df.to_csv(Path(cfg.experiment.output_dir) / "metrics.csv")
    
    pprint.pprint(metrics)
    
    
def train(cfg: OmegaConf) -> None:
    device = get_device()
    
    # Save the exact configuration used in the experiment
    out_dir = cfg.experiment.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, f"{out_dir}/config_used.yaml")
    
    # Set all seeds
    set_seed(cfg.experiment.seed)

    # Build the DataLoaders    
    train_loader = build_dataloader(cfg, 'train')
    val_loader = build_dataloader(cfg, 'val')
    
    # Build the Model
    model = build_model(cfg)
    if not cfg.model.pretrained:
        kaiming_init(model)
        zero_init_residual(model)
    # Prior probability init: bias = log(p/(1-p)) = -log(pos_weight)
    # Ensures focal loss gradients are asymmetric from epoch 0 (focal loss paper §4.1)
    if cfg.training.loss == 'focal':
        pos_weight = train_loader.dataset.get_pos_weight()
        nn.init.constant_(model.fc.bias, -math.log(pos_weight))
    model.to(device)

    # Stage 1: freeze backbone so only the head trains first
    freeze_epochs = cfg.model.get('freeze_backbone_epochs', 0) if cfg.model.pretrained else 0
    if freeze_epochs > 0:
        for name, param in model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False

    # Get the optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Get the scheduler
    scheduler = build_scheduler(cfg, optimizer)
    
    # Get the Loss Function
    loss_fn = get_loss_fn(cfg, train_loader.dataset).to(device)
    
    auroc_metric = BinaryAUROC().to(device)
    ap_metric = AveragePrecision(task='binary').to(device)
    acc_metric = BinaryAccuracy(threshold=0.5).to(device)
    
    best_auroc = 0
    results = []
    # Train
    for epoch in range(cfg.training.max_epochs):
        # Stage 2: unfreeze backbone with differential LR
        if freeze_epochs > 0 and epoch == freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            current_lr = max(g['lr'] for g in optimizer.param_groups)
            optimizer = build_finetune_optimizer(cfg.optimizer, model, backbone_lr=current_lr / 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.training.max_epochs - epoch,
                eta_min=cfg.scheduler.cosine_schedule.eta_min
            )

        # Train on the training data
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, cfg)
        # Validate on the validation data
        val_loss, val_acc, val_auroc, val_ap, val_ppr = validate(model, val_loader, loss_fn, device, auroc_metric, ap_metric, acc_metric)
        print(f"Epoch {epoch}:\tTrain Loss {train_loss}, Train Acc {train_acc}")
        print(f"\t\t:Val Loss {val_loss}, Val Acc {val_acc}, Val ROC-AUC {val_auroc}, Val AP {val_ap}, Val PPR {val_ppr}")

        row = {'epoch': epoch,
               'lr': max(g['lr'] for g in optimizer.param_groups),
               'train_loss': train_loss,
               'train_acc': train_acc,
               'val_loss': val_loss,
               'val_acc': val_acc,
               'val_auroc': val_auroc,
               'val_ap': val_ap,
               'val_ppr': val_ppr}
        results.append(row)
        pd.DataFrame([row]).to_csv(
            f"{cfg.experiment.output_dir}/log.csv",
            mode='a',
            header=(epoch == 0)
        )
        if val_auroc >= best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), f"{cfg.experiment.output_dir}/best.pt")
        # Advance the LR Scheduler
        if scheduler:
            scheduler.step()

    torch.save(model.state_dict(), f"{cfg.experiment.output_dir}/last.pt")
    
    # Final evaluation:
    best_model = load_best_model(cfg)
    test_loader = build_dataloader(cfg, 'test')
    final_evaluation(best_model, test_loader, device, cfg)    
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    
    train(cfg)
    