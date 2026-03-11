import torch
from torch import nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
            
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply alpha if needed
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss
        
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
            
        # Apply the focal loss weighting
        loss = focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss