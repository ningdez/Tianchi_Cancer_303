import torch
import torch.nn as nn
import numpy as np
from ..registry import LOSSES
from .utils import weighted_loss,weight_reduce_loss


@weighted_loss
def balanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction='mean'):
    pred_bbox, pred_vari = pred
    assert beta > 0
    assert pred_bbox.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred_bbox - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        torch.exp(-pred_vari)*(alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff)+0.01*torch.pow(-3-pred_vari,2),
        torch.exp(-pred_vari)*(gamma * diff + gamma / b - alpha * beta)+0.5*pred_vari)

    return loss


@LOSSES.register_module
class KLLoss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self,
                 alpha=0.5,
                 gamma=1.5,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        bbox_pred, vari_pred = pred[..., :4], torch.clamp(pred[..., 4:], min=-3)
        pred = (bbox_pred, vari_pred)
        loss_bbox = self.loss_weight * balanced_l1_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox