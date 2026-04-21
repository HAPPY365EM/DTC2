import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from skimage import segmentation as skimage_seg


# ---------------------------------------------------------------------------
# Original DTC loss functions — unchanged
# ---------------------------------------------------------------------------

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)
    return ent


def softmax_dice_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n
    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    return kl_div


def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)
    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map


# ---------------------------------------------------------------------------
# Boundary ground-truth generation (numpy, CPU)
# ---------------------------------------------------------------------------

def compute_boundary_gt(label_batch_np):
    """
    Derive a binary boundary map from a batch of segmentation masks.
    Args:
        label_batch_np (np.ndarray): shape (B, H, W, D), values in {0, 1}.
    Returns:
        boundary_gt (np.ndarray): float32, shape (B, H, W, D), values {0, 1}.
    """
    boundary_gt = np.zeros_like(label_batch_np, dtype=np.float32)
    for b in range(label_batch_np.shape[0]):
        posmask = label_batch_np[b].astype(bool)
        if posmask.any():
            boundary = skimage_seg.find_boundaries(
                posmask, mode='inner').astype(np.float32)
            boundary_gt[b] = boundary
    return boundary_gt


# ---------------------------------------------------------------------------
# Adaptive DTC loss
# ---------------------------------------------------------------------------

def adaptive_dtc_loss(dis_to_mask, outputs_soft, boundary_weight_map,
                      alpha=0.3, gamma=0.5):
    """
    Extended DTC consistency loss:

        L = mean( W(x) * S(x) * (dis_to_mask(x) - outputs_soft(x))^2 )

    W(x) = exp(-gamma * |dis_to_mask(x) - outputs_soft(x)|)
        Voxels where the two tasks already agree receive W ≈ 1 (full pressure).
        Voxels with high disagreement receive lower weight, softening the
        gradient where the model is uncertain. W is detached so it acts as a
        data-dependent coefficient and does not accumulate its own gradients.

    S(x) = alpha * B(x) + (1 - alpha)
        Boundary spatial emphasis. B(x) is the GT boundary map for labeled
        samples and the boundary-head prediction for unlabeled samples.

    Args:
        dis_to_mask (Tensor): sigmoid(-1500 * out_tanh), shape (B,1,H,W,D)
            or (B,H,W,D).
        outputs_soft (Tensor): sigmoid(out_seg), same shape as dis_to_mask.
        boundary_weight_map (Tensor): float32 boundary map, shape (B,H,W,D).
        alpha (float): boundary emphasis strength. Default 0.3.
        gamma (float): weighting sharpness. Default 0.5.

    Returns:
        loss (Tensor): scalar.
    """
    if dis_to_mask.dim() == 5:
        dis_to_mask = dis_to_mask[:, 0, ...]
    if outputs_soft.dim() == 5:
        outputs_soft = outputs_soft[:, 0, ...]

    diff = dis_to_mask - outputs_soft

    W = torch.exp(-gamma * torch.abs(diff.detach()))
    S = alpha * boundary_weight_map + (1.0 - alpha)

    loss = torch.mean(W * S * diff ** 2)
    return loss
