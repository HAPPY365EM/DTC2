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
# NEW — Improvement 1: Boundary ground-truth generation (numpy, CPU)
# ---------------------------------------------------------------------------

def compute_boundary_gt(label_batch_np):
    """
    Derive a binary boundary map from a batch of segmentation masks.

    Boundary voxels are defined as the inner border of the foreground region,
    consistent with how the SDF ground truth sets boundary voxels to zero.
    The result is used both as supervision for the boundary head (Task 3)
    and to build the spatial emphasis map B(x) for the adaptive DTC loss.

    Args:
        label_batch_np (np.ndarray): integer segmentation masks,
            shape = (B, H, W, D), values in {0, 1}.

    Returns:
        boundary_gt (np.ndarray): float32 binary boundary map,
            shape = (B, H, W, D), values in {0.0, 1.0}.
    """
    boundary_gt = np.zeros_like(label_batch_np, dtype=np.float32)
    for b in range(label_batch_np.shape[0]):
        posmask = label_batch_np[b].astype(bool)
        if posmask.any():
            # find_boundaries with mode='inner' marks voxels on the
            # foreground side of the boundary — consistent with SDF = 0 locus
            boundary = skimage_seg.find_boundaries(
                posmask, mode='inner').astype(np.float32)
            boundary_gt[b] = boundary
    return boundary_gt


# ---------------------------------------------------------------------------
# NEW — Improvement 2: Adaptive DTC loss with boundary spatial emphasis
# ---------------------------------------------------------------------------

def adaptive_dtc_loss(dis_to_mask, outputs_soft, boundary_weight_map,
                      alpha=0.3, gamma=0.5):
    """
    Replaces the original uniform `torch.mean((dis_to_mask - outputs_soft)**2)`.

    Two innovations applied jointly:

    (a) Task-disagreement adaptive weighting W(x):
        Voxels where the two tasks already agree strongly receive a higher
        consistency weight; high-disagreement voxels are down-weighted as
        unreliable and treated as uncertain.
            W(x) = exp(-gamma * |dis_to_mask(x) - outputs_soft(x)|)
        This is parameter-free and adds zero extra computation beyond the
        existing difference map that the original loss already computes.

    (b) Boundary spatial emphasis B(x):
        The boundary weight map (derived from GT on labeled samples, or a
        uniform ones-map on unlabeled samples) amplifies consistency pressure
        near boundaries, directly targeting ASD and 95HD improvement.
        The combined spatial weight is:
            S(x) = alpha * B(x) + (1 - alpha)
        so interior voxels receive weight (1-alpha) and boundary voxels
        receive weight 1.0, with alpha controlling the emphasis strength.

    Final loss:
        L_adtc = mean( W(x) * S(x) * (dis_to_mask(x) - outputs_soft(x))^2 )

    Args:
        dis_to_mask (Tensor): T^{-1}(f2(x)), sigmoid(-1500 * out_tanh),
            shape = (B, 1, H, W, D) or (B, H, W, D).
        outputs_soft (Tensor): sigmoid(out_seg),
            same shape as dis_to_mask.
        boundary_weight_map (Tensor): float32 spatial boundary map,
            shape = (B, H, W, D). Use GT boundary on labeled samples,
            torch.ones_like(...) on unlabeled samples.
        alpha (float): boundary emphasis strength in [0, 1].
            alpha=0 reduces to pure adaptive weighting (no spatial emphasis).
            alpha=1 makes boundary voxels receive 2x weight vs. interior.
            Default 0.3.
        gamma (float): sharpness of adaptive disagreement weighting.
            Higher gamma → more aggressive down-weighting of uncertain voxels.
            Default 0.5.

    Returns:
        loss (Tensor): scalar.
    """
    # Squeeze channel dim if present so shapes align with boundary_weight_map
    if dis_to_mask.dim() == 5:
        dis_to_mask = dis_to_mask[:, 0, ...]       # (B, H, W, D)
    if outputs_soft.dim() == 5:
        outputs_soft = outputs_soft[:, 0, ...]      # (B, H, W, D)

    diff = dis_to_mask - outputs_soft               # (B, H, W, D)

    # (a) Task-disagreement adaptive weight — no extra parameters
    W = torch.exp(-gamma * torch.abs(diff.detach()))  # detach: W is not trained

    # (b) Boundary spatial emphasis
    S = alpha * boundary_weight_map + (1.0 - alpha)   # in [1-alpha, 1]

    loss = torch.mean(W * S * diff ** 2)
    return loss