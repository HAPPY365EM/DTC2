"""
train_ablation.py — Unified training script for ablation variants M0 – M4.

Variant map
-----------
M0  DTC baseline          2-head VNet | uniform DTC MSE | no boundary/aux/HD | no EMA
M1  + Extra heads         4-head VNet | uniform DTC MSE | boundary + aux loss | no HD | no EMA
M2  + Adaptive DTC loss   4-head VNet | adaptive DTC    | boundary + aux loss | no HD | no EMA
M3  + HD loss             4-head VNet | adaptive DTC    | boundary+aux+HD     | no EMA
M4  + EMA teacher         4-head VNet | adaptive DTC    | boundary+aux+HD     | EMA pseudo-label

NOTE: M5 uses the same training checkpoint as M4.
      The only difference is test-time augmentation (TTA) enabled in test_ablation.py.
      Run M4 training, then test with --variant M5 to get M5 results.
"""

import os
import sys
import time
import random
import shutil
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from tqdm import tqdm

# --- project imports ---------------------------------------------------------
from utils import ramps, losses, metrics
from utils.losses import compute_boundary_gt, adaptive_dtc_loss
from utils.losses_2 import hd_loss, compute_dtm
from utils.util import compute_sdf
from dataloaders.la_heart import (
    LAHeart, RandomCrop, CenterCrop, RandomRotFlip,
    RandomNoise, ToTensor, TwoStreamBatchSampler,
)

# =============================================================================
# Variant configuration
# =============================================================================

# Each flag set precisely encodes which components are active.
# This table is the single source of truth — the training loop
# reads only from FLAGS, never from args.variant directly.
VARIANT_FLAGS = {
    'M0': dict(use_4head=False, use_adaptive_dtc=False,
               use_hd=False,    use_ema=False),
    'M1': dict(use_4head=True,  use_adaptive_dtc=False,
               use_hd=False,    use_ema=False),
    'M2': dict(use_4head=True,  use_adaptive_dtc=True,
               use_hd=False,    use_ema=False),
    'M3': dict(use_4head=True,  use_adaptive_dtc=True,
               use_hd=True,     use_ema=False),
    'M4': dict(use_4head=True,  use_adaptive_dtc=True,
               use_hd=True,     use_ema=True),
    # M5 = M4 training weights; TTA differs only at inference (test_ablation.py)
}

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description='Ablation training for improved DTC framework')

parser.add_argument('--variant', type=str, default='M4',
                    choices=list(VARIANT_FLAGS.keys()),
                    help='Ablation variant (M0=baseline, M4=full minus TTA)')
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set/')
parser.add_argument('--exp', type=str, default='LA/Ablation')
parser.add_argument('--max_iterations', type=int, default=6000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--labeled_bs', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--labelnum', type=int, default=16)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--gpu', type=str, default='0')
# Loss weights
parser.add_argument('--beta', type=float, default=0.3,
                    help='Weight for SDF regression loss')
parser.add_argument('--consistency', type=float, default=1.0,
                    help='Max weight for consistency loss')
parser.add_argument('--consistency_rampup', type=float, default=40.0)
parser.add_argument('--boundary_weight', type=float, default=0.3,
                    help='alpha: boundary spatial emphasis in adaptive DTC')
parser.add_argument('--adtc_gamma', type=float, default=0.5,
                    help='gamma: disagreement weighting sharpness in adaptive DTC')
parser.add_argument('--hd_weight', type=float, default=0.1,
                    help='Weight for HD loss (M3/M4 only)')
parser.add_argument('--aux_weight', type=float, default=0.2,
                    help='Weight for auxiliary deep supervision loss (M1-M4)')
parser.add_argument('--ema_decay', type=float, default=0.99)
parser.add_argument('--ema_consistency_weight', type=float, default=0.5,
                    help='Weight for EMA teacher consistency loss (M4 only)')

args = parser.parse_args()

# =============================================================================
# Derived settings
# =============================================================================

FLAGS = VARIANT_FLAGS[args.variant]
USE_4HEAD        = FLAGS['use_4head']
USE_ADAPTIVE_DTC = FLAGS['use_adaptive_dtc']
USE_HD           = FLAGS['use_hd']
USE_EMA          = FLAGS['use_ema']

snapshot_path = (
    f"../model/{args.exp}_{args.variant}"
    f"_{args.labelnum}labels_beta{args.beta}/"
)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size   = args.batch_size * len(args.gpu.split(','))
labeled_bs   = args.labeled_bs
base_lr      = args.base_lr
patch_size   = (112, 112, 80)
num_classes  = 2                 # binary; network n_classes = num_classes - 1 = 1


# =============================================================================
# Helper functions
# =============================================================================

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(
        epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    """Exponential moving average: teacher ← α*teacher + (1-α)*student."""
    alpha = min(1.0 - 1.0 / (global_step + 1), alpha)
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark    = False
    cudnn.deterministic = True

    # ------------------------------------------------------------------
    # Logging / snapshot directory
    # ------------------------------------------------------------------
    os.makedirs(snapshot_path, exist_ok=True)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    ignore=shutil.ignore_patterns('.git', '__pycache__'))

    logging.basicConfig(
        filename=snapshot_path + '/log.txt',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'Variant: {args.variant}  |  Flags: {FLAGS}')
    logging.info(str(args))

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------
    def create_model(ema=False):
        if USE_4HEAD:
            from networks.vnet_sdf import VNet
        else:
            from networks.vnet_base import VNet          # 2-head for M0
        net = VNet(n_channels=1, n_classes=num_classes - 1,
                   normalization='batchnorm', has_dropout=True).cuda()
        if ema:
            for p in net.parameters():
                p.detach_()
        return net

    model = create_model(ema=False)

    # EMA teacher only created for M4
    ema_model = create_model(ema=True) if USE_EMA else None

    # ------------------------------------------------------------------
    # Dataset / dataloader
    # ------------------------------------------------------------------
    db_train = LAHeart(
        base_dir=args.root_path,
        split='train',
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomNoise(),
            RandomCrop(patch_size),
            ToTensor(),
        ]))

    labeled_idxs   = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, 80))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        batch_size, batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn)

    # ------------------------------------------------------------------
    # Optimiser / loss functions
    # ------------------------------------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=1e-4)
    ce_loss  = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    writer   = SummaryWriter(snapshot_path + '/log')
    logging.info(f'{len(trainloader)} iterations per epoch')

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    if ema_model is not None:
        ema_model.train()

    iter_num  = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    lr_       = base_lr
    iterator  = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch = sampled_batch['image'].cuda()
            label_batch  = sampled_batch['label'].cuda()

            # ==============================================================
            # 1. Forward pass
            # ==============================================================
            if USE_4HEAD:
                outputs_tanh, outputs, outputs_boundary, outputs_aux = \
                    model(volume_batch)
                # Upsample auxiliary output to full resolution
                outputs_aux_up   = F.interpolate(
                    outputs_aux, size=patch_size,
                    mode='trilinear', align_corners=False)
                outputs_aux_soft = torch.sigmoid(outputs_aux_up)
            else:
                # M0: 2-head VNet
                outputs_tanh, outputs = model(volume_batch)
                outputs_boundary = None
                outputs_aux      = None

            outputs_soft = torch.sigmoid(outputs)          # (B,1,H,W,D)

            # ==============================================================
            # 2. Ground-truth generation  (no_grad — CPU-side transforms)
            # ==============================================================
            with torch.no_grad():
                # SDF ground truth — required by all variants
                gt_dis_np = compute_sdf(
                    label_batch[:labeled_bs].cpu().numpy(),
                    outputs[:labeled_bs, 0].shape)
                gt_dis = torch.from_numpy(gt_dis_np).float().cuda()

                # Boundary GT — required by M1-M4
                if USE_4HEAD:
                    gt_boundary_np = compute_boundary_gt(
                        label_batch[:labeled_bs].cpu().numpy())
                    gt_boundary = torch.from_numpy(
                        gt_boundary_np).float().cuda()

                # Distance transform map — required by M3/M4 (HD loss)
                if USE_HD:
                    gt_dtm_np = compute_dtm(
                        label_batch[:labeled_bs].cpu().numpy(),
                        outputs[:labeled_bs, 0].shape,
                        normalize=True)
                    gt_dtm = torch.from_numpy(gt_dtm_np).float().cuda()

            # ==============================================================
            # 3. Individual loss computation
            # ==============================================================

            # --- Always-on supervised losses (all variants) ---
            loss_sdf = mse_loss(
                outputs_tanh[:labeled_bs, 0], gt_dis)
            loss_seg = ce_loss(
                outputs[:labeled_bs, 0],
                label_batch[:labeled_bs].float())
            loss_seg_dice = losses.dice_loss(
                outputs_soft[:labeled_bs, 0],
                label_batch[:labeled_bs] == 1)

            # --- Boundary head loss (M1-M4) ---
            if USE_4HEAD:
                n_pos = gt_boundary.sum().clamp(min=1.0)
                n_neg = (1.0 - gt_boundary).sum()
                boundary_pos_weight = (n_neg / n_pos).clamp(max=50.0)
                loss_boundary = F.binary_cross_entropy_with_logits(
                    outputs_boundary[:labeled_bs, 0],
                    gt_boundary,
                    pos_weight=boundary_pos_weight)

                # Auxiliary deep supervision loss (mid-decoder Dice)
                loss_aux = losses.dice_loss(
                    outputs_aux_soft[:labeled_bs, 0],
                    label_batch[:labeled_bs] == 1)

            # --- Hausdorff distance loss (M3/M4) ---
            if USE_HD:
                loss_hd = hd_loss(
                    outputs_soft[:labeled_bs, 0],
                    label_batch[:labeled_bs],
                    gt_dtm=gt_dtm,
                    one_side=True)

            # ==============================================================
            # 4. Consistency loss (DTC)
            # ==============================================================
            dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)   # T^{-1}(f2)

            if USE_ADAPTIVE_DTC:
                # Build boundary weight map:
                #   labeled   samples → GT boundary map
                #   unlabeled samples → predicted boundary (detached)
                with torch.no_grad():
                    boundary_pred_unlabeled = torch.sigmoid(
                        outputs_boundary[labeled_bs:, 0])
                boundary_weight_map = torch.cat(
                    [gt_boundary, boundary_pred_unlabeled], dim=0)

                consistency_loss = adaptive_dtc_loss(
                    dis_to_mask,
                    outputs_soft,
                    boundary_weight_map,
                    alpha=args.boundary_weight,
                    gamma=args.adtc_gamma)
            else:
                # M0 / M1: original uniform MSE
                consistency_loss = torch.mean(
                    (dis_to_mask[:, 0] - outputs_soft[:, 0]) ** 2)

            # ==============================================================
            # 5. EMA teacher pseudo-label consistency (M4 only)
            # ==============================================================
            if USE_EMA:
                with torch.no_grad():
                    if USE_4HEAD:
                        ema_tanh, ema_out, _, _ = ema_model(
                            volume_batch[labeled_bs:])
                    else:
                        ema_tanh, ema_out = ema_model(
                            volume_batch[labeled_bs:])
                    ema_soft   = torch.sigmoid(ema_out)
                    ema_pseudo = (ema_soft[:, 0] > 0.5).float()

                loss_ema_consistency = losses.dice_loss(
                    outputs_soft[labeled_bs:, 0], ema_pseudo)

            # ==============================================================
            # 6. Supervised loss (variant-specific composition)
            # ==============================================================
            supervised_loss = (
                0.5 * loss_seg
                + loss_seg_dice
                + args.beta * loss_sdf
            )
            if USE_4HEAD:
                supervised_loss += (
                    0.1             * loss_boundary
                    + args.aux_weight * loss_aux
                )
            if USE_HD:
                supervised_loss += args.hd_weight * loss_hd

            # ==============================================================
            # 7. Total loss
            # ==============================================================
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)

            loss = supervised_loss + consistency_weight * consistency_loss
            if USE_EMA:
                loss += (consistency_weight
                         * args.ema_consistency_weight
                         * loss_ema_consistency)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA teacher
            if USE_EMA:
                update_ema_variables(model, ema_model,
                                     alpha=args.ema_decay,
                                     global_step=iter_num)

            # ==============================================================
            # 8. Logging
            # ==============================================================
            iter_num += 1

            writer.add_scalar('lr',                lr_,               iter_num)
            writer.add_scalar('loss/total',        loss,              iter_num)
            writer.add_scalar('loss/seg_dice',     loss_seg_dice,     iter_num)
            writer.add_scalar('loss/seg_bce',      loss_seg,          iter_num)
            writer.add_scalar('loss/sdf',          loss_sdf,          iter_num)
            writer.add_scalar('loss/consistency',  consistency_loss,  iter_num)
            writer.add_scalar('loss/cons_weight',  consistency_weight,iter_num)

            if USE_4HEAD:
                writer.add_scalar('loss/boundary', loss_boundary, iter_num)
                writer.add_scalar('loss/aux',      loss_aux,      iter_num)
            if USE_HD:
                writer.add_scalar('loss/hd',       loss_hd,       iter_num)
            if USE_EMA:
                writer.add_scalar('loss/ema_cons', loss_ema_consistency,
                                  iter_num)

            log_msg = (
                f'iter {iter_num:5d} [{args.variant}] '
                f'loss={loss.item():.4f}  dice={loss_seg_dice.item():.4f}  '
                f'sdf={loss_sdf.item():.4f}  dtc={consistency_loss.item():.4f}'
            )
            if USE_4HEAD:
                log_msg += (f'  bnd={loss_boundary.item():.4f}'
                            f'  aux={loss_aux.item():.4f}')
            if USE_HD:
                log_msg += f'  hd={loss_hd.item():.4f}'
            if USE_EMA:
                log_msg += f'  ema={loss_ema_consistency.item():.4f}'
            log_msg += f'  w={consistency_weight:.4f}'
            logging.info(log_msg)

            # Learning-rate decay (same schedule as original DTC)
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                logging.info(f'lr decayed to {lr_:.6f}')

            # Checkpoint every 1000 iterations
            if iter_num % 1000 == 0:
                ckpt_path = os.path.join(
                    snapshot_path, f'iter_{iter_num}.pth')
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f'Saved checkpoint: {ckpt_path}')

            if iter_num >= args.max_iterations:
                break

        if iter_num >= args.max_iterations:
            iterator.close()
            break

    writer.close()
    logging.info('Training complete.')
