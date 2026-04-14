import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet_sdf import VNet
from dataloaders import utils
from utils import ramps, losses, metrics
from dataloaders.la_heart import (LAHeart, RandomCrop, CenterCrop,
                                   RandomRotFlip, ToTensor, TwoStreamBatchSampler)
from utils.util import compute_sdf
# NEW: import the helper functions added to losses.py
from utils.losses import compute_boundary_gt, adaptive_dtc_loss
from utils.losses_2 import hd_loss, compute_dtm

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/DTC_improved', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float, default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16, help='random seed')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float, default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--beta', type=float, default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')

# NEW arguments for the two improvements
parser.add_argument('--boundary_weight', type=float, default=0.3,
                    help='alpha: boundary spatial emphasis strength in adaptive DTC loss. '
                         'Interior voxels receive weight (1-alpha), boundary voxels receive 1.0. '
                         'Range [0,1]. Default 0.3.')
parser.add_argument('--adtc_gamma', type=float, default=0.5,
                    help='gamma: sharpness of task-disagreement adaptive weighting. '
                         'W(x) = exp(-adtc_gamma * |dis_to_mask - outputs_soft|). '
                         'Higher = more aggressive down-weighting of uncertain voxels. '
                         'Default 0.5.')
# NEW: Hausdorff distance loss weight
parser.add_argument('--hd_weight', type=float, default=0.3,
                    help='Weight of the Hausdorff distance loss (supervised, labeled only). '
                         'Directly penalises large surface deviations to reduce 95HD. '
                         'DTM is normalized to [0,1] so squared values are at most 1.0. '
                         'Default 0.3.')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + \
    "_{}labels_beta_{}/".format(args.labelnum, args.beta)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
    # Global ramp-up envelope unchanged from original DTC
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # VNet now returns three outputs: out_tanh, out_seg, out_boundary
        net = VNet(n_channels=1, n_classes=num_classes - 1,
                   normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 80))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # Forward pass — now unpacks three outputs
            outputs_tanh, outputs, outputs_boundary = model(volume_batch)
            outputs_soft = torch.sigmoid(outputs)

            # ------------------------------------------------------------------
            # Supervised losses on labeled samples only (indices :labeled_bs)
            # ------------------------------------------------------------------
            with torch.no_grad():
                # SDF ground truth for the LSF regression head (Task 2)
                gt_dis = compute_sdf(
                    label_batch[:labeled_bs].cpu().numpy(),
                    outputs[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()

                # NEW: boundary ground truth for the boundary head (Task 3)
                # compute_boundary_gt runs on CPU numpy; no GPU memory overhead
                gt_boundary_np = compute_boundary_gt(
                    label_batch[:labeled_bs].cpu().numpy())
                gt_boundary = torch.from_numpy(gt_boundary_np).float().cuda()
                # shape: (labeled_bs, H, W, D)

                # NEW: Distance Transform Map ground truth for HD loss
                # compute_dtm returns a DTM where boundary=0, interior/exterior
                # grow by Euclidean distance — used to weight surface errors.
                # normalize=True scales values to [0,1] so that after squaring
                # inside hd_loss the magnitude stays comparable to dice_loss.
                # Without normalization, raw voxel distances (~50) squared (~2500)
                # overwhelm all other losses even at hd_weight=0.1.
                # one_side=True in hd_loss uses only gt_dtm (stable; no predicted
                # DTM noise).
                gt_dtm_np = compute_dtm(
                    label_batch[:labeled_bs].cpu().numpy(),
                    outputs[:labeled_bs, 0, ...].shape,
                    normalize=True)
                gt_dtm = torch.from_numpy(gt_dtm_np).float().cuda()
                # shape: (labeled_bs, H, W, D), values in [0, 1]

            # Task 2: SDF regression loss (unchanged)
            loss_sdf = mse_loss(outputs_tanh[:labeled_bs, 0, ...], gt_dis)

            # Task 1: segmentation losses (unchanged)
            loss_seg = ce_loss(
                outputs[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice = losses.dice_loss(
                outputs_soft[:labeled_bs, 0, :, :, :],
                label_batch[:labeled_bs] == 1)

            # NEW Task 3: boundary BCE loss on labeled samples only.
            # Boundary voxels are ~1-3% of a 3D volume, so a flat BCE will find
            # it near-optimal to predict all-background and the head collapses.
            # pos_weight = (#neg / #pos) re-balances the loss so both classes
            # contribute equally regardless of their volume fraction.
            # Clamped to 50 to prevent exploding gradients on nearly-empty crops.
            n_pos = gt_boundary.sum().clamp(min=1.0)
            n_neg = (1.0 - gt_boundary).sum()
            boundary_pos_weight = (n_neg / n_pos).clamp(max=50.0)
            loss_boundary = F.binary_cross_entropy_with_logits(
                outputs_boundary[:labeled_bs, 0, ...],
                gt_boundary,
                pos_weight=boundary_pos_weight)

            # NEW: Hausdorff distance loss on labeled samples only
            # hd_loss weights segmentation errors by squared distance from
            # the GT surface (via gt_dtm). Voxels far from the boundary
            # contribute quadratically more, directly penalising the large
            # surface deviations that inflate 95HD.
            # one_side=True uses only gt_dtm (stable; no predicted DTM noise).
            loss_hd = hd_loss(
                outputs_soft[:labeled_bs, 0, ...],
                label_batch[:labeled_bs],
                gt_dtm=gt_dtm,
                one_side=True)

            # ------------------------------------------------------------------
            # Consistency loss on ALL samples (labeled + unlabeled)
            # ------------------------------------------------------------------
            # T^{-1} transform: convert LSF output to probability space
            dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)

            # NEW: build the spatial boundary emphasis map S(x) for the full batch.
            # For labeled samples: use gt_boundary derived from GT labels.
            # For unlabeled samples: use a uniform ones-map (no GT available),
            # which reduces adaptive_dtc_loss to pure adaptive weighting for them.
            ones_unlabeled = torch.ones(
                batch_size - labeled_bs,
                *gt_boundary.shape[1:],
                device=gt_boundary.device)
            boundary_weight_map = torch.cat(
                [gt_boundary, ones_unlabeled], dim=0)
            # shape: (batch_size, H, W, D)

            # NEW: adaptive DTC loss replacing the original uniform MSE
            # Original: consistency_loss = torch.mean((dis_to_mask - outputs_soft)**2)
            consistency_loss = adaptive_dtc_loss(
                dis_to_mask,
                outputs_soft,
                boundary_weight_map,
                alpha=args.boundary_weight,   # boundary spatial emphasis
                gamma=args.adtc_gamma)        # disagreement weighting sharpness

            # ------------------------------------------------------------------
            # Total loss
            # ------------------------------------------------------------------
            # Supervised losses (labeled samples only):
            #   loss_seg      — BCE per-voxel: dense gradient signal, fast convergence
            #   loss_seg_dice — Dice: region-normalised, robust to class imbalance
            #   loss_sdf      — SDF regression: geometric shape constraint (Task 2)
            #   loss_boundary — boundary BCE (pos_weight balanced): surface signal
            #                   weighted 0.1 to prevent high-variance boundary
            #                   gradients from drowning out Dice + SDF
            #   loss_hd       — Hausdorff loss: penalises large surface deviations
            #                   (DTM normalized to [0,1] so hd_weight=0.3 is safe)
            supervised_loss = (loss_seg
                               + loss_seg_dice
                               + args.beta * loss_sdf
                               + 0.1 * loss_boundary
                               + args.hd_weight * loss_hd)

            # Global ramp-up weight — unchanged from original DTC
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dc = metrics.dice(
                torch.argmax(outputs_soft[:labeled_bs], dim=1),
                label_batch[:labeled_bs])

            iter_num = iter_num + 1

            # ------------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------------
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('loss/loss_boundary', loss_boundary, iter_num)
            writer.add_scalar('loss/loss_hd', loss_hd, iter_num)            # NEW
            writer.add_scalar('loss/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)

            logging.info(
                'iteration %d : loss: %.4f  loss_consis: %.4f  loss_haus: %.4f  '
                'loss_seg: %.4f  loss_dice: %.4f  loss_boundary: %.4f  loss_hd: %.4f' %
                (iter_num, loss.item(), consistency_loss.item(), loss_sdf.item(),
                 loss_seg.item(), loss_seg_dice.item(), loss_boundary.item(),
                 loss_hd.item()))

            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = dis_to_mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Dis2Mask', grid_image, iter_num)

                image = outputs_tanh[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/DistMap', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = gt_dis[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_DistMap', grid_image, iter_num)

                # NEW: visualise predicted boundary and GT boundary
                image = torch.sigmoid(outputs_boundary[0, 0:1, :, :, 20:61:10]).permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_boundary', grid_image, iter_num)

                image = gt_boundary[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_boundary', grid_image, iter_num)

            # Learning rate decay
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 1000 == 0:
                # Save periodic checkpoint — consistent with original DTC paper.
                # iter_6000.pth is the primary reported result, matching the
                # evaluation protocol of UA-MT, SASSNet, and DTC.
                # No test-set-informed model selection is performed here to
                # avoid data leakage between validation and final evaluation.
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
