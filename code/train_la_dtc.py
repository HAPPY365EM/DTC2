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
                                   RandomRotFlip, RandomNoise,
                                   ToTensor, TwoStreamBatchSampler)
from utils.util import compute_sdf
from utils.losses import compute_boundary_gt, adaptive_dtc_loss
from utils.losses_2 import hd_loss, compute_dtm

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set/',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/DTC_improved_fixed', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000,
                    help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='base learning rate')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16,
                    help='number of labeled samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--beta', type=float, default=0.3,
                    help='weight for SDF regression loss')
parser.add_argument('--consistency', type=float, default=1.0,
                    help='consistency loss max weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0,
                    help='consistency ramp-up length in epochs')
parser.add_argument('--boundary_weight', type=float, default=0.3,
                    help='alpha: boundary spatial emphasis in adaptive DTC loss')
parser.add_argument('--hd_weight', type=float, default=0.05,
                    help='weight for Hausdorff distance loss (reduced to prevent boundary suppression)')
parser.add_argument('--ema_decay', type=float, default=0.99,
                    help='EMA decay rate for mean teacher')
parser.add_argument('--ema_consistency_weight', type=float, default=0.5,
                    help='weight for EMA teacher consistency on unlabeled data')
parser.add_argument('--aux_weight', type=float, default=0.2,
                    help='weight for auxiliary deep supervision Dice loss')

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
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Fix: Update BOTH parameters and buffers (BatchNorm stats).
    This is critical for the Teacher model to work correctly.
    """
    alpha = min(1.0 - 1.0 / (global_step + 1), alpha)
    
    # Update parameters (weights & biases)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)
    
    # Update buffers (running_mean, running_var for BatchNorm)
    for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
        if buf.dtype == torch.float32:
            ema_buf.data.mul_(alpha).add_(buf.data, alpha=1.0 - alpha)
        else:
            # For integer buffers like num_batches_tracked, just copy
            ema_buf.data.copy_(buf.data)


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        net = VNet(n_channels=1, n_classes=num_classes - 1,
                   normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model(ema=False)     # student
    ema_model = create_model(ema=True)  # teacher

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomNoise(),
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
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()

            volume_batch, label_batch = (sampled_batch['image'],
                                         sampled_batch['label'])
            volume_batch, label_batch = (volume_batch.cuda(),
                                         label_batch.cuda())

            # Student forward pass
            outputs_tanh, outputs, outputs_boundary, outputs_aux = \
                model(volume_batch)
            outputs_soft = torch.sigmoid(outputs)

            # Upsample auxiliary output
            outputs_aux_up = F.interpolate(
                outputs_aux, size=patch_size, mode='trilinear',
                align_corners=False)
            outputs_aux_soft = torch.sigmoid(outputs_aux_up)

            # ------------------------------------------------------------------
            # Ground-truth preparation
            # ------------------------------------------------------------------
            with torch.no_grad():
                gt_dis = compute_sdf(
                    label_batch[:labeled_bs].cpu().numpy(),
                    outputs[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()

                gt_boundary_np = compute_boundary_gt(
                    label_batch[:labeled_bs].cpu().numpy())
                gt_boundary = torch.from_numpy(gt_boundary_np).float().cuda()

                gt_dtm_np = compute_dtm(
                    label_batch[:labeled_bs].cpu().numpy(),
                    outputs[:labeled_bs, 0, ...].shape,
                    normalize=True)
                gt_dtm = torch.from_numpy(gt_dtm_np).float().cuda()

            # ------------------------------------------------------------------
            # Supervised losses (labeled samples only)
            # ------------------------------------------------------------------
            loss_sdf = mse_loss(outputs_tanh[:labeled_bs, 0, ...], gt_dis)

            loss_seg = ce_loss(
                outputs[:labeled_bs, 0, ...],
                label_batch[:labeled_bs].float())
            loss_seg_dice = losses.dice_loss(
                outputs_soft[:labeled_bs, 0, ...],
                label_batch[:labeled_bs] == 1)

            # Boundary BCE
            n_pos = gt_boundary.sum().clamp(min=1.0)
            n_neg = (1.0 - gt_boundary).sum()
            boundary_pos_weight = (n_neg / n_pos).clamp(max=50.0)
            loss_boundary = F.binary_cross_entropy_with_logits(
                outputs_boundary[:labeled_bs, 0, ...],
                gt_boundary,
                pos_weight=boundary_pos_weight)

            loss_hd = hd_loss(
                outputs_soft[:labeled_bs, 0, ...],
                label_batch[:labeled_bs],
                gt_dtm=gt_dtm,
                one_side=True)

            loss_aux = losses.dice_loss(
                outputs_aux_soft[:labeled_bs, 0, ...],
                label_batch[:labeled_bs] == 1)

            # ------------------------------------------------------------------
            # Consistency losses (all samples)
            # ------------------------------------------------------------------
            dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)

            # --- FIX: Boundary Weight Map Construction ---
            # Labeled: use GT boundary.
            # Unlabeled: use ONES map (no bias). DO NOT use predicted boundary for weighting.
            boundary_weight_map = torch.cat(
                [gt_boundary, torch.ones_like(gt_boundary).repeat(batch_size - labeled_bs, 1, 1, 1)], dim=0)

            # Adaptive DTC Loss (Simplified)
            consistency_loss = adaptive_dtc_loss(
                dis_to_mask,
                outputs_soft,
                boundary_weight_map,
                alpha=args.boundary_weight)

            # --- EMA Mean Teacher Consistency ---
            with torch.no_grad():
                ema_tanh, ema_out, ema_boundary, _ = ema_model(volume_batch[labeled_bs:])
                ema_soft = torch.sigmoid(ema_out)
                ema_pseudo = (ema_soft[:, 0, ...] > 0.5).float()
                
                # Boundary pseudo-labels from teacher
                ema_boundary_prob = torch.sigmoid(ema_boundary)

            # Segmentation Consistency (Dice on hard pseudo-labels)
            loss_ema_seg = losses.dice_loss(
                outputs_soft[labeled_bs:, 0, ...],
                ema_pseudo)

            # --- NEW: Boundary Consistency (MSE on soft pseudo-labels) ---
            # Helps the boundary head learn on unlabeled data
            loss_ema_boundary = F.mse_loss(
                torch.sigmoid(outputs_boundary[labeled_bs:, 0, ...]),
                ema_boundary_prob
            )

            # ------------------------------------------------------------------
            # Total loss
            # ------------------------------------------------------------------
            supervised_loss = (
                0.5 * loss_seg
                + loss_seg_dice
                + args.beta * loss_sdf
                + 0.1 * loss_boundary
                + args.hd_weight * loss_hd
                + args.aux_weight * loss_aux
            )

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss = (supervised_loss
                    + consistency_weight * consistency_loss
                    + consistency_weight * args.ema_consistency_weight * loss_ema_seg
                    + consistency_weight * 0.1 * loss_ema_boundary) # Small weight for boundary consistency

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model,
                                 alpha=args.ema_decay,
                                 global_step=iter_num)

            dc = metrics.dice(
                torch.argmax(outputs_soft[:labeled_bs], dim=1),
                label_batch[:labeled_bs])

            iter_num += 1

            # ------------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------------
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/total', loss, iter_num)
            writer.add_scalar('loss/seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/sdf', loss_sdf, iter_num)
            writer.add_scalar('loss/boundary', loss_boundary, iter_num)
            writer.add_scalar('loss/hd', loss_hd, iter_num)
            writer.add_scalar('loss/consistency_dtc', consistency_loss, iter_num)
            writer.add_scalar('loss/ema_seg', loss_ema_seg, iter_num)
            writer.add_scalar('loss/ema_boundary', loss_ema_boundary, iter_num)

            if iter_num % 50 == 0:
                # Optional: Add logging images here as in original script
                pass

            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                logging.info('lr decayed to {:.6f}'.format(lr_))

            if iter_num % 1000 == 0:
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
