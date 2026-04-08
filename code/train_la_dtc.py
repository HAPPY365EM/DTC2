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
from scipy.ndimage import morphology as morph

from networks.vnet_sdf import VNet
from dataloaders import utils
from utils import ramps, losses, metrics
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.util import compute_sdf

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
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='random seed')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float,  default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')

# ========== 新增参数 ==========
parser.add_argument('--lambda_boundary', type=float, default=0.1,
                    help='weight for boundary loss')
parser.add_argument('--gamma_uncertainty', type=float, default=5.0,
                    help='gamma for adaptive weight W(x) = exp(-gamma * diff)')
parser.add_argument('--alpha_boundary', type=float, default=0.5,
                    help='alpha for blending boundary attention')
parser.add_argument('--use_adaptive_weight', action='store_true', default=True,
                    help='use adaptive weight instead of fixed rampup')
parser.add_argument('--use_boundary_attn', action='store_true', default=True,
                    help='use boundary attention map in consistency loss')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "_{}labels_beta_{}/".format(args.labelnum, args.beta)

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
    """原始预热函数，用于边界损失权重或全局一致性系数（可选）"""
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def generate_boundary_gt(label_batch):
    """
    从分割标签生成边界真值（形态学梯度）
    label_batch: torch.Tensor, shape (B, D, H, W) 或 (B, 1, D, H, W), 值为0/1
    return: boundary map, shape (B, D, H, W) 或 (B, 1, D, H, W), 值为0/1
    """
    # 确保输入是 4D 或 5D，统一转为 (B, D, H, W)
    if label_batch.dim() == 5:
        label_batch = label_batch.squeeze(1)
    device = label_batch.device
    label_np = label_batch.cpu().numpy().astype(np.uint8)
    boundary_np = np.zeros_like(label_np)
    struct = morph.generate_binary_structure(3, 1)  # 3x3x3 结构元素
    for i in range(label_np.shape[0]):
        dilated = morph.binary_dilation(label_np[i], struct)
        eroded = morph.binary_erosion(label_np[i], struct)
        boundary_np[i] = (dilated.astype(np.uint8) - eroded.astype(np.uint8))
    boundary_tensor = torch.from_numpy(boundary_np).float().to(device)
    # 增加通道维度 (B, 1, D, H, W)
    return boundary_tensor.unsqueeze(1)


def dice_loss_boundary(pred, target, smooth=1e-5):
    """
    适用于稀疏边界图的 Dice 损失
    pred: sigmoid 输出，shape (B, 1, D, H, W)
    target: binary, shape (B, 1, D, H, W)
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


if __name__ == "__main__":
    # 创建日志和模型保存目录
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition (输出三个分支: 水平集, 分割logits, 边界logits)
        net = VNet(n_channels=1, n_classes=num_classes-1,
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
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

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

            # 前向传播，得到三个输出
            outputs_tanh, outputs, outputs_boundary = model(volume_batch)
            outputs_soft = torch.sigmoid(outputs)  # 分割概率图 (B,1,D,H,W)
            dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)  # 水平集转分割概率 (B,1,D,H,W)

            # ========== 有标签数据的监督损失 ==========
            # 1. 分割 Dice 损失
            loss_seg_dice = losses.dice_loss(
                outputs_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)

            # 2. 水平集回归损失 (SDF)
            with torch.no_grad():
                gt_dis = compute_sdf(label_batch[:labeled_bs].cpu().numpy(),
                                     outputs[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            loss_sdf = mse_loss(outputs_tanh[:labeled_bs, 0, ...], gt_dis)

            # 3. 边界损失 (仅在有标签数据上)
            with torch.no_grad():
                boundary_gt = generate_boundary_gt(label_batch[:labeled_bs])  # (B,1,D,H,W)
            loss_boundary = dice_loss_boundary(torch.sigmoid(outputs_boundary[:labeled_bs]), boundary_gt)

            supervised_loss = loss_seg_dice + args.beta * loss_sdf + args.lambda_boundary * loss_boundary

            # ========== 自适应双任务一致性损失 (同时用于有标签和无标签) ==========
            # 像素级差异图: |分割概率 - 水平集转换概率|
            diff_map = torch.abs(outputs_soft - dis_to_mask)  # (B,1,D,H,W)
            # 自适应权重 W(x) = exp(-gamma * diff)
            gamma = args.gamma_uncertainty
            W = torch.exp(-gamma * diff_map)  # (B,1,D,H,W)

            # 边界注意力图: 从边界预测头生成，用于空间加权
            if args.use_boundary_attn:
                with torch.no_grad():
                    boundary_attn = torch.sigmoid(outputs_boundary)  # (B,1,D,H,W)
                alpha = args.alpha_boundary
                spatial_weight = alpha * boundary_attn + (1 - alpha)
            else:
                spatial_weight = torch.ones_like(W)

            # 加权 MSE 一致性损失
            weighted_mse = (W * spatial_weight) * ((outputs_soft - dis_to_mask) ** 2)
            consistency_loss = weighted_mse.mean()

            # 全局一致性系数：可选预热或固定为1.0，这里使用预热但作用在整体上
            if args.use_adaptive_weight:
                # 自适应权重已经内部调节，全局系数可以固定为1.0或也用预热
                global_cons_weight = 1.0
            else:
                global_cons_weight = get_current_consistency_weight(iter_num // 150)

            loss = supervised_loss + global_cons_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算有标签数据的 Dice 用于监控
            with torch.no_grad():
                pred_label = torch.argmax(outputs_soft[:labeled_bs], dim=1)  # 因为输出是单通道，argmax 得到0/1
                # 实际 outputs_soft 是 (B,1,D,H,W)，因此直接 threshold 0.5
                pred_mask = (outputs_soft[:labeled_bs, 0] > 0.5).float()
                dc = metrics.dice(pred_mask, label_batch[:labeled_bs].float())

            iter_num = iter_num + 1

            # TensorBoard 记录
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_sdf', loss_sdf, iter_num)
            writer.add_scalar('loss/loss_boundary', loss_boundary, iter_num)
            writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('loss/global_cons_weight', global_cons_weight, iter_num)
            writer.add_scalar('loss/adaptive_weight_mean', W.mean(), iter_num)
            writer.add_scalar('metrics/dice_labeled', dc, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_dice: %f, loss_sdf: %f, loss_boundary: %f, consis: %f, dice: %f' %
                (iter_num, loss.item(), loss_seg_dice.item(), loss_sdf.item(),
                 loss_boundary.item(), consistency_loss.item(), dc))

            # 可视化（每50轮）
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = dis_to_mask[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Dis2Mask', grid_image, iter_num)

                image = outputs_tanh[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/DistMap', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = gt_dis[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_DistMap', grid_image, iter_num)

                # 边界预测可视化
                boundary_pred = torch.sigmoid(outputs_boundary[0, 0, ...]).unsqueeze(0).unsqueeze(0)
                image = boundary_pred[:, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Boundary_pred', grid_image, iter_num)

            # 学习率衰减
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            # 保存模型
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    # 保存最终模型
    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save final model to {}".format(save_mode_path))