"""
test_ablation.py — Unified test script for ablation variants M0 – M4.

All variants are evaluated under the same inference protocol:
    - 4‑fold test‑time augmentation (TTA)
    - Dual‑head ensemble (segmentation probability + SDF‑derived probability)

Differences among variants are only in the trained model weights (number of
heads, training components). The evaluation protocol remains identical.
"""

import argparse
import math
import os

import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm

# =============================================================================
# Fixed inference settings
# =============================================================================

USE_TTA      = True   # 4‑fold flip TTA (W and H axes only)
USE_ENSEMBLE = True   # blend seg and SDF heads

# TTA flip combinations — W and H axes only (D excluded; see test_util.py for rationale)
TTA_FLIP_AXES = [
    [],       # original
    [2],      # flip W
    [3],      # flip H
    [2, 3],   # flip W + H
]

# =============================================================================
# Variant → model architecture mapping
# =============================================================================

# Only need to know whether the model has 2 or 4 outputs
USE_4HEAD_MAP = {
    'M0': False,
    'M1': True,
    'M2': True,
    'M3': True,
    'M4': True,
}

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=str, default='M4',
                    choices=['M0', 'M1', 'M2', 'M3', 'M4'])
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to saved model checkpoint (.pth)')
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set/')
parser.add_argument('--test_list', type=str,
                    default='../data/2018LA_Seg_Training Set/test.list',
                    help='Text file with one test volume path per line')
parser.add_argument('--save_result', type=int, default=1)
parser.add_argument('--test_save_path', type=str, default=None)
parser.add_argument('--patch_size', nargs=3, type=int,
                    default=[112, 112, 80])
parser.add_argument('--stride_xy', type=int, default=18)
parser.add_argument('--stride_z',  type=int, default=4)
parser.add_argument('--nms', type=int, default=0,
                    help='Apply largest-connected-component post-processing')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

# Derived
USE_4HEAD = USE_4HEAD_MAP[args.variant]

# =============================================================================
# Utilities
# =============================================================================

def getLargestCC(segmentation):
    """Keep only the largest connected component."""
    labels = label(segmentation)
    assert labels.max() != 0, 'No foreground voxels in prediction'
    largest = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc   = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd  = metric.binary.asd(pred, gt)
    return dice, jc, hd95, asd


# =============================================================================
# Core inference — one 3D volume
# =============================================================================

def _forward_single(net, patch, use_4head):
    """
    Single forward pass (no TTA). Returns a probability map of shape (1, H, W, D).
    """
    if use_4head:
        y_tanh, y_seg, _, _ = net(patch)
    else:
        y_tanh, y_seg = net(patch)

    prob_seg = torch.sigmoid(y_seg)                     # (1, 1, H, W, D)
    prob_sdf = torch.sigmoid(-1500 * y_tanh)            # (1, 1, H, W, D)

    if USE_ENSEMBLE:
        prob = 0.5 * prob_seg + 0.5 * prob_sdf
    else:
        prob = prob_seg
    return prob   # (1, 1, H, W, D)


def test_single_case(net, image, stride_xy, stride_z, patch_size,
                     num_classes=1, use_4head=False):
    """
    Sliding‑window inference over one volume with TTA + dual‑head ensemble.

    Args:
        net         : trained model in eval mode.
        image       : numpy array (W, H, D).
        stride_xy   : sliding‑window stride in W/H dimensions.
        stride_z    : sliding‑window stride in D dimension.
        patch_size  : (W, H, D) tuple.
        num_classes : number of foreground classes (1 for binary).
        use_4head   : True for 4‑output network, False for 2‑head.

    Returns:
        label_map  (np.ndarray): binary prediction, shape (W, H, D).
        score_map  (np.ndarray): probability map, shape (1, W, H, D).
    """
    w, h, d = image.shape

    # --- Pad if needed ---
    pad = [(0, 0), (0, 0), (0, 0)]
    padded = False
    for axis, (sz, psz) in enumerate(zip((w, h, d), patch_size)):
        if sz < psz:
            pad[axis] = (0, psz - sz)
            padded = True

    if padded:
        image = np.pad(image, pad, mode='constant', constant_values=0)

    ww, hh, dd = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z)  + 1

    score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    cnt        = np.zeros(image.shape, dtype=np.float32)

    for xi in range(sx):
        xs = min(stride_xy * xi, ww - patch_size[0])
        for yi in range(sy):
            ys = min(stride_xy * yi, hh - patch_size[1])
            for zi in range(sz):
                zs = min(stride_z * zi, dd - patch_size[2])

                patch_np = image[xs:xs + patch_size[0],
                                 ys:ys + patch_size[1],
                                 zs:zs + patch_size[2]]
                patch_t  = torch.from_numpy(
                    patch_np[None, None].astype(np.float32)).cuda()

                with torch.no_grad():
                    if USE_TTA:
                        # 4‑fold TTA: average over flip augmentations.
                        # acc shape: (1, 1, H, W, D) — matches _forward_single output.
                        acc = torch.zeros(1, 1, *patch_t.shape[2:],
                                         device=patch_t.device)
                        for flip_axes in TTA_FLIP_AXES:
                            aug = (torch.flip(patch_t, flip_axes)
                                   if flip_axes else patch_t)
                            prob_aug = _forward_single(net, aug, use_4head)
                            if flip_axes:
                                prob_aug = torch.flip(prob_aug, flip_axes)
                            acc += prob_aug
                        # Average and take the foreground class probability
                        prob_final = (acc / len(TTA_FLIP_AXES))[0, 0].cpu().numpy()
                    else:
                        prob = _forward_single(net, patch_t, use_4head)
                        prob_final = prob[0, 0].cpu().numpy()  # (H, W, D)

                score_map[:, xs:xs + patch_size[0],
                             ys:ys + patch_size[1],
                             zs:zs + patch_size[2]] += prob_final
                cnt[xs:xs + patch_size[0],
                    ys:ys + patch_size[1],
                    zs:zs + patch_size[2]] += 1

    score_map /= cnt[None]
    label_map  = (score_map[0] > 0.5).astype(np.int64)

    # Strip padding if it was added
    if padded:
        (wl, _), (hl, _), (dl, _) = (
            (pad[0][0], None), (pad[1][0], None), (pad[2][0], None))
        label_map = label_map[wl:wl + w, hl:hl + h, dl:dl + d]
        score_map = score_map[:, wl:wl + w, hl:hl + h, dl:dl + d]

    return label_map, score_map


# =============================================================================
# Batch evaluation
# =============================================================================

def test_all_case(net, image_list, num_classes, patch_size,
                  stride_xy, stride_z,
                  use_4head,
                  save_result=True, test_save_path=None,
                  metric_detail=False, nms=False):
    """
    Evaluate the model on all volumes in image_list.

    Returns:
        avg_metric (np.ndarray): mean [Dice, Jaccard, 95HD, ASD].
    """
    total_metric = np.zeros(4)
    loader = image_list if metric_detail else tqdm(image_list)

    for ith, image_path in enumerate(loader):
        with h5py.File(image_path, 'r') as f:
            image    = f['image'][:]
            label_gt = f['label'][:]

        prediction, score_map = test_single_case(
            net, image, stride_xy, stride_z, patch_size,
            num_classes=num_classes, use_4head=use_4head)

        if nms:
            prediction = getLargestCC(prediction)

        if np.sum(prediction) == 0:
            single_metric = (0.0, 0.0, 0.0, 0.0)
        else:
            single_metric = calculate_metric_percase(prediction, label_gt)

        if metric_detail:
            print('%02d\tDice=%.4f  Jc=%.4f  HD95=%.4f  ASD=%.4f' % (
                ith, *single_metric))

        total_metric += np.asarray(single_metric)

        if save_result and test_save_path is not None:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                     f'{test_save_path}/{ith:02d}_pred.nii.gz')
            nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)),
                     f'{test_save_path}/{ith:02d}_img.nii.gz')
            nib.save(nib.Nifti1Image(label_gt.astype(np.float32), np.eye(4)),
                     f'{test_save_path}/{ith:02d}_gt.nii.gz')

    avg_metric = total_metric / len(image_list)
    print(f'Average metric — Dice={avg_metric[0]:.4f}  '
          f'Jc={avg_metric[1]:.4f}  '
          f'HD95={avg_metric[2]:.4f}  '
          f'ASD={avg_metric[3]:.4f}')
    return avg_metric


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # ---- Load model --------------------------------------------------------
    if USE_4HEAD:
        from networks.vnet_sdf import VNet
    else:
        from networks.vnet_base import VNet

    net = VNet(n_channels=1, n_classes=1,
               normalization='batchnorm', has_dropout=False).cuda()
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    print(f'Variant: {args.variant}  |  '
          f'4-head={USE_4HEAD}  TTA={USE_TTA}  ensemble={USE_ENSEMBLE}')
    print(f'Loaded checkpoint: {args.model_path}')

    # ---- Test image list ---------------------------------------------------
    with open(args.test_list, 'r') as f:
        image_list = [args.root_path + line.strip() + '/mri_norm2.h5'
                      for line in f if line.strip()]

    # ---- Output directory --------------------------------------------------
    save_path = args.test_save_path
    if args.save_result and save_path is None:
        ckpt_dir  = os.path.dirname(args.model_path)
        save_path = os.path.join(ckpt_dir, f'predictions_{args.variant}')
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # ---- Run evaluation ----------------------------------------------------
    avg = test_all_case(
        net,
        image_list,
        num_classes=1,
        patch_size=tuple(args.patch_size),
        stride_xy=args.stride_xy,
        stride_z=args.stride_z,
        use_4head=USE_4HEAD,
        save_result=bool(args.save_result),
        test_save_path=save_path,
        metric_detail=True,
        nms=bool(args.nms),
    )

    print(f'\n[{args.variant}] Final  '
          f'Dice={avg[0]:.4f}  Jc={avg[1]:.4f}  '
          f'HD95={avg[2]:.4f}  ASD={avg[3]:.4f}')
