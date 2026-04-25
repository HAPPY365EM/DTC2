"""
test_ablation.py — Unified test script for ablation variants M0 – M5.

Variant differences at inference
----------------------------------
M0          2-head VNet | single forward pass | seg head only
M1 – M4     4-head VNet | single forward pass | seg head only
M5          4-head VNet | 4-fold TTA          | dual-head ensemble
              (uses M4 checkpoint; only inference differs)

Usage examples
--------------
# Test M0
python test_ablation.py --variant M0 --model_path ../model/.../iter_6000.pth

# Test M5 (using M4 checkpoint)
python test_ablation.py --variant M5 --model_path ../model/.../iter_6000.pth
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
# Variant flags (mirrors train_ablation.py)
# =============================================================================

VARIANT_FLAGS = {
    'M0': dict(use_4head=False, use_tta=False, use_ensemble=False),
    'M1': dict(use_4head=True,  use_tta=False, use_ensemble=False),
    'M2': dict(use_4head=True,  use_tta=False, use_ensemble=False),
    'M3': dict(use_4head=True,  use_tta=False, use_ensemble=False),
    'M4': dict(use_4head=True,  use_tta=False, use_ensemble=False),
    # M5 = M4 weights + TTA + dual-head ensemble at inference
    'M5': dict(use_4head=True,  use_tta=True,  use_ensemble=True),
}

# TTA flip combinations — W and H axes only (D excluded; see test_util.py for rationale)
TTA_FLIP_AXES = [
    [],       # original
    [2],      # flip W
    [3],      # flip H
    [2, 3],   # flip W + H
]

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=str, default='M5',
                    choices=list(VARIANT_FLAGS.keys()))
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

def _forward(net, patch, use_4head, use_ensemble):
    """
    Single forward pass on a 5D patch (1,1,H,W,D).
    Returns a probability map tensor of shape (1, H, W, D).
    """
    if use_4head:
        y_tanh, y_seg, _, _ = net(patch)
    else:
        y_tanh, y_seg = net(patch)

    prob_seg = torch.sigmoid(y_seg)       # Task 1 probability

    if use_ensemble:
        # Dual-head ensemble: average seg and SDF-derived probabilities.
        # The SDF output is negated before sigmoid so that interior voxels
        # (negative SDF) map to high probability, matching the seg head.
        prob_sdf = torch.sigmoid(-1500 * y_tanh)   # Task 2 probability
        prob = 0.5 * prob_seg + 0.5 * prob_sdf
    else:
        prob = prob_seg

    # Remove the batch dimension: (1, C, H, W, D) → (C, H, W, D)
    # so that prob_final aligns with score_map shape (num_classes, H, W, D).
    return prob[0]


def test_single_case(net, image, stride_xy, stride_z, patch_size,
                     num_classes=1, use_4head=False,
                     use_tta=False, use_ensemble=False):
    """
    Sliding-window inference over one volume.

    Args:
        net         : trained model in eval mode.
        image       : numpy array (W, H, D).
        stride_xy   : sliding-window stride in W/H dimensions.
        stride_z    : sliding-window stride in D dimension.
        patch_size  : (W, H, D) tuple.
        num_classes : number of foreground classes (1 for binary).
        use_4head   : True for 4-output network (M1-M5), False for M0.
        use_tta     : enable 4-fold flip TTA (M5 only).
        use_ensemble: blend seg + SDF-derived probabilities (M5 only).

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
                    if use_tta:
                        # 4-fold TTA: average over flip augmentations.
                        # acc shape: (num_classes, H, W, D) — batch dim already
                        # removed by _forward, so we allocate without it.
                        acc = torch.zeros(
                            num_classes, *patch_t.shape[2:],
                            device=patch_t.device)
                        for flip_axes in TTA_FLIP_AXES:
                            aug = (torch.flip(patch_t, flip_axes)
                                   if flip_axes else patch_t)
                            prob = _forward(net, aug, use_4head, use_ensemble)
                            if flip_axes:
                                prob = torch.flip(prob, flip_axes)
                            acc += prob
                        prob_final = (acc / len(TTA_FLIP_AXES)).cpu().numpy()
                    else:
                        # Single forward pass (M0 – M4)
                        prob_final = _forward(
                            net, patch_t, use_4head,
                            use_ensemble).cpu().numpy()

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
                  use_4head, use_tta, use_ensemble,
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
            num_classes=num_classes,
            use_4head=use_4head,
            use_tta=use_tta,
            use_ensemble=use_ensemble)

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

    FLAGS      = VARIANT_FLAGS[args.variant]
    use_4head   = FLAGS['use_4head']
    use_tta     = FLAGS['use_tta']
    use_ensemble = FLAGS['use_ensemble']

    # ---- Load model --------------------------------------------------------
    if use_4head:
        from networks.vnet_sdf import VNet
    else:
        from networks.vnet_base import VNet

    net = VNet(n_channels=1, n_classes=1,
               normalization='batchnorm', has_dropout=False).cuda()
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    print(f'Variant: {args.variant}  |  '
          f'4-head={use_4head}  TTA={use_tta}  ensemble={use_ensemble}')
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
        use_4head=use_4head,
        use_tta=use_tta,
        use_ensemble=use_ensemble,
        save_result=bool(args.save_result),
        test_save_path=save_path,
        metric_detail=True,
        nms=bool(args.nms),
    )

    print(f'\n[{args.variant}] Final  '
          f'Dice={avg[0]:.4f}  Jc={avg[1]:.4f}  '
          f'HD95={avg[2]:.4f}  ASD={avg[3]:.4f}')
