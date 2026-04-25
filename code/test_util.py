import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80),
                  stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None,
                  metric_detail=0, nms=0):
    total_metric = 0.0
    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label_gt = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(
            net, image, stride_xy, stride_z, patch_size,
            num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label_gt[:])

        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
                ith, single_metric[0], single_metric[1],
                single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label_gt[:].astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0

    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image,
                       [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                       mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0],
                                   ys:ys + patch_size[1],
                                   zs:zs + patch_size[2]]
                test_patch = np.expand_dims(
                    np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                # Test-time augmentation: average predictions over 4 flip
                # combinations across the W (axis 2) and H (axis 3) axes only.
                #
                # IMPORTANT: D-axis (axis 4) flips are intentionally excluded.
                # RandomRotFlip during training only flips W (axis 0) or H
                # (axis 1) of the 3D image — it never flips D. Including D-axis
                # flips at test time creates out-of-distribution inputs that the
                # network has never seen, producing garbage predictions that
                # drag the ensemble average below the single-pass baseline.
                # TTA augmentations must be a subset of training augmentations.
                y_tta = torch.zeros(
                    1, num_classes, *test_patch.shape[2:],
                    device=test_patch.device)
                flip_axes_list = [
                    [],        # original
                    [2],       # flip W  — seen during training
                    [3],       # flip H  — seen during training
                    [2, 3],    # flip W+H — seen during training
                ]
                with torch.no_grad():
                    for flip_axes in flip_axes_list:
                        patch_aug = torch.flip(test_patch, flip_axes) \
                            if flip_axes else test_patch
                        y1_tanh, y1, _, _ = net(patch_aug)
                        y_seg = torch.sigmoid(y1)
                        y_sdf = torch.sigmoid(-1500 * y1_tanh)
                        y_aug = 0.5 * y_seg + 0.5 * y_sdf
                        if flip_axes:
                            y_aug = torch.flip(y_aug, flip_axes)
                        y_tta += y_aug
                    y_tta /= len(flip_axes_list)

                y = y_tta.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0],
                             ys:ys + patch_size[1],
                             zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0],
                    ys:ys + patch_size[1],
                    zs:zs + patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int64)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                              hl_pad:hl_pad + h,
                              dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w,
                                 hl_pad:hl_pad + h,
                                 dl_pad:dl_pad + d]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i).astype(np.float64)
        label_tmp = (label == i).astype(np.float64)
        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
               (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice
    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd
