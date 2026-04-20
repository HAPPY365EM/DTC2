# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler

# FIX: removed `import networks` which was copied from an unrelated codebase
# and referred to a `networks` package and a `models` variable that do not
# exist in this project. Because Python executes all top-level imports when a
# module is first loaded, this caused an ImportError whenever train_la_dtc.py
# ran `from utils.util import compute_sdf`, even though load_model() (the only
# function that referenced `networks`) was never called.
#
# load_model() itself is also unused in this project — it was leftover scaffold
# from the original Facebook Research clustering codebase. It has been removed
# below to avoid confusion. If you need checkpoint loading, use torch.load()
# directly in your training / evaluation scripts.


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def compute_sdf(img_gt, out_shape):
    """
    Compute the signed distance map of a binary segmentation mask.

    Args:
        img_gt  (np.ndarray): integer mask, shape = (batch_size, x, y, z).
        out_shape (tuple):    desired output shape, same as img_gt.shape.

    Returns:
        normalized_sdf (np.ndarray): float64 SDF normalised to [-1, 1],
            shape = out_shape.
            sdf(x) = 0   on the segmentation boundary
                   < 0   inside  the segmentation (normalised negative distance)
                   > 0   outside the segmentation (normalised positive distance)
    """
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(
                posmask, mode='inner').astype(np.uint8)
            sdf = ((negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis))
                   - (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis)))
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf

    return normalized_sdf
