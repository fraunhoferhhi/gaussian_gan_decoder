# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""
import os

import numpy as np
import scipy.linalg
from . import metric_utils

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import wandb


# ----------------------------------------------------------------------------


def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    detector_kwargs = dict(return_features=True)  # Return raw features before the softmax layer.
    folder = f"./metric_results/"
    os.makedirs(folder, exist_ok=True)
    wandb.init(project="GaussianHeadDecoder", dir=folder, group="metrics", name=opts.dataset_kwargs['camera_sample_mode'])

    if opts.use_decoder:
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_decoder(
            opts=opts,
            detector_url=detector_url,
            detector_kwargs=detector_kwargs,
            rel_lo=0,
            rel_hi=1,
            capture_mean_cov=True,
            max_items=num_gen,
        ).get_mean_cov()
    else:
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts,
            detector_url=detector_url,
            detector_kwargs=detector_kwargs,
            rel_lo=0,
            rel_hi=1,
            capture_mean_cov=True,
            max_items=num_gen,
        ).get_mean_cov()


    # max_real, num_gen = 1000, 1000
    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=0,
        capture_mean_cov=True,
        max_items=max_real,
    ).get_mean_cov()


    fig_np = calculate_fid(
        mu_real, mu_gen, sigma_real, sigma_gen
    )
    if opts.rank != 0:
        return float("nan")

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    fid = float(fid)
    print(fid)
    wandb.log({f"fid: {fid}"}, step=0)
    return float(fid)


# ----------------------------------------------------------------------------


def calculate_fid(mu1, mu2, sigma1, sigma2):
    # calculate mean and covariance statistics
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
