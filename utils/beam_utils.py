import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
import cv2
from torch.nn import functional as F
import torch
import pdb
import random

def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)
    theta = (theta / np.pi) * 180

    # sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    # phi_ = np.arcsin(sin_phi)
    # phi_ = (phi_ / np.pi) * 180

    # cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    # phi = np.arccos(cos_phi)
    # phi = (phi / np.pi) * 180

    # phi[phi_ < 0] = 360 - phi[phi_ < 0]
    # phi[phi == 360] = 0

    return theta
    # return theta, phi

def beam_label(theta, beam):
    estimator=KMeans(n_clusters=beam)
    # pdb.set_trace()
    res=estimator.fit_predict(theta.reshape(-1, 1))
    label=estimator.labels_
    centroids=estimator.cluster_centers_
    return label, centroids[:,0]


def get_downsampled_mask(beam_label):
    
    # choose final number of beams
    final_beam_num = random.randint(32, 64)
    # randomly choose beams to exclude
    exclude_beam_list = random.sample(range(0, 64), 64-final_beam_num)
    mask = np.ones((beam_label.shape[0])).astype(np.bool) 
    # pdb.set_trace()
    for exclude_beam in exclude_beam_list:
        mask_ = beam_label == exclude_beam
        mask[mask_] = False

    return mask


def exclude_from_range(pts, fov_info, proj_H=64, proj_W=2048):
    """ Project a pointcloud into a spherical projection image.projection"""
    # laser parameters
    fov_down, fov_up = fov_info
    
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(pts, 2, axis=1)

    # get scan components
    scan_x = pts[:, 0]
    scan_y = pts[:, 1]
    scan_z = pts[:, 2]

    # get angles of all points
    # yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    # proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    # proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]

    # round and clamp for use as index
    # proj_x = np.floor(proj_x)
    # proj_x = np.minimum(proj_W - 1, proj_x)
    # proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    # proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    proj_y = np.copy(proj_y)  # stope a copy in original order
    # choose final number of beams
    final_beam_num = random.randint(32, 64)
    # randomly choose beams to exclude
    exclude_beam_list = random.sample(range(0, 64), 64-final_beam_num)
    mask = np.ones((proj_y.shape[0])).astype(np.bool) 
    # pdb.set_trace()
    for exclude_beam in exclude_beam_list:
        mask_ = proj_y == exclude_beam
        mask[mask_] = False
        
    return mask