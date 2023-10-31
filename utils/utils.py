import numpy as np
import torch
import numba as nb
from torch.utils import data
from scipy import stats as s
import torch.distributed as dist
import os
import pickle
import pdb
import cv2
import yaml

from sklearn.neighbors import NearestNeighbors

def get_common_color_map(label_mapping, return_torch_tenor=False, return_numpy_ndarray=False):
    with open(label_mapping, 'r') as stream:
        commonyaml = yaml.safe_load(stream)
    SemKITTI_color_map = dict()
    for i in sorted(list(commonyaml['color_map'].keys()))[::-1]:
        SemKITTI_color_map[i] = commonyaml['color_map'][i][::-1] # bgr -> rgb

    if return_torch_tenor:
        SemKITTI_color_map_ten = torch.zeros(len(SemKITTI_color_map), 3)
        for i in sorted(list(SemKITTI_color_map.keys())):
            SemKITTI_color_map_ten[i] = torch.tensor(SemKITTI_color_map[i])
        return SemKITTI_color_map_ten
    
    if return_numpy_ndarray:
        SemKITTI_color_map_np = np.zeros((len(SemKITTI_color_map), 3))
        for i in sorted(list(SemKITTI_color_map.keys())):
            SemKITTI_color_map_np[i] = np.array(SemKITTI_color_map[i])
        return SemKITTI_color_map_np

    return SemKITTI_color_map

def get_SemKITTI_color_map(label_mapping, return_torch_tenor=False, return_numpy_ndarray=False):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_color_map = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_color_map[semkittiyaml['learning_map'][i]] = semkittiyaml['color_map'][i][::-1] # bgr -> rgb

    if return_torch_tenor:
        SemKITTI_color_map_ten = torch.zeros(len(SemKITTI_color_map), 3)
        for i in sorted(list(SemKITTI_color_map.keys())):
            SemKITTI_color_map_ten[i] = torch.tensor(SemKITTI_color_map[i])
        return SemKITTI_color_map_ten
    
    if return_numpy_ndarray:
        SemKITTI_color_map_np = np.zeros((len(SemKITTI_color_map), 3))
        for i in sorted(list(SemKITTI_color_map.keys())):
            SemKITTI_color_map_np[i] = np.array(SemKITTI_color_map[i])
        return SemKITTI_color_map_np

    return SemKITTI_color_map

COLOR_MAP_KITTI = get_SemKITTI_color_map('configs/label_mapping/semantic-kitti.yaml', return_numpy_ndarray=True)
COLOR_MAP = get_common_color_map('configs/label_mapping/common_map.yaml', return_numpy_ndarray=True)



def get_semcolor_kitti(pt_label):
    if pt_label.dim() == 2:
        pt_label = pt_label.squeeze(1)
        
    pt_color = np.take(COLOR_MAP_KITTI, pt_label, axis=0).astype(np.uint8)
    return pt_color

def get_semcolor_common(pt_label):
    if pt_label.ndim == 2:
        pt_label = pt_label.squeeze(1)
        
    pt_color = np.take(COLOR_MAP, pt_label, axis=0).astype(np.uint8)
    return pt_color
    
def merge_configs(args, new_configs):
    
    if hasattr(args, 'expname'):
        new_configs['exp_params']['exp_name'] = args.expname

    if hasattr(args, 'model'):
        new_configs['model']['name'] = args.model

    if hasattr(args, 'ckpt_path'):
        new_configs['EXP_PARAMS']['CKPT_PATH'] = args.ckpt_path

    
    return new_configs


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

def rotate_pts(xyz, rotate):
    rotate_rad = np.deg2rad(rotate)
    c, s = np.cos(rotate_rad), np.sin(rotate_rad)
    j = np.matrix([[c, s], [-s, c]])
    if isinstance(xyz, list):
        for xyz_ in xyz:
            xyz_[:,:2] = np.dot(xyz_[:,:2],j)
    else:
        xyz[:,:2] = np.dot(xyz[:,:2],j)
        
    return xyz