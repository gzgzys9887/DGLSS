import os
import numpy as np
from torch.utils import data
import yaml
import pickle
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from utils import beam_utils

from datasets.augmentor import Augmentor

import torch
import math
import pdb

class CustomDataset(Dataset):
    
    # 10 classes
    CLASSES = ('car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'pedestrian', 'drivable-surface',
             'sidewalk', 'terrain', 'vegetation')

    def __init__(self,
                 data_path,
                 ignore_label=-100,
                 label_mapping=None,
                 max_volume_space=[50., 50., 2.],
                 min_volume_space=[-50., -50., -4.],
                 out_shape=[512,512,32],
                 min_coordinate=[-256,-256,-21],
                 voxel_size=0.2,
                 beam=64,
                 fov=[-23.6, 3.2],
                 training=False,
                 use_sparse_aug=False,
                 positive_num=1,
                 beam_sampling=[0.3, 0.7],
                 ):
        self.data_root = data_path
        self.training = training
        self.imageset = 'train' if training else 'val'
        self.ignore_label = ignore_label
        self.label_mapping = label_mapping
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.out_shape = out_shape
        self.min_coordinate = min_coordinate
        self.voxel_size = voxel_size
        self.beam = beam
        self.fov_info = fov
        
        self.use_sparse_aug = use_sparse_aug
        self.positive_num = positive_num
        self.beam_sampling = beam_sampling
        
        self.prefix = None
        self.suffix = None
        
        self.augmentor = Augmentor(beam, fov, beam_sampling)

    def getitem(self, index, scan_id, data):
        assert len(data) == 3, "data tuple should be (xyz, ref, label)"

        if self.training:
            if self.use_sparse_aug:
                data = self.make_positive_samples(*data)
            data = self.transform_train(*data)
        else:
            data = self.transform_test(*data)
        
        if data is None:
            return None
        
        xyz, ref, semantic_label = data
        xyz = self.clip_data(xyz)
        
        if not self.training:
            coord = torch.from_numpy(xyz)
            feat = torch.ones((coord.shape[0], 1)).float()
            semantic_label = torch.from_numpy(semantic_label).int()

            return (scan_id, coord, feat, semantic_label, torch.tensor(index))

        if not isinstance(xyz, list):
            xyz = [xyz]
            ref = [ref]
            semantic_label = [semantic_label]
        
        coords, feats, semantic_labels = [], [], []
        for i in range(len(xyz)):
            coord = torch.from_numpy(xyz[i])
            
            feat = torch.ones((coord.shape[0], 1)).float()
            label = torch.from_numpy(semantic_label[i]).int()
            quantized_coord, quantized_feat, quantized_label, voxel_idx = ME.utils.sparse_quantize(coord,
                                                                                                    feat,
                                                                                                    labels=label,
                                                                                                    ignore_label=self.ignore_label,
                                                                                                    quantization_size=self.voxel_size,
                                                                                                    return_index=True)
            # return unique coords, feats, and sem_lables. index refer to the index of corresponding unique coords, feats, and sem_labels (the first appears one).
            # if the voxel contains more than one labels, assign the ignore_label.
            coords.append(quantized_coord)
            feats.append(quantized_feat)
            semantic_labels.append(quantized_label)
                          
        return (scan_id, coords, feats, semantic_labels, torch.tensor(index))

    def data_augment(self, xyz, flip=False, rot=False, scale_aug=False, noise_transform=False):

        # random data augmentation by rotation
        if rot:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            if isinstance(xyz, list):
                for xyz_ in xyz:
                    xyz_[:,:2] = np.dot(xyz_[:,:2],j)
            else:
                xyz[:,:2] = np.dot(xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if flip:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                if isinstance(xyz, list):
                    for xyz_ in xyz:
                        xyz_[:,0] = -xyz_[:,0]    
                else:
                    xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                if isinstance(xyz, list):
                    for xyz_ in xyz:
                        xyz_[:,1] = -xyz_[:,1]    
                else:
                    xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                if isinstance(xyz, list):
                    for xyz_ in xyz:
                        xyz_[:,:2] = -xyz_[:,:2]
                else:
                    xyz[:,:2] = -xyz[:,:2]

        if scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            if isinstance(xyz, list):
                    for xyz_ in xyz:
                        xyz_[:,0] = noise_scale * xyz_[:,0]
                        xyz_[:,1] = noise_scale * xyz_[:,1]
                        xyz_[:,2] = noise_scale * xyz_[:,2]        
            else:
                xyz[:,0] = noise_scale * xyz[:,0]
                xyz[:,1] = noise_scale * xyz[:,1]
                xyz[:,2] = noise_scale * xyz[:,2]

        if noise_transform:
            noise_translate = np.array([np.random.normal(0, 0.1, 1),
                                np.random.normal(0, 0.1, 1),
                                np.random.normal(0, 0.1, 1)]).T
            if isinstance(xyz, list):
                for xyz_ in xyz:
                    xyz_[:, 0:3] += noise_translate
            else:
                xyz[:, 0:3] += noise_translate
            
        return xyz
    
    def absoluteFilePaths(self, directory):
        for dirpath, _, filenames in os.walk(directory):
            filenames.sort()
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
    
    def clip_data(self, xyz):
        max_bound = np.asarray(self.max_volume_space)
        min_bound = np.asarray(self.min_volume_space)
        if isinstance(xyz, list):
            return_xyz = []
            for xyz_ in xyz:
                return_xyz.append(np.clip(xyz_, min_bound, max_bound))
        else:
            return_xyz = np.clip(xyz, min_bound, max_bound)
        return return_xyz

    def transform_train(self, xyz, ref, semantic_label):
        xyz = self.data_augment(xyz, True, True, True)
        return xyz, ref, semantic_label

    def transform_test(self, xyz, ref, semantic_label):
        xyz = self.data_augment(xyz, False, False, False)
        return xyz, ref, semantic_label
    
    def get_beam_id(self, xyz):
        beam_id = self.augmentor.get_beam_id(xyz)
        
        return beam_id[:,None]
    
    def make_positive_samples(self, xyz, ref, semantic_label):
        return_xyz, return_ref, return_label = [xyz], [ref], [semantic_label]
        for _ in range(self.positive_num):
            # theta = beam_utils.compute_angles(xyz)
            # beam_label, _ = beam_utils.beam_label(theta, self.beam)
            mask = self.augmentor.excludeAugment(pts=xyz)
            
            return_xyz.append(xyz[mask])
            return_ref.append(ref[mask])
            return_label.append(semantic_label[mask])
        
        return return_xyz, return_ref, return_label

    def collate_fn_test(self, batch):
        scan_ids = []
        list_coord = []
        list_feat = []
        list_label = []
        list_idx = []

        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, feat, semantic_label, idx) = data
            scan_ids.append(scan_id)
            list_coord.append(coord)
            list_feat.append(feat)
            list_label.append(semantic_label)
            list_idx.append(idx.view(-1,1))
        
            batch_id += 1
            
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            print(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        idx = torch.cat(list_idx, dim=0)
    
        return {
            'scan_ids': scan_ids,
            'batch_idxs': idx,
            'coordinates': list_coord,
            'features': list_feat,
            'labels': list_label,
            'batch_size': batch_id,
        }
        
    def collate_fn(self, batch):
        
        scan_ids = []
        list_d = []
        list_d_aug = []
        list_idx = []

        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, feat, semantic_label, idx) = data
            scan_ids.append(scan_id)
            list_d.append((coord[0], feat[0], semantic_label[0]))
            
            if self.use_sparse_aug:
                for aug_idx in range(1, self.positive_num+1):
                    list_d_aug.append((coord[aug_idx], feat[aug_idx], semantic_label[aug_idx]))
            
            list_idx.append(idx.view(-1,1))
            batch_id += 1
            
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            print(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(dtype=torch.float32)(list_d)
        idx = torch.cat(list_idx, dim=0)

        return_dict = {'scan_ids': scan_ids,
                       'batch_idxs': idx,
                       'coordinates': coordinates_batch,
                       'features': features_batch,
                       'labels': labels_batch,
                       'batch_size': batch_id
                       }
        
        if self.use_sparse_aug:
            coordinates_aug_batch, features_aug_batch, labels_aug_batch = ME.utils.SparseCollation(dtype=torch.float32)(list_d_aug)
            return_dict.update({'coordinates_aug': coordinates_aug_batch,
                                'features_aug': features_aug_batch,
                                'labels_aug': labels_aug_batch
                                })
        return return_dict
        