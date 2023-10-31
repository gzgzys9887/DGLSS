import os
import os.path as osp
from glob import glob
from pathlib import Path

import numpy as np
import yaml
from munch import Munch
import pickle
import torch
import MinkowskiEngine as ME
from nuscenes import NuScenes

from .custom import CustomDataset
from tqdm import tqdm

import pdb

class nuScenesDataset(CustomDataset):

    def __init__(self,
                 data_path,
                 ignore_label=-100,
                 label_mapping=None,
                 max_volume_space=[50., 50., 3.],
                 min_volume_space=[-50., -50., -5.],
                 out_shape=[512,512,48],
                 min_coordinate=[-256,-256,-29],
                 voxel_size=0.2,
                 beam=32,
                 fov=[-30.0, 10.0],
                 training=False,
                 use_sparse_aug=False,
                 positive_num=1,
                 beam_sampling=[0.3, 0.7],
                 ):
        super(nuScenesDataset, self).__init__(data_path, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                           out_shape, min_coordinate, voxel_size, beam, fov, training, 
                                           use_sparse_aug, positive_num, beam_sampling)       
        with open(osp.join(label_mapping), 'r') as f:
            nuscenes = yaml.safe_load(f)
            
        with open(osp.join(data_path, f'nuscenes_infos_{self.imageset}.pkl'), 'rb') as pk:
            data = pickle.load(pk)

        self.learning_map = nuscenes['learning_map_common']
        
        self.im_idx = data['infos'] # dict_keys(['lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts'])
        
        self.ignore_label = ignore_label
        
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

        # ignore -100
        # for k, v in self.learning_map.items():
        #     if v == 0: self.learning_map[k] = -100
            
    def __len__(self):
        return len(self.im_idx)
    
    def __getitem__(self, index):
        info = self.im_idx[index]
        lidar_path = info['lidar_path'][16:] # ex: 'samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin'
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_sd_token)['filename'])

        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        points = np.fromfile(os.path.join(self.nusc.dataroot, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        data = (points[:,:3], points[:,3], points_label)

        # RV_image_32 = self.augmentor.makeRV(points[:,:3], points_label, 32, 1024)
        # RV_image_64_label = self.augmentor.makeRV_label(points[:,:3], points_label, 64, 1024)
        # return lidar_path, RV_image_32, RV_image_64_label, data
        
        return self.getitem(index, lidar_path, data)
    
    
    def make_positive_samples(self, xyz, ref, semantic_label):
        return_xyz, return_ref, return_label = [xyz], [ref], [semantic_label]
        for _ in range(self.positive_num):
            # theta = beam_utils.compute_angles(xyz)
            # beam_label, _ = beam_utils.beam_label(theta, self.beam)
            
            mask = self.augmentor.excludeAugment(pts=xyz, proj_H=32, proj_W=1024)
            return_xyz.append(xyz[mask])
            return_ref.append(ref[mask])
            return_label.append(semantic_label[mask])
        
        return return_xyz, return_ref, return_label

    # same collate function as CustomDataset

def generate_laser_directions(lidar):
    """
    Generate the laser directions using the LiDAR specification.

    :param lidar: LiDAR specification
    :return: a set of the query laser directions;
    """
    v_dir = np.linspace(start=lidar['min_v'], stop=lidar['max_v'], num=lidar['channels'])
    h_dir = np.linspace(start=lidar['min_h'], stop=lidar['max_h'], num=lidar['points_per_ring'], endpoint=False)

    v_angles = []
    h_angles = []

    for i in range(lidar['channels']):
        v_angles = np.append(v_angles, np.ones(lidar['points_per_ring']) * v_dir[i])
        h_angles = np.append(h_angles, h_dir)

    return np.stack((v_angles, h_angles), axis=-1).astype(np.float32)

def range_image_to_points_ours(range_image, lidar, remove_zero_range=True):
    """
    Convert a range image to the points in the sensor coordinate.

    :param range_image: denormalized range image
    :param lidar: LiDAR specification
    :param remove_zero_range: flag to remove the points with zero ranges
    :return: points in sensor coordinate
    """
    max_v = lidar['max_v']
    min_v = lidar['min_v']
    lidar['min_v'] = -max_v
    lidar['max_v'] = -min_v
    angles = generate_laser_directions(lidar)
    angles[:,0] = -angles[:,0]
    
    r = range_image.flatten()

    x = np.sin(angles[:, 1]) * np.cos(angles[:, 0]) * r
    y = np.cos(angles[:, 1]) * np.cos(angles[:, 0]) * r
    z = np.sin(angles[:, 0]) * r

    points = np.stack((x, y, z), axis=-1)  # sensor coordinate

    # Remove the points having invalid detection distances
    if remove_zero_range is True:
        points = np.delete(points, np.where(r < 1e-5), axis=0)

    return points

from utils.ply_vis import write_ply

if __name__ == '__main__':
    cfg_txt = open('configs/config_kitti.yaml', 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    dataset = nuScenesDataset(**cfg.dataset_nuScenes, training=True,
                                positive_num=cfg.generalization_params.positive_num,
                                beam_sampling=cfg.generalization_params.beam_sampling,
                            )
    
    ### make RV
    num_train = len(dataset)
    assert num_train == 28130
    
    from utils.utils import get_semcolor_common
    from utils.ply_vis import write_ply
    
    for i in tqdm(range(num_train)): 
        pdb.set_trace()
        
        scan_id, coord, feat, semantic_label, idx = dataset.__getitem__(i)
        vis_color = get_semcolor_common(semantic_label[i])
        vis_xyz = coord[i].float().cpu().numpy()
        os.makedirs('sample', exist_ok=True)
        write_ply(f'sample/nuscenes_1.ply', [vis_xyz, vis_color], ['x','y','z','red','green','blue'])
        
# python -m datasets.nuScenes