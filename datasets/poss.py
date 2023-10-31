import os
import os.path as osp
from glob import glob
from pathlib import Path

import numpy as np
import yaml
from munch import Munch
import torch
import MinkowskiEngine as ME

from .custom import CustomDataset
import pdb

class POSSDataset(CustomDataset):

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
        super(POSSDataset, self).__init__(data_path, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                           out_shape, min_coordinate, voxel_size, beam, fov, training, 
                                           use_sparse_aug, positive_num, beam_sampling)
        with open(label_mapping, 'r') as f:
            possyaml = yaml.safe_load(f)
        if self.imageset == 'train':
            split = possyaml['split']['train']
        elif self.imageset == 'val':
            split = possyaml['split']['valid']
        elif self.imageset == 'test':
            split = possyaml['split']['test']
        
        self.learning_map = possyaml['learning_map_common']
        # self.learning_map_inv = possyaml['learning_map_inv']
        # ignore -100
        # for k, v in self.learning_map.items():
        #     if v == 0: self.learning_map[k] = -100

        self.im_idx = []
        for i_folder in split:
            self.im_idx += self.absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))
        
    def __len__(self):
        return len(self.im_idx)
       
    def __getitem__(self, index):        
        scan_id = self.im_idx[index]
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data = (raw_data[:, :3], raw_data[:, 3][:,None], annotated_data.astype(np.int32))

        return self.getitem(index, scan_id, data)

if __name__ == '__main__':
    cfg_txt = open('configs/config.yaml', 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    dataset = POSSDataset(**cfg.dataset_SemPOSS, training=False, 
                           positive_num=cfg.generalization_params.positive_num,
                           beam_sampling=cfg.generalization_params.beam_sampling,
                           )
    ### visualize
    while True:
        (scan_id, coord, feat, semantic_label, idx) = dataset.__getitem__(0)
        # semantic_label[semantic_label==-100] = 0
        pdb.set_trace()
        from utils.utils import get_semcolor_common
        from utils.ply_vis import write_ply
        
        if not dataset.training:
            vis_color = get_semcolor_common(semantic_label)
            vis_xyz = coord.float().cpu().numpy()
            os.makedirs('sample', exist_ok=True)
            write_ply(f'sample/poss-val-idx={idx}.ply', [vis_xyz, vis_color], ['x','y','z','red','green','blue'])
        
