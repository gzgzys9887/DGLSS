import os
import numpy as np
import yaml
from munch import Munch
import MinkowskiEngine as ME
import pdb

from .custom import CustomDataset


class WaymoDataset(CustomDataset):

    def __init__(self,
                 data_path,
                 ignore_label=-100,
                 label_mapping=None,
                 max_volume_space=[50., 50., 3.],
                 min_volume_space=[-50., -50., -5.],
                 out_shape=[768,768,32],
                 min_coordinate=[-384,-384,-11],
                 voxel_size=0.2,
                 beam=64,
                 fov=[-17.6, 2.4],
                 training=False,
                 use_sparse_aug=False,
                 positive_num=1,
                 beam_sampling=[0.3, 0.7],
                 ):
        
        super(WaymoDataset, self).__init__(data_path, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                           out_shape, min_coordinate, voxel_size, beam, fov, training, 
                                           use_sparse_aug, positive_num, beam_sampling)

        with open(label_mapping, 'r') as f:
            waymo = yaml.safe_load(f)
        with open(label_mapping.replace('waymo', 'waymo_split'), 'r') as f:
            waymo_split = yaml.safe_load(f)    
        
        if self.imageset == 'train':
            split = waymo_split['train']
        elif self.imageset == 'val':
            split = waymo_split['valid']
        elif self.imageset == 'test':
            split = waymo_split['test']
        
        self.learning_map = waymo['learning_map_common']
        self.ignore_label = ignore_label

        # # ignore -100
        # for k, v in self.learning_map.items():
        #     if v == 0: self.learning_map[k] = -100
        
        self.im_idx = []
        for i_folder in split:
            self.im_idx += self.absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(4), 'velodyne']))
    
    def __len__(self):
        return len(self.im_idx)
    
    def __getitem__(self, index):
        scan_id = self.im_idx[index]
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1,4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            
        ref = np.tanh(raw_data[:,3])
        data = (raw_data[:, :3], ref, annotated_data.astype(np.int32))

        return self.getitem(index, scan_id, data)
        
    # same collate function as CustomDataset
    
if __name__ == '__main__':
    cfg_txt = open('configs/config.yaml', 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    dataset = WaymoDataset(**cfg.dataset_Waymo, training=True,
                           use_sparse_aug=cfg.generalization_params.use_sparse_aug,
                           positive_num=cfg.generalization_params.positive_num,
                           beam_sampling=cfg.generalization_params.beam_sampling,
                           )
    
    # save training set
    from utils.utils import get_semcolor_common
    from utils.ply_vis import write_ply
    from tqdm import tqdm
    
    ### visualize
    while True:
        (scan_id, coord, feat, semantic_label, idx) = dataset.__getitem__(0)
        pdb.set_trace()
        
        name_list=['orig', 'aug1', 'aug2', 'aug3', 'aug4']
        for i in range(len(coord)):
            vis_color = get_semcolor_common(semantic_label[i])
            vis_xyz = coord[i].float().cpu().numpy()
            os.makedirs('sample', exist_ok=True)
            write_ply(f'sample/waymo_sample_{name_list[i]}_gg.ply', [vis_xyz, vis_color], ['x','y','z','red','green','blue'])
    
# python -m datasets.waymo