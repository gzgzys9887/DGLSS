import os
import numpy as np
import yaml
from munch import Munch
import MinkowskiEngine as ME

from .custom import CustomDataset
import pdb


class KITTIDataset(CustomDataset):

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
        super(KITTIDataset, self).__init__(data_path, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                           out_shape, min_coordinate, voxel_size, beam, fov, training, 
                                           use_sparse_aug, positive_num, beam_sampling)
        with open(label_mapping, 'r') as f:
            semkittiyaml = yaml.safe_load(f)
        if self.imageset == 'train':
            split = semkittiyaml['split']['train']
        elif self.imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif self.imageset == 'test':
            split = semkittiyaml['split']['test']
            
        self.learning_map = semkittiyaml['learning_map_common']
        self.learning_map_inv = semkittiyaml['learning_map_inv']
        
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
    dataset = KITTIDataset(**cfg.dataset_SemKITTI, training=True, 
                           positive_num=cfg.generalization_params.positive_num,
                           beam_sampling=cfg.generalization_params.beam_sampling)

    ### visualize
    while True:
        (scan_id, coord, feat, semantic_label, idx) = dataset.__getitem__(0)
        pdb.set_trace()
        from utils.utils import get_semcolor_common
        from utils.ply_vis import write_ply
 
        for i in range(len(coord)):
            vis_color = get_semcolor_common(semantic_label[i])
            vis_xyz = coord[i].float().cpu().numpy()
            os.makedirs('sample', exist_ok=True)
            pdb.set_trace()
            write_ply(f'sample/kitti_2.ply', [vis_xyz, vis_color], ['x','y','z','red','green','blue'])
