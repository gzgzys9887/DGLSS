import os
import time
import argparse
import numpy as np
import random
import yaml
from easydict import EasyDict
from munch import Munch

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils.utils as utils
import models as models
from datasets.initialization import get_dataset, get_dataset_single
import pipeline.exp as exp
import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        default="configs/train_kitti.yaml",
                        # default="configs/train_waymo.yaml",
                        type=str,
                        help="Path to config file")
    parser.add_argument('-ld', '--logdir', default='Test')
    parser.add_argument('-en', '--expname', default='ExpSemantic')
    parser.add_argument('-db', '--debug', action='store_true')
    args = parser.parse_args()
    return args

        
if __name__ == '__main__':
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    
    cfg.exp_name = args.expname
    cfg.log_dir = args.logdir
    
    # fix random seed
    seed = 3407
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    
    nuscenes_dataset = get_dataset_single('nuscenes', training=True, positive_num=1, cfg=cfg)
    waymo_dataset = get_dataset_single('waymo', training=True, positive_num=1, cfg=cfg)
    
    loader_param = cfg.val_dataloader
    loader_param.pop('training')
    nusc_dataloader = DataLoader(nuscenes_dataset, **loader_param, collate_fn=nuscenes_dataset.collate_fn)
    waymo_dataloader = DataLoader(waymo_dataset, **loader_param, collate_fn=waymo_dataset.collate_fn)
    
    NUSC_CLASS = [0.]*11
    WAYMO_CLASS = [0.]*11
    
    for iter, batch in enumerate(tqdm(nusc_dataloader)):
        labels = batch['labels']
        assert len(labels) == len(batch['coordinates'])
        
        if iter % 500 == 0: print(f'Finished Nuscenes {iter}')
        
        unq_labels = np.unique(labels)
        
        for unq in unq_labels:
            pt_num = (labels == unq).sum().item()
            NUSC_CLASS[unq] += pt_num
    
    print('======NuScenes=====')
    whole_num = np.sum(NUSC_CLASS)
    for idx, num in enumerate(NUSC_CLASS):
        print(f'{idx}: {1/(num/whole_num+1e-3)}')
    
    for iter, batch in enumerate(tqdm(waymo_dataloader)):
        labels = batch['labels']
        assert len(labels) == len(batch['coordinates'])
        
        if iter % 500 == 0: print(f'Finished Waymo {iter}')
        
        unq_labels = np.unique(labels)
        
        for unq in unq_labels:
            pt_num = (labels == unq).sum().item()
            WAYMO_CLASS[unq] += pt_num
    
    
    print('\n======Waymo=====')
    whole_num = np.sum(WAYMO_CLASS)
    for idx, num in enumerate(WAYMO_CLASS):
        print(f'{idx}: {1/(num/whole_num+1e-3)}')
        
    pdb.set_trace()
    print('finish')