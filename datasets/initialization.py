import os
import numpy as np
from datasets.custom import CustomDataset
from datasets.kitti import KITTIDataset
from datasets.nuScenes import nuScenesDataset
from datasets.waymo import WaymoDataset
from datasets.poss import POSSDataset

import pdb

def get_dataset_single(dataset_name,
                       training,
                       cfg,
                       generalization_param):
    assert dataset_name in ['kitti', 'nuscenes', 'waymo', 'poss'], 'unexpected dataset_name is given.'
    
    if dataset_name == 'kitti':
        dataset = KITTIDataset(**cfg.dataset_SemKITTI, **generalization_param, training=training)
    
    elif dataset_name == 'nuscenes':
        dataset = nuScenesDataset(**cfg.dataset_nuScenes, **generalization_param, training=training)
    
    elif dataset_name == 'waymo':
        dataset = WaymoDataset(**cfg.dataset_Waymo, **generalization_param, training=training)
    
    elif dataset_name == 'poss':
        dataset = POSSDataset(**cfg.dataset_SemPOSS, **generalization_param, training=training)
    
    else:
        raise NotImplementedError
        
    return dataset

def get_dataset(cfg, debug):
    generalization_param = cfg.generalization_params.copy()
    source = generalization_param.pop('source')
    target = generalization_param.pop('target')
    
    assert source in ['kitti', 'waymo']
    
    training_dataset = get_dataset_single(source, training=True, cfg=cfg, generalization_param=generalization_param)
    
    validation_dataset = []
    for target_name in target:
        validation_dataset.append(get_dataset_single(target_name, training=False, cfg=cfg, generalization_param=generalization_param))
        
    if debug:
        training_dataset.im_idx = training_dataset.im_idx[:64]
        for dataset in validation_dataset:
            dataset.im_idx = dataset.im_idx[:64]
    
    return training_dataset, validation_dataset

def get_test_dataset(cfg, debug):    
    target = cfg.test_params.target
    test_dataset = []
    for target_name in target:
        test_dataset.append(get_dataset_single(target_name, training=False, cfg=cfg, generalization_param={}))
        
    if debug:
        for dataset in test_dataset:
            dataset.im_idx = dataset.im_idx[:64]
    
    return test_dataset