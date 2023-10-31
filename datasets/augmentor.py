from selectors import EpollSelector
from builtins import NotImplementedError
import numpy as np
import random
import torch
import torch.nn as nn
import pdb
import utils.utils as utils

class Augmentor():
    def __init__(self, beam, fov, beam_sampling):
                
        self.beam = beam
        self.fov = fov
        
        self.beam_sampling_ratio = beam_sampling
        
        self.things_id = [1,2,3,4,5,6] # TODO: change hard coding
    
    def get_beam_id(self, pts, proj_H=64, proj_W=2048):
        fov_down, fov_up = self.fov
        
        fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
        
        # get scan components
        scan_x = pts[:, 0]
        scan_y = pts[:, 1]
        scan_z = pts[:, 2]
        
        # get x^2 + y^2
        scan_x2py2 = np.sqrt(scan_x**2 + scan_y**2)
        pitch = np.arctan2(scan_z, scan_x2py2)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]
        proj_y *= proj_H                              # in [0.0, H]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        
        return proj_y
    
    def excludeAugment(self, pts, proj_H=64, proj_W=2048):
        """ Project a pointcloud into a spherical projection image.projection"""
        # laser parameters
        fov_down, fov_up = self.fov
        
        fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
        
        # get scan components
        scan_x = pts[:, 0]
        scan_y = pts[:, 1]
        scan_z = pts[:, 2]
        
        # get x^2 + y^2
        scan_x2py2 = np.sqrt(scan_x**2 + scan_y**2)
        pitch = np.arctan2(scan_z, scan_x2py2)
        
        # get projections in image coords
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_y *= proj_H                              # in [0.0, H]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        proj_y = np.copy(proj_y)  # stope a copy in original order
        
        # choose final number of beams
        min_beam_num = int(self.beam*self.beam_sampling_ratio[0])
        max_beam_num = int(self.beam*self.beam_sampling_ratio[1])

        final_beam_num = random.randint(min_beam_num, max_beam_num)
        
        # randomly choose beams to remain
        remain_beam_list = random.sample(range(0, self.beam), final_beam_num)
        mask = np.zeros((proj_y.shape[0])).astype(np.bool)

        for i in range(proj_H):
            beam_mask = proj_y == i
            if i in remain_beam_list:
                mask[beam_mask] = True

        return mask
    
    
    def makeRV(self, pts, labels=None, proj_H=64, proj_W=2048):
        """ Project a pointcloud into a spherical projection image.projection"""
        RVimg = np.full((proj_H, proj_W), -1, dtype=np.float16)
        
        # laser parameters
        fov_down, fov_up = self.fov
        
        fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
        
        # get scan components
        scan_x = pts[:, 0]
        scan_y = pts[:, 1]
        scan_z = pts[:, 2]
        
        # get x^2 + y^2
        scan_x2py2 = np.sqrt(scan_x**2 + scan_y**2)
        yaw = np.arctan2(scan_x, scan_y)
        pitch = np.arctan2(scan_z, scan_x2py2)
        
        # get projections in image coords
        # pdb.set_trace()
        proj_x = 0.5 * (yaw / np.pi + 1.0) # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= proj_W
        proj_y *= proj_H                              # in [0.0, H]
        
        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        proj_x = np.copy(proj_x)  # store a copy in orig order
        
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        proj_y = np.copy(proj_y)  # stope a copy in original order
        
        depth = np.linalg.norm(pts, 2, axis=1)
        
        # order in decreasing depth
        # pdb.set_trace()
        # indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        # indices = indices[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assign to images
        RVimg[proj_y, proj_x] = depth
        
        return RVimg
    
    
    def makeRV_label(self, pts, labels=None, proj_H=64, proj_W=2048):
        """ Project a pointcloud into a spherical projection image.projection"""
        RVimg = np.full((proj_H, proj_W), 255, dtype=np.uint8)
        
        # laser parameters
        fov_down, fov_up = self.fov
        
        fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
        
        # get scan components
        scan_x = pts[:, 0]
        scan_y = pts[:, 1]
        scan_z = pts[:, 2]
        
        # get x^2 + y^2
        scan_x2py2 = np.sqrt(scan_x**2 + scan_y**2)
        yaw = np.arctan2(scan_x, scan_y)
        pitch = np.arctan2(scan_z, scan_x2py2)
        
        # get projections in image coords
        # pdb.set_trace()
        proj_x = 0.5 * (yaw / np.pi + 1.0) # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= proj_W
        proj_y *= proj_H                              # in [0.0, H]
        
        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        proj_x = np.copy(proj_x)  # store a copy in orig order
        
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        proj_y = np.copy(proj_y)  # stope a copy in original order
        
        depth = np.linalg.norm(pts, 2, axis=1)
        
        # order in decreasing depth
        # indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        
        labels = labels.squeeze()
        labels = labels[order].astype(np.uint8)
        # indices = indices[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assign to images
        # RVimg[proj_y, proj_x] = depth
        RVimg[proj_y, proj_x] = labels.squeeze()
        
        return RVimg
        