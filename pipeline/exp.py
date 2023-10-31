from stat import FILE_ATTRIBUTE_ENCRYPTED
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import pdb
import yaml
import os

from utils.utils import get_semcolor_common
from utils.ply_vis import write_ply
from utils.eval import SemanticEval
import utils.losses as losses

import MinkowskiEngine as ME
import copy

import models as models


class ExpBase(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        
        self.save_hyperparameters(ignore='model')
        
        self.model = model
        
        self.model_params = cfg.model
        self.train_params = cfg.train_params
        self.generalization_params = cfg.generalization_params
        
        self.exp_params = cfg.exp_params
        self.log_freq = cfg.exp_params.log_freq
        
        if self.train_params.use_pretrained:
            self.load_pretrained(self.model, self.train_params.pretrained_ckpt_path, 'model')
            print(f'==> Load pretrained weights from {self.train_params.pretrained_ckpt_path}')
        
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def load_pretrained(self, model, dict_path, model_name):
        state_dict = torch.load(dict_path)['state_dict']
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if model_name in key:
                new_state_dict[key.replace(model_name + '.', '')] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(new_state_dict, strict=False)
        
        
class ExpSemantic(ExpBase):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        
        ds = cfg.dataset_SemKITTI.voxel_size
        self.voxel_size = ds
        self.quantization = torch.Tensor([ds, ds, ds])
        self.class_num = cfg.model.out_classes
        self.common_map = yaml.safe_load(open('configs/label_mapping/common_map.yaml', 'r'))
        self.class_str = self.common_map['labels']
        
        self.source = cfg.generalization_params.source
        self.target = cfg.generalization_params.target
        self.target_mapping = {}
        for i, target in enumerate(self.target):
            self.target_mapping[i] = target
            
        self.configure_evaluator()
        self.configure_criterion()
        
        self.vis_frames = [[0] for _ in self.target]
        
    def configure_optimizers(self):
        opt = optim.Adam(list(self.model.parameters()),
                         lr=self.train_params.learning_rate)
        return opt
        
    def configure_evaluator(self):
        target = copy.deepcopy(self.target)
        if 'poss' in target: 
            assert target[-1] == 'poss', 'poss should be at the last.'
            idx_poss = target.index('poss')
            target.pop(idx_poss)
        self.evaluators = [SemanticEval(self.class_num+1, None, [0]) for _ in target]
        
        if 'poss' in self.target:
            self.evaluators.append(SemanticEval(self.class_num+1, None, [0,3,4,5,8,10]))
        
        for evaluator in self.evaluators:
            evaluator.reset()
        return
    
    def configure_criterion(self):
        if self.train_params.criterion_sem == 'wce':
            if self.source == 'kitti':
                ce_weight = self.common_map['kitti_weight']
            elif self.source == 'waymo':
                ce_weight = self.common_map['waymo_weight']
            else:
                raise NotImplementedError
            
            weight = torch.zeros(self.class_num, dtype=torch.float)
            for i in range(self.class_num):
                weight[i] = ce_weight[i+1]
            self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
            
        elif self.train_params.criterion_sem == 'dice':
            self.criterion = losses.DICELoss(ignore_label=-1)
            
        elif self.train_params.criterion_sem == 'softdice':
            self.criterion = losses.SoftDICELoss(ignore_label=-1)
            
    def test_setup(self, target, save_result):
        self.target = target
        self.target_mapping = {}
        for i, target in enumerate(self.target):
            self.target_mapping[i] = target
        
        self.configure_evaluator()
        self.save_result = save_result

        return
    
    def forward(self, batch):
        coordinates = batch['coordinates'] # N, 3
        features = batch['features'] # N, 1
        
        if self.training:
            sparse_tensor = ME.SparseTensor(coordinates=coordinates.int(), features=features)
            predicted_sparse_tensor, _ = self.model(sparse_tensor)
            out = predicted_sparse_tensor
            # sparse voxel prediction
                 
        else:
            quantization = self.quantization.type_as(coordinates[0])
            coordinates = [torch.div(coordinate, quantization) for coordinate in coordinates]
            
            coords, features = ME.utils.sparse_collate(coordinates, features)
            tensor_field = ME.TensorField(features=features, coordinates=coords)

            sparse_tensor = tensor_field.sparse() # N_sparse, 1

            predicted_sparse_tensor, _ = self.model(sparse_tensor)
            
            out = predicted_sparse_tensor.slice(tensor_field) # pointwise prediction N 10 (Tensorfield)
            out.coordinates[:, 1:] = torch.mul(out.coordinates[:, 1:], quantization)
            # point-wise prediction
        return out
    
    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        
        out = self.forward(batch)
        preds = out.F

        loss = 0
        # semantic_lovasz_loss = lovasz_softmax(F.softmax(preds, dim=1), labels-1, ignore=-1) # TODO: TBD

        
        semantic_vox_loss =  self.criterion(preds, labels-1)
        loss += semantic_vox_loss
        
        return {'loss': loss}
    
    def training_epoch_end(self, training_step_outputs):
        
        loss = 0
        for output in training_step_outputs:
            loss += output['loss']
        
        split = 'training'
        self.log(f'{split}_loss', loss / len(training_step_outputs), rank_zero_only=True)
        return
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch['batch_size']
        labels = batch['labels']
        
        with torch.no_grad():
            out = self.forward(batch)

            for b_idx in range(batch_size):
                feats = out.features_at(b_idx)
                predictions = torch.add(torch.argmax(feats, 1), 1).cpu().numpy()
                labels = labels[b_idx].squeeze(1).cpu().numpy()
                
                if self.target_mapping[dataloader_idx] == 'poss':
                    predictions[predictions==10] = 9
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)
                else:
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)
             
        
                # visualize
                if batch_idx in self.vis_frames[dataloader_idx]:
                    save_dir = self.logger.log_dir + '/vis/' + self.target_mapping[dataloader_idx]
                    os.makedirs(save_dir, exist_ok=True)
                    save_filename = os.path.join(save_dir, f"epoch={self.current_epoch}-step={self.global_step}-name={batch_idx}")
                    
                    coords = out.coordinates_at(b_idx)
                    self.visualize(save_filename, coords.cpu().numpy(), predictions, labels)
        return
    
    def validation_epoch_end(self, val_step_outputs):
        if self.global_step == 0: return
        
        output_list = []
        miou = 0
        mioup = 0
        hiou = 0
        hioup = 0
        num = 0
        nump = 0
        for idx, evaluator in enumerate(self.evaluators):
            iou, output_dict, iou_th, iou_st = self.log_metric(evaluator)
            output_list.append(output_dict)
            
            self.log(f'val_mIoU_{self.target_mapping[idx]}', iou, rank_zero_only=True)
            self.log(f'val_mIoU_th_{self.target_mapping[idx]}', iou_th, rank_zero_only=True)
            self.log(f'val_mIoU_st_{self.target_mapping[idx]}', iou_st, rank_zero_only=True)
            
            evaluator.reset()
            if not self.target_mapping[idx] == 'poss':
                miou += iou
                hiou += 1/iou
                num += 1
            mioup += iou
            hioup += 1/iou
            nump += 1
        
        if num == 0:
            self.log(f'val_mIoUP', mioup/nump, rank_zero_only=True)
            self.log(f'val_hIoUP', nump/hioup, rank_zero_only=True)
            self.write_txt(output_list, mioup=mioup/nump, hioup=nump/hioup)
        else:
            self.log(f'val_mIoU', miou/num, rank_zero_only=True)
            self.log(f'val_mIoUP', mioup/nump, rank_zero_only=True)
            self.log(f'val_hIoU', num/hiou, rank_zero_only=True)
            self.log(f'val_hIoUP', nump/hioup, rank_zero_only=True)
            self.write_txt(output_list, miou=miou/num, mioup=mioup/nump, hiou=num/hiou, hioup=nump/hioup)
        return
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch['batch_size']
        labels = batch['labels']
        
        with torch.no_grad():
            out = self.forward(batch)

            for b_idx in range(batch_size):
                feats = out.features_at(b_idx)
                predictions = torch.add(torch.argmax(feats, 1), 1).cpu().numpy()
                labels = labels[b_idx].squeeze(1).cpu().numpy()
                
                if self.target_mapping[dataloader_idx] == 'poss':
                    predictions[predictions==10] = 9
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)
                else:
                    self.evaluators[dataloader_idx].addBatchSemIoU(predictions, labels)

                # visualize
                if self.save_result:
                    save_dir = os.path.join(self.logger.log_dir, self.target_mapping[dataloader_idx])
                    os.makedirs(save_dir, exist_ok=True)
                    save_filename = os.path.join(save_dir, f"{batch_idx:05}")
                    
                    # background masking for visualization
                    bg_mask = labels == 0
                    predictions[bg_mask] = 0
                    
                    coords = out.coordinates_at(b_idx)
                    self.visualize(save_filename, coords.cpu().numpy(), predictions, labels)
        return
    
    def test_epoch_end(self, test_step_outputs):
        output_list = []
        miou_list = []
        miou = 0
        mioup = 0
        hiou = 0
        hioup = 0
        num = 0
        nump = 0
        for idx, evaluator in enumerate(self.evaluators):
            iou, output_dict, _, _ = self.log_metric(evaluator)
            output_list.append(output_dict)
            miou_list.append(iou)

            evaluator.reset()
            if not self.target_mapping[idx] == 'poss':
                miou += iou
                hiou += 1/iou
                num += 1
            mioup += iou
            hioup += 1/iou
            nump += 1
        
        for idx, iou in enumerate(miou_list):
            self.log(f'val_mIoU_{self.target_mapping[idx]}', iou, rank_zero_only=True)
        if num == 0:
            self.log(f'val_mIoUP', mioup/nump, rank_zero_only=True)
            self.log(f'val_hIoUP', nump/hioup, rank_zero_only=True)
            self.write_txt(output_list, mioup=mioup/nump, hioup=nump/hioup)
        else:
            self.log(f'val_mIoU', miou/num, rank_zero_only=True)
            self.log(f'val_hIoU', num/hiou, rank_zero_only=True)
            self.log(f'val_mIoUP', mioup/nump, rank_zero_only=True)
            self.log(f'val_hIoUP', nump/hioup, rank_zero_only=True)
            self.write_txt(output_list, miou=miou/num, mioup=mioup/nump, hiou=num/hiou, hioup=nump/hioup)
        return
    
    def visualize(self, filename, coords, preds, labels):
        color_preds = get_semcolor_common(preds)
        color_labels = get_semcolor_common(labels)
       
        write_ply(filename + '-gt.ply', [coords, color_labels], ['x','y','z','red','green','blue'])
        write_ply(filename + '-pd.ply', [coords, color_preds], ['x','y','z','red','green','blue'])
        return
    
    def log_metric(self, evaluator):
        assert isinstance(evaluator, SemanticEval)

        class_IoU, class_all_IoU, class_IoU_things, class_IoU_stuff = evaluator.getSemIoU()

        # now make a nice dictionary
        output_dict = {}

        # make python variables
        class_IoU = class_IoU.item()
        class_all_IoU = class_all_IoU.flatten().tolist()
        class_IoU_things = class_IoU_things.item()
        class_IoU_stuff = class_IoU_stuff.item()

        output_dict["all"] = {}
        output_dict["all"]["IoU"] = class_IoU
        output_dict["all"]["IoU_things"] = class_IoU_things
        output_dict["all"]["IoU_stuff"] = class_IoU_stuff

        for idx, iou in enumerate(class_all_IoU):
            class_str = self.class_str[idx]
            output_dict[class_str] = {}
            output_dict[class_str]["IoU"] = iou

        mIoU = output_dict["all"]["IoU"]
        return mIoU, output_dict, class_IoU_things, class_IoU_stuff

    def write_txt(self, output_dict, miou=None, mioup=None, hiou=None, hioup=None):
        file_dir = os.path.join(self.logger.log_dir, 'metric.txt')
        fd = open(file_dir, 'a')
        fd.write('-'*144 + '\n')
        fd.write('|'+ ' '*67 + f'Epoch {self.current_epoch:02}' + ' '*67 + '|\n')
        fd.write('|{}'.format(' '*10))
        
        for k, v in output_dict[0].items():
                fd.write('|{}'.format(k.ljust(10)[:10]))
        fd.write('|\n')
                
        for idx, output in enumerate(output_dict):
            fd.write('|{}'.format(self.target_mapping[idx].ljust(10)))
            for k, v in output.items():
                fd.write('|{:.4f}    '.format(v['IoU']))
            fd.write('|\n')
        
        if miou is not None:
            fd.write('|{}'.format('mean'.ljust(10)))
            fd.write('|{:.4f}    '.format(miou))
            fd.write('|\n')
        if mioup is not None:
            fd.write('|{}'.format('meanP'.ljust(10)))
            fd.write('|{:.4f}    '.format(mioup))
            fd.write('|\n')
        if hiou is not None:
            fd.write('|{}'.format('harmonic'.ljust(10)))
            fd.write('|{:.4f}    '.format(hiou))
            fd.write('|\n')
        if hioup is not None:
            fd.write('|{}'.format('harmonicP'.ljust(10)))
            fd.write('|{:.4f}    '.format(hioup))
            fd.write('|\n')
        
        fd.write('-'*144 + '\n')
        
        fd.close()
        return


class ExpDGLSS(ExpSemantic):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        
        self.out_shape = self.model_params.out_shape
        self.min_coordinate = self.model_params.min_coordinate
        self.positive_num = self.generalization_params.positive_num
        
        self.consistency_loss_weight = cfg.train_params.consistency_loss_weight
        self.correlation_loss_weight = cfg.train_params.correlation_loss_weight

        self.use_consistency = cfg.train_params.use_consistency
        self.use_correlation = cfg.train_params.use_correlation
        self.affinity_threshold = cfg.train_params.affinity_threshold
        
        self.metric_learner = models.MetricLearner()

    def configure_criterion(self):
        if self.train_params.criterion_sem == 'wce':
            if self.source == 'kitti':
                ce_weight = self.common_map['kitti_weight']
            elif self.source == 'waymo':
                ce_weight = self.common_map['waymo_weight']
            else:
                raise NotImplementedError
            
            weight = torch.zeros(self.class_num, dtype=torch.float)
            for i in range(self.class_num):
                weight[i] = ce_weight[i+1]
            self.criterion_sem = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
            
        elif self.train_params.criterion_sem == 'dice':
            self.criterion_sem = losses.DICELoss(ignore_label=-1)
            
        elif self.train_params.criterion_sem == 'softdice':
            self.criterion_sem = losses.SoftDICELoss(ignore_label=-1)
            
        if self.train_params.criterion_const == 'L1':
            self.criterion_const = nn.L1Loss()
        else:
            raise NotImplementedError
    
    def configure_optimizers(self):
        opt = optim.Adam(list(self.model.parameters()) + list(self.metric_learner.parameters()),
                         lr=self.train_params.learning_rate)
        return opt
    
    def forward(self, batch, use_metric=False):
        coordinates = batch['coordinates'] # N, 3
        features = batch['features'] # N, 1
        
        if self.training:
            coordinates_aug = batch['coordinates_aug']
            features_aug = batch['features_aug']
            
            coordinates_aug[:,0] += batch['batch_size']
            
            concat_coordinates = torch.cat((coordinates, coordinates_aug), dim=0)
            concat_features = torch.cat((features, features_aug), dim=0)
            sparse_tensor = ME.SparseTensor(coordinates=concat_coordinates.int(), features=concat_features)
            predicted_sparse_tensor, encoder_tensor, encoder_tensor_bottle = self.model(sparse_tensor, use_both=True)
            
            if use_metric:
                encoder_tensor = self.metric_learner(encoder_tensor.F)
                
            return predicted_sparse_tensor, encoder_tensor, encoder_tensor_bottle
                 
        else:
            quantization = self.quantization.type_as(coordinates[0])
            coordinates = [torch.div(coordinate, quantization) for coordinate in coordinates]
            
            coords, features = ME.utils.sparse_collate(coordinates, features)
            tensor_field = ME.TensorField(features=features, coordinates=coords)

            sparse_tensor = tensor_field.sparse() # N_sparse, 1

            predicted_sparse_tensor, _ = self.model(sparse_tensor)
            
            out = predicted_sparse_tensor.slice(tensor_field) # pointwise prediction N 10 (Tensorfield)
            out.coordinates[:, 1:] = torch.mul(out.coordinates[:, 1:], quantization)
            # point-wise prediction
        return out
    
    def visualize_aug(self, batch, batch_idx):
        labels = batch['labels'].detach().cpu().numpy()
        xyz = batch['coordinates'].detach().cpu().numpy()
        aug_labels = batch['labels_aug'].detach().cpu().numpy()
        aug_xyz = batch['coordinates_aug'].detach().cpu().numpy()
        
        batch_ids = np.unique(xyz[:, 0])
        for b_id in batch_ids:
            orig_mask = xyz[:,0]==b_id
            aug_mask = aug_xyz[:,0]==b_id
            xyz_orig = xyz[orig_mask][:,1:]
            label_orig = labels[orig_mask]
            xyz_aug = aug_xyz[aug_mask][:,1:]
            label_aug = aug_labels[aug_mask]
            
            orig_color = get_semcolor_common(label_orig)
            aug_color = get_semcolor_common(label_aug)
                    
            save_dir = self.logger.log_dir + f'/vis/{batch_idx}/'
            os.makedirs(save_dir, exist_ok=True)
            write_ply(save_dir+f"{b_id}_orig.ply", [xyz_orig, orig_color], ['x','y','z','red','green','blue'])
            write_ply(save_dir+f"{b_id}_aug.ply", [xyz_aug, aug_color], ['x','y','z','red','green','blue'])
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['batch_size']
        labels = batch['labels']
        labels_aug = batch['labels_aug']
        
        use_metric = self.use_correlation or self.use_contrastive
        out_cat, metric_feat, enc_bottle_cat = self.forward(batch, use_metric=use_metric)
        
        loss = 0
        
        # segmentation loss
        labels_cat = torch.cat((labels, labels_aug),dim=0)
        semantic_vox_loss =  self.criterion_sem(out_cat.F, labels_cat-1)
        loss += semantic_vox_loss
        
        # Sparsity Invariant Feature Consistency loss
        consistency_loss = torch.tensor(0.0, device=labels.device)
        if self.use_consistency:
            consistency_loss += losses.SIFCloss(features=enc_bottle_cat, 
                                                batch_size=batch_size, 
                                                positive_num=self.positive_num, 
                                                affinity_thld=self.affinity_threshold)
                    
        consistency_loss = self.consistency_loss_weight * consistency_loss / (self.positive_num * batch_size)
        loss += consistency_loss
        
        # Semantic Correlation Consistency loss
        batch_idx_list = out_cat.coordinates[:,0]
        correlation_loss = losses.SCCloss(features=metric_feat, 
                                          labels=labels_cat-1, 
                                          batch_ids=batch_idx_list, 
                                          unique_id=torch.unique(batch_idx_list), 
                                          batch_size=batch_size*(1+self.positive_num))
        
        correlation_loss = self.correlation_loss_weight * correlation_loss
        
        loss += correlation_loss
                
        self.log('sem', semantic_vox_loss, prog_bar=True, logger=False)
        self.log('cons', consistency_loss, prog_bar=True, logger=False)
        self.log('corr', correlation_loss, prog_bar=True, logger=False)
        
        return {'loss': loss, 'segmentation_loss': semantic_vox_loss.detach(), 'consistency_loss': consistency_loss.detach(), 
                'correlation_loss': correlation_loss.detach()}
        
    def training_epoch_end(self, training_step_outputs):
        
        loss = 0
        segmentation_loss = 0
        consistency_loss = 0
        correlation_loss = 0
        
        for output in training_step_outputs:
            loss += output['loss']
            segmentation_loss += output['segmentation_loss']
            consistency_loss += output['consistency_loss']
            correlation_loss += output['correlation_loss']
        
        split = 'training'
        self.log(f'{split}_loss', loss / len(training_step_outputs), rank_zero_only=True)
        self.log(f'{split}_segmentation_loss', segmentation_loss / len(training_step_outputs), rank_zero_only=True)
        self.log(f'{split}_consistency_loss', consistency_loss / len(training_step_outputs), rank_zero_only=True)
        self.log(f'{split}_correlation_loss', correlation_loss / len(training_step_outputs), rank_zero_only=True)
        return
    
    def get_mask(self, coord, coord_aug):
        with torch.no_grad():
            coord = coord.detach()
            coord_aug = coord_aug.detach()
            N = coord.shape[0]
            M = coord_aug.shape[0]
            coord = coord[None].repeat(M,1,1)
            coord_aug = coord_aug[:,None].repeat(1,N,1)
            
            mask = (coord - coord_aug).abs().sum(-1) == 0
        return torch.nonzero(mask)[:,1]

