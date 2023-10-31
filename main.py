import argparse
import yaml
from munch import Munch
import pdb

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import MinkowskiEngine as ME

import models as models
from datasets.initialization import get_dataset, get_test_dataset
import pipeline.exp as exp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        default="configs/config_kitti.yaml",
                        type=str,
                        help="Path to config file")
    parser.add_argument('-ld', '--logdir', default='Test')
    parser.add_argument('-en', '--expname', default='ExpDGLSS')
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    return args

def train(cfg, args):
    training_dataset, validation_dataset = get_dataset(cfg=cfg, debug=args.debug)
    
    training_loader_param = cfg.train_dataloader
    training_loader_param.pop('training')
    training_dataloader = DataLoader(training_dataset, **training_loader_param, collate_fn=training_dataset.collate_fn)
    
    validation_loader_param = cfg.val_dataloader
    validation_loader_param.pop('training')
    validation_dataloader = [DataLoader(val_dataset, **validation_loader_param, collate_fn=val_dataset.collate_fn_test) for val_dataset in validation_dataset]
    
    if cfg.model.out_shape is None:
        cfg.model.out_shape = training_dataset.out_shape
    if cfg.model.min_coordinate is None:
        cfg.model.min_coordinate = training_dataset.min_coordinate
    
    Model = getattr(models, cfg.model.name)
    model = Model(cfg.model.in_feat_size, cfg.model.out_classes)

    # For multi GPU
    # model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    pl_module = getattr(exp, cfg.exp_name)(model, cfg)

    logger = TensorBoardLogger('logs', name=cfg.log_dir, default_hp_metric=True)

    source_name = cfg.generalization_params.source
    
    checkpoint_callback_source = ModelCheckpoint(
        filename=source_name + "-{epoch}-{step}",
        monitor=f'val_mIoU_{source_name}',
        save_last=True,
        save_top_k=10,
        mode='max',
        )
    
    trainer = Trainer(max_steps=cfg.train_params.max_steps, gpus=1, 
                        logger=[logger], callbacks=[checkpoint_callback_source], 
                        check_val_every_n_epoch=1)
    
    if args.resume:
        trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader,
                ckpt_path=cfg.train_params.resume_ckpt)
    else:    
        trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)


def test(cfg, args):
    
    test_dataset = get_test_dataset(cfg=cfg, debug=args.debug)
    test_loader_param = cfg.test_dataloader
    test_loader_param.pop('training')
    test_dataloader = [DataLoader(dataset, **test_loader_param, collate_fn=dataset.collate_fn_test) for dataset in test_dataset]
    
    ckpt_cfg = torch.load(cfg.test_params.ckpt_path, map_location='cuda')['hyper_parameters']['cfg']
    print(f'==> Test with ckpt {cfg.test_params.ckpt_path}')
    
    model = getattr(models, ckpt_cfg.model.name)(ckpt_cfg.model.in_feat_size, ckpt_cfg.model.out_classes)
    
    # For multi GPU
    # model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    
    ckpt_path = cfg.test_params.pop('ckpt_path')
    pl_module = getattr(exp, cfg.exp_name).load_from_checkpoint(ckpt_path, model=model)
    
    log_dir = cfg.test_params.pop('log_dir')
    logger = TensorBoardLogger('test_logs', name=log_dir, default_hp_metric=True)
    
    # test setup. reset the target and log_dir
    pl_module.test_setup(**cfg.test_params)

    trainer = Trainer(gpus=1, logger=[logger])
    trainer.test(pl_module, dataloaders=test_dataloader)



if __name__ == '__main__':
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    
    cfg.exp_name = args.expname
    cfg.log_dir = args.logdir
    if args.test:
        cfg.test_params.log_dir = args.logdir

    if args.test:
        test(cfg, args)
    else:
        train(cfg, args)
