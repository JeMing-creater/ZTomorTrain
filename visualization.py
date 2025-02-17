import os
import sys
from datetime import datetime
from typing import Dict

import monai
import pytz
import torch
import yaml
import numpy as np
import nibabel as nib
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader, get_transforms, read_usedata, load_MR_dataset_images
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model, MetricSaver, load_model_dict, resume_train_state, ensure_directory_exists

from src.model.HWAUNETR import HWAUNETR

@torch.no_grad()
def visualize_for_single(config, model, accelerator):
    model.eval()
    choose_image = config.loader.dataPath + '/' + config.visualization.choose_dir + '/' + f'{config.visualization.choose_image}'
    accelerator.print('visualize for image: ', choose_image)

    load_transform, _, _ = get_transforms(config=config)
    
    images = []
    labels = []
    image_size = []
    affines = []
    for i in range(len(config.loader.checkModels)):
        image_path = choose_image + '/' + config.loader.checkModels[i] + '/' + f'{config.visualization.choose_image}.nii.gz'
        label_path = choose_image + '/' + config.loader.checkModels[i] + '/' + f'{config.visualization.choose_image}seg.nii.gz'
        
        batch = load_transform[i]({
            'image': image_path,
            'label': label_path
        })
        
        images.append(batch['image'].unsqueeze(1))
        labels.append(batch['label'].unsqueeze(1))
        image_size.append(tuple(batch['image_meta_dict']['spatial_shape'][i].item() for i in range(3)))
        affines.append(batch['label_meta_dict']['affine'])
        
    image_tensor = torch.cat(images, dim=1)
    label_tensor = torch.cat(labels, dim=1)
     
    inference = monai.inferers.SlidingWindowInferer(roi_size=config.loader.target_size, overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])
    
    img = inference(image_tensor.to(accelerator.device), model.to(accelerator.device))
    seg = post_trans(img[0])
    
    for i in range(len(config.loader.checkModels)):
        # seg_now = monai.transforms.Resize(spatial_size=image_size[i], mode="nearest")(seg)
        seg_now = monai.transforms.Resize(spatial_size=image_size[i], mode=("nearest-exact"))(seg[i].unsqueeze(0))
        seg_now = seg_now[0]
        affine = affines[i]
        seg_out = np.zeros((seg_now.shape[0], seg_now.shape[1], seg_now.shape[2]))
        
        seg_now = seg_now.cpu()
        seg_out[seg_now==1] = 1
        res = nib.Nifti1Image(seg_out.astype(np.uint8), affine)
        
        save_path  = config.visualization.image_path + '/' + config.visualization.choose_dir + '/' + f'{config.visualization.choose_image}' + '/' + config.loader.checkModels[i]
        ensure_directory_exists(save_path)
        picture = nib.load(choose_image + '/' + config.loader.checkModels[i] + '/' + f'{config.visualization.choose_image}seg.nii.gz')
        
        qform = picture.get_qform()
        res.set_qform(qform)
        sfrom = picture.get_sform()
        res.set_sform(sfrom)
        
        original_str = f"{save_path}/{config.visualization.choose_image}inference.nii.gz"

        print('save ', original_str)
        # 然后保存 NIFTI 图像
        nib.save(
            res,
            original_str,
        )
 
@torch.no_grad()
def visualize_for_all(config, image_list, model, accelerator):
    model.eval()
    load_transform, _, _ = get_transforms(config=config)
    for i in range(len(image_list)):
        visualization_choose_image = image_list[i]['image'][0].split('/')[-1].replace('.nii.gz','')
        visualization_choose_dir   = ''
        for word in image_list[i]['image'][0].split('/'):
            if 'urgical' in word:
                visualization_choose_dir   = word
        choose_image = config.loader.dataPath + '/' + visualization_choose_dir + '/' + f'{visualization_choose_image}'
        
        
        accelerator.print('visualize for image: ', choose_image)

    
        images = []
        labels = []
        image_size = []
        affines = []
        for i in range(len(config.loader.checkModels)):
            image_path = choose_image + '/' + config.loader.checkModels[i] + '/' + f'{visualization_choose_image}.nii.gz'
            label_path = choose_image + '/' + config.loader.checkModels[i] + '/' + f'{visualization_choose_image}seg.nii.gz'
            
            batch = load_transform[i]({
                'image': image_path,
                'label': label_path
            })
            images.append(batch['image'].unsqueeze(1))
            labels.append(batch['label'].unsqueeze(1))
            image_size.append(tuple(batch['image_meta_dict']['spatial_shape'][i].item() for i in range(3)))
            affines.append(batch['label_meta_dict']['affine'])
            
        image_tensor = torch.cat(images, dim=1)
        label_tensor = torch.cat(labels, dim=1)
        
        inference = monai.inferers.SlidingWindowInferer(roi_size=config.loader.target_size, overlap=0.5,
                                                        sw_device=accelerator.device, device=accelerator.device)
        post_trans = monai.transforms.Compose([
            monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
        ])
        
        img = inference(image_tensor.to(accelerator.device), model.to(accelerator.device))
        seg = post_trans(img[0])
        
        for i in range(len(config.loader.checkModels)):
            seg_now = monai.transforms.Resize(spatial_size=image_size[i], mode=("nearest-exact"))(seg[i].unsqueeze(0))
            seg_now = seg_now[0]
            affine = affines[i]
            seg_out = np.zeros((seg_now.shape[0], seg_now.shape[1], seg_now.shape[2]))
            
            seg_now = seg_now.cpu()
            seg_out[seg_now==1] = 1
            res = nib.Nifti1Image(seg_out.astype(np.uint8), affine)
            
            save_path  = config.visualization.image_path + '/' + config.visualization.choose_dir + '/' + f'{config.visualization.choose_image}' + '/' + config.loader.checkModels[i]
            ensure_directory_exists(save_path)
            picture = nib.load(choose_image + '/' + config.loader.checkModels[i] + '/' + f'{visualization_choose_image}seg.nii.gz')
            
            qform = picture.get_qform()
            res.set_qform(qform)
            sfrom = picture.get_sform()
            res.set_sform(sfrom)
            
            
            original_str = f"{save_path}/{visualization_choose_image}inference.nii.gz"

            print('save ', original_str)
            # 然后保存 NIFTI 图像
            nib.save(
                res,
                original_str,
            )

def warm_up(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, accelerator: Accelerator):
    # 训练
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch['image'])
        total_loss = 0
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch['label'])
            total_loss += loss
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.print(
            f'Warm up [{i + 1}/{len(train_loader)}] Warm up Loss:{total_loss}',
            flush=True)
    scheduler.step(0)
    return model


@torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  inference: monai.inferers.Inferer, val_loader: torch.utils.data.DataLoader,
                  metrics: Dict[str, monai.metrics.CumulativeIterationMetric], step: int,
                  post_trans: monai.transforms.Compose, accelerator: Accelerator):
    # 验证
    model.eval()
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch['image'], model)
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch['label'])
        accelerator.print(
            f'[{i + 1}/{len(val_loader)}] Validation Loading...',
            flush=True)
        
        step += 1
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0]
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc.to(accelerator.device)) / accelerator.num_processes
        metrics[metric_name].reset()
        if metric_name == 'dice_metric':
            metric.update({
                f'Val/mean {metric_name}': float(batch_acc.mean()),
                f'Val/Object1 {metric_name}': float(batch_acc[0]),
                f'Val/Object2 {metric_name}': float(batch_acc[1])
            })
            dice_acc = torch.Tensor([metric['Val/mean dice_metric']]).to(accelerator.device)
            dice_class = batch_acc
        else:
            metric.update({
                f'Val/mean {metric_name}': float(batch_acc.mean()),
                f'Val/Object1 {metric_name}': float(batch_acc[0]),
                f'Val/Object2 {metric_name}': float(batch_acc[1])
            })
            hd95_acc = torch.Tensor([metric['Val/mean hd95_metric']]).to(accelerator.device)
            hd95_class = batch_acc
    return dice_acc, dice_class, hd95_acc, hd95_class, step


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], project_dir=logging_dir)
    
    model = HWAUNETR(in_chans=2, out_chans=2, kernel_sizes=[2, 2, 2, 2], depths=[1, 1, 1, 1], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8], out_indices=[0, 1, 2, 3])
    
    train_loader, val_loader = get_dataloader(config)
    inference = monai.inferers.SlidingWindowInferer(roi_size=config.loader.target_size, overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    
    model = load_pretrain_model('model_store' + '/' + f'{config.finetune.checkpoint}' + "/best/new/model.pth", model, accelerator)
    
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)
                                    
    # model = warm_up(model, loss_functions, train_loader,
    #         optimizer, scheduler, accelerator)
    
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=True,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
        # 'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True,
        #                                                      reduction=monai.utils.MetricReduction.MEAN_BATCH,
        #                                                      get_not_nans=False)
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])
    
    dice_acc, dice_class, hd95_acc, hd95_class, val_step = val_one_epoch(model, inference, val_loader,
                                                                   metrics, 0,
                                                                   post_trans, accelerator)
    accelerator.print(f'dice acc: {dice_acc} best class: {dice_class}')
    
    if config.visualization.for_single == True:
        visualize_for_single(config=config, model = model, accelerator = accelerator)
    else:
        datapath = config.loader.dataPath
        use_models = config.loader.checkModels
        
        datapath1 = datapath + '/' + 'NonsurgicalMR' + '/'
        datapath2 = datapath + '/' + 'SurgicalMR' + '/'
        usedata1 = datapath + '/' + 'NonsurgicalMR.txt'
        usedata2 = datapath + '/' + 'SurgicalMR.txt'
        usedata1 = read_usedata(usedata1)
        usedata2 = read_usedata(usedata2)
        
        data1 = load_MR_dataset_images(datapath1, usedata1, use_models)
        data2 = load_MR_dataset_images(datapath2, usedata2, use_models)
        
        data = data1 + data2
        
        visualize_for_all(config=config, image_list=data, model = model, accelerator = accelerator)
    
    
    
    
    