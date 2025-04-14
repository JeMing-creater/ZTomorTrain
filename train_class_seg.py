import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.class_loader import get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, write_example, resume_train_state, split_metrics
from src.eval import calculate_f1_score, specificity, quadratic_weighted_kappa, top_k_accuracy, calculate_metrics, accumulate_metrics, compute_final_metrics

from src.model.HWAUNETR_class import HWAUNETR
from src.model.FMUNETR_class_seg import FMUNETR
from src.model.SwinUNETR import MultiTaskSwinUNETR
from monai.networks.nets import SwinUNETR
from visualization import visualize_for_all

def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
          metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
          post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, step: int):
    # 训练
    model.train()
    for i, image_batch in enumerate(train_loader):
        # get model output.
        logits = model(image_batch['image'])
        class_x, seg_x = logits
        # get label.
        class_label = image_batch['class_label']
        seg_label   = image_batch['label']
        # get loss.
        total_loss = 0
        for name in loss_functions:
            if name == 'focal_loss':
                loss1 = loss_functions[name](class_x, class_label.float())
                loss2 = loss_functions[name](seg_x, seg_label.float())
                loss = loss1 + loss2
            elif name != 'dice_loss':
                loss = loss_functions[name](class_x, class_label.float())
            else:
                loss = loss_functions[name](seg_x, seg_label.float())
            accelerator.log({'Train/' + name: float(loss)}, step=step)
            total_loss += loss

        # get metric compute.
        class_x = post_trans(class_x)
        seg_x   = post_trans(seg_x)
        for metric_name in metrics:
            if metric_name == 'miou_metric':
                y_pred = class_x.unsqueeze(2)
                y      =      y.unsqueeze(2)
            elif metric_name == 'dice_metric' or metric_name == 'hd95_metric':
                y_pred = seg_x
                y      = seg_label
            else:
                y_pred = class_x
                y      = class_label 
            metrics[metric_name](y_pred=y_pred, y=y)

        # loss backward.
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()

        # chack which param not be used to training.
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)

        # log writed.
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
            flush=True
            )
        step += 1
    scheduler.step(epoch)
    
    # compute all metric data.
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        # give every single task metric
        metrics[metric_name].reset()
        metric.update({
                f'Train/{metric_name}': float(batch_acc.mean()),
            })
        
    for metric_name in metrics:
        all_data = []
        for key in metric.keys():
            if metric_name in key:
                all_data.append(metric[key])
        me_data = sum(all_data) / len(all_data)        
        metric.update({f'Train/{metric_name}': float(me_data)})
        
    accelerator.log(metric, step=epoch)
    return metric, step

@torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  inference: monai.inferers.Inferer, val_loader: torch.utils.data.DataLoader,
                  metrics: Dict[str, monai.metrics.CumulativeIterationMetric], step: int,
                  post_trans: monai.transforms.Compose, accelerator: Accelerator, test: bool=False):
    # 验证
    model.eval()
    if test:
        flag = 'Test'
    else:
        flag = 'Val'
    
    for i, image_batch in enumerate(val_loader):
        
        logits = model(image_batch['image'])  

        class_x, seg_x = logits
        # get label.
        class_label = image_batch['class_label']
        seg_label   = image_batch['label']

        total_loss = 0

        # get loss.
        total_loss = 0
        for name in loss_functions:
            if name == 'focal_loss':
                loss1 = loss_functions[name](class_x, class_label.float())
                loss2 = loss_functions[name](seg_x, seg_label.float())
                loss = loss1 + loss2
            elif name != 'dice_loss':
                loss = loss_functions[name](class_x, class_label.float())
            else:
                loss = loss_functions[name](seg_x, seg_label.float())
            accelerator.log({f'{flag}/' + name: float(loss)}, step=step)
            total_loss += loss

        # log writed.
        accelerator.log({
            f'{flag}/Total Loss': float(total_loss),
        }, step=step)

        # get metric compute.
        class_x = post_trans(class_x)
        seg_x   = post_trans(seg_x)
        for metric_name in metrics:
            if metric_name == 'miou_metric':
                y_pred = class_x.unsqueeze(2)
                y      =      y.unsqueeze(2)
            elif metric_name == 'dice_metric' or metric_name == 'hd95_metric':
                y_pred = seg_x
                y      = seg_label
            else:
                y_pred = class_x
                y      = class_label 
            metrics[metric_name](y_pred=y_pred, y=y)
        
        accelerator.print(
            f'[{i + 1}/{len(val_loader)}] {flag} Validation Loading...',
            flush=True)
        
        step += 1    

    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        # give every single task metric
        metrics[metric_name].reset()
        # task_num = channel + 1
        metric.update({
                f'{flag}/{metric_name}': float(batch_acc.mean()),
            })
        
    accelerator.log(metric, step=epoch)
    return metric, step


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.checkpoint + str(datetime.now()).replace(' ','_').replace('-','_').replace(':','_').replace('.','_')
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))
    
    accelerator.print('load model...')
    model = FMUNETR(in_chans=3, out_chans=3, fussion = [1, 2, 4, 8], kernel_sizes=[4, 2, 2, 2], depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8], out_indices=[0, 1, 2, 3])
    accelerator.print('load dataset...')
    train_loader, val_loader, test_loader, example = get_dataloader(config)

    # keep example log
    if accelerator.is_main_process == True:
        write_example(example, logging_dir)
    
    inference = monai.inferers.SlidingWindowInferer(roi_size=config.loader.target_size, overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'bce_loss'  : nn.BCEWithLogitsLoss().to(accelerator.device),
        'dice_loss' : monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    
    metrics = {
        'accuracy': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name="accuracy"),
        'f1': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name='f1 score'),
        'specificity': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name="specificity"),
        'recall': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name="recall"),
        'miou_metric':monai.metrics.MeanIoU(include_background=False),
        'dice_metric': monai.metrics.DiceMetric(include_background=True,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
        'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True, reduction=monai.utils.MetricReduction.MEAN_BATCH,
                                                             get_not_nans=True)
    }
    
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])

    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    
    
    # start training
    accelerator.print("Start Training! ")
    train_step = 0
    best_eopch = -1
    val_step = 0

    best_val_metric_data = torch.tensor(0)
    best_val_metrics = {}

    best_test_metric_data = torch.tensor(0)
    best_test_metrics = {}
    
    starting_epoch = 0
    
    if config.trainer.resume:
        model, optimizer, scheduler, starting_epoch, train_step, best_val_metric_data, best_test_metric_data, best_val_metrics, best_test_metrics = utils.resume_train_state(model, '{}'.format(
            config.finetune.checkpoint), optimizer, scheduler, train_loader, accelerator, seg=False)
        val_step = train_step
    
    model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader, test_loader)
    
    best_val_metric_data = torch.Tensor([best_val_metric_data]).to(accelerator.device)
    best_test_metric_data = torch.Tensor([best_test_metric_data]).to(accelerator.device)
    
    for epoch in range(starting_epoch, config.trainer.num_epochs):
        train_metric, train_step = train_one_epoch(model, loss_functions, train_loader,
                     optimizer, scheduler, metrics,
                     post_trans, accelerator, epoch, train_step)


        final_metrics, val_step = val_one_epoch(model, inference, val_loader,metrics, val_step, post_trans, accelerator)

        val_class = final_metrics['Val/accuracy']
        val_seg   = final_metrics['Val/dice_metric']
        val_data = val_class + val_seg

        # 保存模型
        if val_data > best_val_metric_data:
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best")
            best_val_metric_data = val_data
            best_val_metrics = final_metrics
            # 记录最优test acc
            final_metrics, _ = val_one_epoch(model, inference, test_loader,
                                                                   metrics, -1,
                                                                   post_trans, accelerator, test=True)
            test_class = final_metrics['Test/accuracy']
            test_seg   = final_metrics['Test/dice_metric']
            test_data = test_class + test_seg
            best_test_metric_data = test_data
            best_test_metrics = final_metrics


        accelerator.print(f'Epoch [{epoch+1}/{config.trainer.num_epochs}] Now train class acc: {train_metric["Train/accuracy"]}, Now train seg dice: {train_metric["Train/dice_metric"]}\n')

        accelerator.print(f'Epoch [{epoch+1}/{config.trainer.num_epochs}] Now Val class acc: {best_val_metrics["Val/accuracy"]}, Now Val seg dice: {best_val_metrics["Val/dice_metric"]}\n')

        accelerator.print(f'Epoch [{epoch+1}/{config.trainer.num_epochs}] Now Test class acc: {best_test_metrics["Test/accuracy"]}, Now Test seg dice: {best_test_metrics["Test/dice_metric"]}\n')

        accelerator.print('Cheakpoint...')
        accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint")
        torch.save({'epoch': epoch, 'best_top_1': best_val_metric_data, 'best_metrics': best_val_metrics, 'best_test_top_1': best_test_metric_data, 'best_test_metrics': best_test_metrics},
                    f'{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint/epoch.pth.tar')
        
    accelerator.print(f"best class acc: {best_test_metrics['Test/accuracy']}\n")
    accelerator.print(f"best seg acc:   {best_test_metrics['Test/dice_metric']}\n")
    accelerator.print(f"best metrics:   {best_test_metrics}")
    

