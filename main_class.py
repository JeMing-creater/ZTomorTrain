import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
from src.utils import Logger, resume_train_state
from src.eval import calculate_f1_score, specificity, quadratic_weighted_kappa, top_k_accuracy, calculate_metrics, accumulate_metrics, compute_final_metrics

from src.model.HWAUNETR import HWAUNETR
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
        logits = model(image_batch['image'])
        total_loss = 0
        log = ''
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch['label'])
            accelerator.log({'Train/' + name: float(loss)}, step=step)
            total_loss += alpth * loss
        val_outputs = post_trans(logits)
        # for metric_name in metrics:
        #     metrics[metric_name](y_pred=val_outputs, y=image_batch['label'])
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
            flush=True
            )
        step += 1
    scheduler.step(epoch)
    # metric = {} 
    # for metric_name in metrics:
    #     batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
    #     if accelerator.num_processes > 1:
    #         batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
    #     metric.update({
    #         f'Train/mean {metric_name}': float(batch_acc.mean()),
    #         f'Train/Object1 {metric_name}': float(batch_acc[0]),
    #         f'Train/Object2 {metric_name}': float(batch_acc[1]),
    #         f'Train/Object3 {metric_name}': float(batch_acc[2])
    #     })
    # accelerator.log(metric, step=epoch)
    return step

@torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  inference: monai.inferers.Inferer, val_loader: torch.utils.data.DataLoader,
                  metrics: Dict[str, monai.metrics.CumulativeIterationMetric], step: int,
                  post_trans: monai.transforms.Compose, accelerator: Accelerator, test: bool=False):
    # 验证
    model.eval()
    accumulated_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for i, image_batch in enumerate(val_loader):
        # logits = inference(model, image_batch['image'])
        logits = model(image_batch['image'])
        val_outputs = post_trans(logits)
        # 计算当前批次的度量
        metrics = calculate_metrics(val_outputs, image_batch['class_label'])
        
        # 累积度量
        accumulated_metrics = accumulate_metrics(metrics, accumulated_metrics)
    #     logits = inference(image_batch['image'], model)
    #     val_outputs = post_trans(logits)
    #     for metric_name in metrics:
    #         metrics[metric_name](y_pred=val_outputs, y=image_batch['label'])
        accelerator.print(
            f'[{i + 1}/{len(val_loader)}] Validation Loading...',
            flush=True)
        
        step += 1
        
    # 使用gather_for_metrics()将所有GPU上的累积度量聚合到一起
    all_gathered_metrics = accelerator.gather_for_metrics(accumulated_metrics)
    
    final_accumulated_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    # 合并所有GPU上的度量
    for metrics in all_gathered_metrics.keys():
        for key in final_accumulated_metrics.keys():
            # 确保metrics[key]是一个数值而不是Tensor，如果是Tensor则需要.item()
            if isinstance(all_gathered_metrics[key], torch.Tensor):
                final_accumulated_metrics[key] += all_gathered_metrics[key].item()
            else:
                final_accumulated_metrics[key] += all_gathered_metrics[key]
    

    # 计算最终度量
    final_metrics = compute_final_metrics(final_accumulated_metrics)
    
    metric = {}
    if test:
        flag = 'Test'
    else:
        flag = 'Val'
    metric.update({
            f'{flag}/Top1': float(final_metrics['accuracy']),
            f'{flag}/precision': float(final_metrics['precision']),
            f'{flag}/recall': float(final_metrics['recall']),
            f'{flag}/specificity': float(final_metrics['specificity']),
            f'{flag}/f1': float(final_metrics['f1'])
        })
    
    return final_metrics, step


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.checkpoint +str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))
    
    accelerator.print('load model...')
    model = MultiTaskSwinUNETR(img_size=config.loader.target_size, in_channels=3, num_tasks=2, num_classes_per_task=1)
    accelerator.print('load dataset...')
    train_loader, val_loader, test_loader = get_dataloader(config)
    
    inference = monai.inferers.SlidingWindowInferer(roi_size=config.loader.target_size, overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'bce_loss':  nn.BCEWithLogitsLoss().to(accelerator.device),
    }
    
    metrics = {
        'top-k': top_k_accuracy,
        'f1_score': calculate_f1_score,
        'specificity': specificity,
        'QWK': quadratic_weighted_kappa,
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
    best_top_1 = torch.tensor(0)
    best_metrics = {}
    best_test_top_1 = torch.tensor(0)
    best_test_metrics = {}
    
    starting_epoch = 0
    
    
    if config.trainer.resume:
        model, optimizer, scheduler, starting_epoch, train_step, best_top_1, best_test_top_1, best_metrics, best_test_metrics = utils.resume_train_state(model, '{}'.format(
            config.finetune.checkpoint), optimizer, scheduler, train_loader, accelerator)
        val_step = train_step
    
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)
    
    best_top_1 = torch.Tensor([best_top_1]).to(accelerator.device)
    
    for epoch in range(starting_epoch, config.trainer.num_epochs):
        train_step = train_one_epoch(model, loss_functions, train_loader,
                     optimizer, scheduler, metrics,
                     post_trans, accelerator, epoch, train_step)

        final_metrics, val_step = val_one_epoch(model, inference, val_loader,metrics, val_step, post_trans, accelerator)
        
        # 保存模型
        if final_metrics['accuracy'] > best_top_1:
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best")
            best_top_1 = final_metrics['accuracy']
            best_metrics = final_metrics
            # 记录最优test acc
            final_metrics, _ = val_one_epoch(model, inference, test_loader,
                                                                   metrics, -1,
                                                                   post_trans, accelerator)
            best_test_top_1 = final_metrics['accuracy']
            best_test_metrics = final_metrics
            
        accelerator.print(f'Epoch [{epoch+1}/{config.trainer.num_epochs}] best acc: {best_top_1}, best test acc: {best_test_top_1}')
        accelerator.print('Cheakpoint...')
        accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint")
        torch.save({'epoch': epoch, 'best_top_1': best_top_1, 'best_metrics': best_metrics, 'best_test_top_1': best_test_top_1, 'best_test_metrics': best_test_metrics},
                    f'{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint/epoch.pth.tar')
        
    accelerator.print(f"best top1: {best_test_top_1}")
    accelerator.print(f"best metrics: {best_test_metrics}")
    

