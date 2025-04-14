import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
from src.class_loader import get_dataloader, get_transforms
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, write_example, resume_train_state, split_metrics, load_model_dict
from src.eval import calculate_f1_score, specificity, quadratic_weighted_kappa, top_k_accuracy, calculate_metrics, accumulate_metrics, compute_final_metrics

from src.model.HWAUNETR_class import HWAUNETR
from src.model.SwinUNETR import MultiTaskSwinUNETR
from monai.networks.nets import SwinUNETR
from visualization import visualize_for_all



def load_model(model, accelerator, checkpoint):
    try:
        check_path = f"{os.getcwd()}/model_store/{checkpoint}/best/"
        accelerator.print('load model from %s' % check_path)
        checkpoint = load_model_dict(check_path + 'pytorch_model.bin', )
        model.load_state_dict(checkpoint)
        accelerator.print(f'Load checkpoint model successfully!')
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'Failed to load checkpoint model!')
    return model


@torch.no_grad()
def val_one_epoch(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
                  metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                  post_trans: monai.transforms.Compose, accelerator: Accelerator):
    # 验证
    model.eval()
    for i, image_batch in enumerate(val_loader):
        logits = model(image_batch['image'])  
        log = ''
        total_loss = 0
        
        logits_loss = logits
        labels_loss = image_batch['class_label']
        
        for name in loss_functions:
            loss = loss_functions[name](logits_loss, labels_loss.float())
            log += f'{name}: {float(loss):1.5f} ; '
            total_loss += loss
        
        log += f'Total Loss: {float(total_loss):1.5f}'
        
        for metric_name in metrics:
            y_pred = post_trans(logits)
            y = labels_loss
            if metric_name =='miou_metric':
                y_pred = y_pred.unsqueeze(2)
                y      =      y.unsqueeze(2)
            metrics[metric_name](y_pred=y_pred, y=y)

        accelerator.print(
            f'[{i + 1}/{len(val_loader)}] {log} ',
            flush=True)
    metric = {}
    
    for metric_name in metrics:
        # for channel in range(channels):
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        # give every single task metric
        metrics[metric_name].reset()
        # task_num = channel + 1
        metric.update({
                f'{metric_name}': float(batch_acc.mean()),
            })
    accelerator.print(metric)
    return 

@torch.no_grad()
def compute_dl_score_for_example(model, config, examples):
    def compute_for_sinlge_example(example):
        pass
    train_example, val_example, test_example = examples
    load_transform, _, _ = get_transforms(config)

    
    


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.checkpoint + str(datetime.now()).replace(' ','_').replace('-','_').replace(':','_').replace('.','_')
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))
    

    accelerator.print('load model...')
    model = HWAUNETR(in_chans=len(config.loader.checkModels), fussion = [1,2,4,8], kernel_sizes=[4, 2, 2, 2], depths=[1, 1, 1, 1], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8],
                out_indices=[0, 1, 2, 3])
    model = load_model(model, accelerator, config.finetune.checkpoint)


    accelerator.print('load dataset...')
    train_loader, val_loader, test_loader, example = get_dataloader(config)
    
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'bce_loss':  nn.BCEWithLogitsLoss().to(accelerator.device),
    }
    
    metrics = {
        'accuracy': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name="accuracy"),
        'f1': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name='f1 score'),
        'specificity': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name="specificity"),
        'recall': monai.metrics.ConfusionMatrixMetric(include_background=False, metric_name="recall"),
        'miou_metric':monai.metrics.MeanIoU(include_background=False),
    }

    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])
    
    model, train_loader, val_loader, test_loader = accelerator.prepare(model, train_loader, val_loader, test_loader)

    # start valing
    accelerator.print("Start Valing! ")
    val_one_epoch(model, test_loader, metrics, post_trans, accelerator)
    

