import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader_GCM as get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, write_example, resume_train_state, split_metrics, reload_pre_train_model
from src.eval import (
    calculate_f1_score,
    specificity,
    quadratic_weighted_kappa,
    top_k_accuracy,
    calculate_metrics,
    accumulate_metrics,
    compute_final_metrics,
)

# from src.model.HWAUNETR_class import HWAUNETR
# from src.model.SwinUNETR import MultiTaskSwinUNETR
# from monai.networks.nets import SwinUNETR
# from src.model.ResNet import ResNet3DClassifier as ResNet
# from src.model.Vit import ViT3D
from get_model import get_model


def train_one_epoch(
    model: torch.nn.Module,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    # 训练
    model.train()
    accelerator.print(f"Training...", flush=True)
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    # for i, image_batch in enumerate(train_loader):
    for i, image_batch in loop:
        # for i, image_batch in enumerate(train_loader):
        logits = model(image_batch["image"])
        total_loss = 0
        logits_loss = logits
        labels = image_batch["class_label"]
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits_loss, labels.float())
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss

        for metric_name in metrics:
            y_pred = post_trans(logits)
            y = labels
            if metric_name == "miou_metric":
                y_pred = y_pred.unsqueeze(2)
                y = y.unsqueeze(2)
            metrics[metric_name](y_pred=y_pred, y=y)

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=step,
        )
        # accelerator.print(
        #     f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
        #     flush=True
        #     )
        # 更新信息
        loop.set_description(f"Epoch [{epoch+1}/{config.trainer.num_epochs}]")
        loop.set_postfix(loss=total_loss)
        step += 1
    scheduler.step(epoch)

    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)

        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        # give every single task metric
        metrics[metric_name].reset()
        metric.update(
            {
                f"Train/{metric_name}": float(batch_acc.mean()),
            }
        )
    for metric_name in metrics:
        all_data = []
        for key in metric.keys():
            if metric_name in key:
                all_data.append(metric[key])
        me_data = sum(all_data) / len(all_data)
        metric.update({f"Train/{metric_name}": float(me_data)})

    accelerator.log(metric, step=epoch)
    return metric, step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    test: bool = False,
):
    # 验证
    model.eval()
    if test:
        flag = "Test"
        accelerator.print(f"Testing...", flush=True)
    else:
        flag = "Val"
        accelerator.print(f"Valing...", flush=True)
    loop = tqdm(enumerate(val_loader), total=len(val_loader))
    # for i, image_batch in enumerate(val_loader):
    for i, image_batch in loop:
        # logits = inference(model, image_batch['image'])
        logits = model(
            image_batch["image"]
        )  # some moedls can not accepted inference, I do not know why.
        log = ""
        total_loss = 0

        logits_loss = logits
        labels_loss = image_batch["class_label"]

        for name in loss_functions:
            loss = loss_functions[name](logits_loss, labels_loss.float())
            accelerator.log({f"{flag}/" + name: float(loss)}, step=step)
            log += f"{name} {float(loss):1.5f} "
            total_loss += loss

        accelerator.log(
            {
                f"{flag}/Total Loss": float(total_loss),
            },
            step=step,
        )

        for metric_name in metrics:
            y_pred = post_trans(logits)
            y = labels_loss
            if metric_name == "miou_metric":
                y_pred = y_pred.unsqueeze(2)
                y = y.unsqueeze(2)
            metrics[metric_name](y_pred=y_pred, y=y)

        # accelerator.print(
        #     f'[{i + 1}/{len(val_loader)}] {flag} Validation Loading...',
        #     flush=True)
        loop.set_description(f"Epoch [{epoch+1}/{config.trainer.num_epochs}]")
        loop.set_postfix(loss=total_loss)
        step += 1
    metric = {}

    for metric_name in metrics:
        # for channel in range(channels):
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)

        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        # give every single task metric
        metrics[metric_name].reset()
        # task_num = channel + 1
        metric.update(
            {
                f"{flag}/{metric_name}": float(batch_acc.mean()),
            }
        )

    accelerator.log(metric, step=epoch)
    return metric, step


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)
    
    if config.finetune.GCM.checkpoint != 'None':
        checkpoint_name = config.finetune.GCM.checkpoint
    else:
        checkpoint_name = config.trainer.choose_dataset + "_" + config.trainer.task + config.trainer.choose_model
    
    logging_dir = (
        os.getcwd()
        + "/logs/"
        + checkpoint_name
        + str(datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace(".", "_")
    )
    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")
    model = get_model(config)
    
    reload_pre_train_model(model, accelerator, "GCM_SegmentationTFM_UNET_seg")