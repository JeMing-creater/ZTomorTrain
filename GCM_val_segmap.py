import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import csv
import cv2
from datetime import datetime
from typing import Dict
from copy import deepcopy
import monai
import torch
import yaml
import math
import nibabel
import scipy.ndimage as ndimage
from skimage.transform import resize
import pandas as pd
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory
import torch.nn.functional as F
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from src.optimizer import LinearWarmupCosineAnnealingLR

from src import utils
from src.loader import get_dataloader_GCM as get_dataloader
from src.loader import get_GCM_transforms as get_transforms
from src.utils import (
    Logger,
    write_example,
    resume_train_state,
    split_metrics,
    load_model_dict,
    ensure_directory_exists,
    copy_file,
)
import nibabel as nib
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
from get_model import get_model

# from src.model.HWAUNETR import HWAUNETR


def load_model(model, accelerator, checkpoint):
    try:
        check_path = f"{os.getcwd()}/model_store/{checkpoint}/best/"
        accelerator.print("load model from %s" % check_path)
        checkpoint = load_model_dict(
            check_path + "pytorch_model.bin",
        )
        model.load_state_dict(checkpoint)
        accelerator.print(f"Load checkpoint model successfully!")
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Failed to load checkpoint model!")
    return model


@torch.no_grad()
def visualize_for_single(config, model, accelerator):
    model.eval()
    choose_image = (
        config.GCM_loader.root
        + "/ALL/"
        + f"{config.visualization.visual.GCM.choose_image}"
    )
    accelerator.print("visualize for image: ", choose_image)

    load_transform, _, _ = get_transforms(config=config)

    images = []
    labels = []
    image_size = []
    affines = []
    for i in range(len(config.GCM_loader.checkModels)):
        image_path = (
            choose_image
            + "/"
            + config.GCM_loader.checkModels[i]
            + "/"
            + f"{config.visualization.visual.GCM.choose_image}.nii.gz"
        )
        label_path = (
            choose_image
            + "/"
            + config.GCM_loader.checkModels[i]
            + "/"
            + f"{config.visualization.visual.GCM.choose_image}seg.nii.gz"
        )

        batch = load_transform[i]({"image": image_path, "label": label_path})

        images.append(batch["image"].unsqueeze(1))
        labels.append(batch["label"].unsqueeze(1))
        image_size.append(
            tuple(batch["image_meta_dict"]["spatial_shape"][i].item() for i in range(3))
        )
        affines.append(batch["label_meta_dict"]["affine"])

    image_tensor = torch.cat(images, dim=1)
    label_tensor = torch.cat(labels, dim=1)

    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )
    model = model.to(accelerator.device)
    _, img = model(image_tensor.to(accelerator.device))

    seg = post_trans(img[0])

    for i in range(len(config.GCM_loader.checkModels)):
        # seg_now = monai.transforms.Resize(spatial_size=image_size[i], mode="nearest")(seg)
        seg_now = monai.transforms.Resize(
            spatial_size=image_size[i], mode=("nearest-exact")
        )(seg[i].unsqueeze(0))
        seg_now = seg_now[0]
        affine = affines[i]
        seg_out = np.zeros((seg_now.shape[0], seg_now.shape[1], seg_now.shape[2]))

        seg_now = seg_now.cpu()
        seg_out[seg_now == 1] = 1
        res = nib.Nifti1Image(seg_out.astype(np.uint8), affine)

        save_path = (
            config.visualization.visual.GCM.write_path
            + "/"
            + f"{config.visualization.visual.GCM.choose_image}"
            + "/"
            + config.GCM_loader.checkModels[i]
        )
        ensure_directory_exists(save_path)
        picture = nib.load(
            choose_image
            + "/"
            + config.GCM_loader.checkModels[i]
            + "/"
            + f"{config.visualization.visual.GCM.choose_image}seg.nii.gz"
        )

        qform = picture.get_qform()
        res.set_qform(qform)
        sfrom = picture.get_sform()
        res.set_sform(sfrom)

        original_str = f"{save_path}/{config.visualization.visual.GCM.choose_image}inference.nii.gz"

        print("save ", original_str)
        # 然后保存 NIFTI 图像
        nib.save(
            res,
            original_str,
        )


def warm_up(
    model: torch.nn.Module,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
):
    # 训练
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch["image"])
        total_loss = 0
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch["label"])
            total_loss += loss
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.print(
            f"Warm up [{i + 1}/{len(train_loader)}] Warm up Loss:{total_loss}",
            flush=True,
        )
    scheduler.step(0)
    return model


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
):
    # 验证
    model.eval()
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch["image"], model)
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
        accelerator.print(
            f"[{i + 1}/{len(val_loader)}] Validation Loading...", flush=True
        )

        step += 1
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0]
        if accelerator.num_processes > 1:
            batch_acc = (
                accelerator.reduce(batch_acc.to(accelerator.device))
                / accelerator.num_processes
            )
        metrics[metric_name].reset()
        if metric_name == "dice_metric":
            metric.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc.mean()),
                    f"Val/Object1 {metric_name}": float(batch_acc[0]),
                    f"Val/Object2 {metric_name}": float(batch_acc[1]),
                }
            )
            dice_acc = torch.Tensor([metric["Val/mean dice_metric"]]).to(
                accelerator.device
            )
            dice_class = batch_acc
        else:
            metric.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc.mean()),
                    f"Val/Object1 {metric_name}": float(batch_acc[0]),
                    f"Val/Object2 {metric_name}": float(batch_acc[1]),
                }
            )
            hd95_acc = torch.Tensor([metric["Val/mean hd95_metric"]]).to(
                accelerator.device
            )
            hd95_class = batch_acc
    return dice_acc, dice_class, hd95_acc, hd95_class, step


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)

    if config.finetune.GCM.checkpoint != "None":
        checkpoint_name = config.finetune.GCM.checkpoint
    else:
        checkpoint_name = (
            config.trainer.choose_dataset
            + "_"
            + config.trainer.task
            + config.trainer.choose_model
        )

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

    model = get_model(config)

    model = load_model(model, accelerator, checkpoint_name)

    train_loader, val_loader, test_loader, example = get_dataloader(config)
    inference = monai.inferers.SlidingWindowInferer(
        roi_size=config.GCM_loader.target_size,
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )

    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=config.trainer.weight_decay,
        lr=config.trainer.lr,
        betas=(0.9, 0.95),
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # model = warm_up(model, loss_functions, train_loader,
    #         optimizer, scheduler, accelerator)

    # loss_functions = {
    #     "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
    #     "dice_loss": monai.losses.DiceLoss(
    #         smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
    #     ),
    # }
    # metrics = {
    #     "dice_metric": monai.metrics.DiceMetric(
    #         include_background=True,
    #         reduction=monai.utils.MetricReduction.MEAN_BATCH,
    #         get_not_nans=True,
    #     ),
    # }
    # post_trans = monai.transforms.Compose(
    #     [
    #         monai.transforms.Activations(sigmoid=True),
    #         monai.transforms.AsDiscrete(threshold=0.5),
    #     ]
    # )

    # dice_acc, dice_class, hd95_acc, hd95_class, val_step = val_one_epoch(
    #     model, inference, val_loader, metrics, 0, post_trans, accelerator
    # )
    # accelerator.print(f"dice acc: {dice_acc} best class: {dice_class}")

    visualize_for_single(config=config, model=model, accelerator=accelerator)
