import os

import test

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import csv
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
import pandas as pd
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader_GCNC as get_dataloader
from src.loader import get_GCNC_transforms as get_transforms
from src.loader import read_csv_for_GCNC
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import (
    Logger,
    write_example,
    resume_train_state,
    split_metrics,
    load_model_dict,
)
from src.eval import (
    calculate_f1_score,
    specificity,
    quadratic_weighted_kappa,
    top_k_accuracy,
    calculate_metrics,
    accumulate_metrics,
    compute_final_metrics,
)


from get_model import get_model


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
def val_one_epoch(
    model: torch.nn.Module,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    config: EasyDict,
    test: bool = False,
):
    # 验证
    model.eval()
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    for i, image_batch in enumerate(val_loader):
        if config.trainer.choose_model == "HSL_Net":
            _, logits = model(image_batch["image"])
        else:
            logits = model(image_batch["image"])
        # logits = inference(image_batch["image"], model)
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
        accelerator.print(
            f"[{i + 1}/{len(val_loader)}] Validation Loading...", flush=True
        )

        step += 1
    metric = {}
    if test:
        flag = "Test"
    else:
        flag = "Val"
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)

        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metrics[metric_name].reset()
        if metric_name == "dice_metric":
            metric.update(
                {
                    f"{flag}/mean {metric_name}": float(batch_acc.mean()),
                    # f"{flag}/Object1 {metric_name}": float(batch_acc[0]),
                    # f"{flag}/Object2 {metric_name}": float(batch_acc[1]),
                    # f"{flag}/Object3 {metric_name}": float(batch_acc[2]),
                }
            )
            dice_acc = torch.Tensor([metric[f"{flag}/mean dice_metric"]]).to(
                accelerator.device
            )
            dice_class = batch_acc
        else:
            metric.update(
                {
                    f"{flag}/mean {metric_name}": float(batch_acc.mean()),
                    # f"{flag}/Object1 {metric_name}": float(batch_acc[0]),
                    # f"{flag}/Object2 {metric_name}": float(batch_acc[1]),
                    # f"{flag}/Object3 {metric_name}": float(batch_acc[2]),
                }
            )
            hd95_acc = torch.Tensor([metric[f"{flag}/mean hd95_metric"]]).to(
                accelerator.device
            )
            hd95_class = batch_acc
    # accelerator.log(metric, step=epoch)
    accelerator.print(
            f"dice acc: {dice_acc} dice_class: {dice_class} hd95_acc: {hd95_acc} hd95_class: {hd95_class}"
        )


def compute_dice_from_probs(
    probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
):
    """
    直接计算Dice系数（不依赖MONAI Metric）。

    参数:
        probs (torch.Tensor): 模型输出概率, 形状 [B, 1, H, W, (D)]
        labels (torch.Tensor): 真实标签, 形状相同
        threshold (float): 将概率转为二值mask的阈值 (默认0.5)
        eps (float): 防止除零的微小常数

    返回:
        dice_per_sample (list): 每个样本的Dice系数 (长度=B)
        mean_dice (float): 平均Dice值
    """
    # 确保形状一致
    assert (
        probs.shape == labels.shape
    ), f"Shape mismatch: {probs.shape} vs {labels.shape}"

    # 二值化预测
    # preds = (probs > threshold).float()

    # 展平为向量方便计算
    preds_flat = probs.contiguous().view(probs.shape[0], -1)
    labels_flat = labels.contiguous().view(labels.shape[0], -1)

    # 计算交集和并集
    intersection = (preds_flat * labels_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + labels_flat.sum(dim=1)

    dice_per_sample = (2.0 * intersection + eps) / (union + eps)
    mean_dice = dice_per_sample.mean().item()

    return dice_per_sample.tolist(), mean_dice


@torch.no_grad()
def compute_seg_dice_for_example(model, config, post_trans, examples):
    """
    对每个样本执行分割推理，计算Dice分数并保存结果到CSV。
    """
    from monai.metrics import DiceMetric
    from monai.transforms import EnsureType
    from monai.data import decollate_batch

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    ensure_type = EnsureType()

    def compute_for_single_example(example_ids):
        results = []

        # 加载transform，与compute_dl_score_for_example一致
        load_transform, _, transforms = get_transforms(config)

        for e in example_ids:
            img_dir = os.path.join(config.GCNC_loader.root, "ALL", e)
            accelerator.print(f"Processing segmentation for: {img_dir}")

            # 1️⃣ 加载多模态图像与标签
            images, labels = [], []

            models = os.listdir(os.path.join(img_dir))

            for i, mod in enumerate(config.GCNC_loader.checkModels):
                if mod not in models:
                    if mod == "T1+C":
                        mod = "T1WI+C"
                        if mod not in models:
                            mod = "CT1"
                        if mod not in models:
                            mod = "T1+c"
                    else:
                        mod = mod + "WI"

                image_path = os.path.join(img_dir, mod, f"{e}.nii.gz")
                label_path = os.path.join(img_dir, mod, f"{e}seg.nii.gz")
                data = load_transform[i]({"image": image_path, "label": label_path})

                images.append(data["image"])
                labels.append(data["label"])

            # images.append(data["image"].unsqueeze(1))
            # labels.append(data["label"].unsqueeze(1))

            image_tensor = torch.cat(images, dim=0).to(accelerator.device)
            label_tensor = torch.cat(labels, dim=0).to(accelerator.device)
            result = {"image": image_tensor, "label": label_tensor}
            result = transforms(result)

            image_tensor = result["image"].unsqueeze_(0)
            label_tensor  = result["label"].unsqueeze_(0)

            # 2️⃣ 模型推理输出预测mask
            logits = model(image_tensor)
            probs = post_trans(logits)
            # preds = (probs > 0.5).float()

            # # 3️⃣ 计算Dice分数
            # dice_metric(y_pred=preds, y=label_tensor)
            # dice_value = dice_metric.aggregate().item()
            # dice_metric.reset()
            _, dice_value = compute_dice_from_probs(probs, label_tensor)

            results.append([e, dice_value])
            accelerator.print(f"Dice for {e}: {dice_value:.4f}")

        return results

    # 按数据集分别计算
    train_ex, val_ex, test_ex = examples
    train_res = compute_for_single_example(train_ex)
    val_res = compute_for_single_example(val_ex)
    test_res = compute_for_single_example(test_ex)

    # 4️⃣ 写入CSV
    save_dir = config.valer.dl_score_csv_path
    os.makedirs(save_dir, exist_ok=True)

    for name, data in zip(
        ["train_dice.csv", "val_dice.csv", "test_dice.csv"],
        [train_res, val_res, test_res],
    ):
        csv_path = os.path.join(save_dir, name)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Sample_ID", "Dice_Score"])
            writer.writerows(data)
        accelerator.print(f"Saved dice results to {csv_path}")

    return train_res, val_res, test_res


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)

    if config.finetune.GCNC.checkpoint != "None":
        checkpoint_name = config.finetune.GCNC.checkpoint
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

    accelerator.print("load model...")
    # model = HWAUNETR(in_chans=len(config.GCM_loader.checkModels), fussion = [1,2,4,8], kernel_sizes=[4, 2, 2, 2], depths=[1, 1, 1, 1], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8],
    #             out_indices=[0, 1, 2, 3])
    # model = ResNet(in_channels=len(config.GCM_loader.checkModels), pretrained=False)
    model = get_model(config)

    model = load_model(model, accelerator, checkpoint_name)

    accelerator.print("load dataset...")
    train_loader, val_loader, test_loader, example = get_dataloader(config)

    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "bce_loss": nn.BCEWithLogitsLoss().to(accelerator.device),
    }

    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=True,
        ),
        "hd95_metric": monai.metrics.HausdorffDistanceMetric(
            percentile=95,
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=True,
        ),
    }

    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    model, train_loader, val_loader, test_loader = accelerator.prepare(
        model, train_loader, val_loader, test_loader
    )
    
    inference = monai.inferers.SlidingWindowInferer(
        roi_size=config.GCNC_loader.target_size,
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )

    # start valing
    accelerator.print("Start Valing! ")
    # val_one_epoch(model,
    #         inference,
    #         val_loader,
    #         metrics,
    #         -1,
    #         post_trans,
    #         accelerator,
    #         config,
    #         test=False,)
    
    # val_one_epoch(model,
    #         inference,
    #         test_loader,
    #         metrics,
    #         -1,
    #         post_trans,
    #         accelerator,
    #         config,
    #         test=False,)
    compute_seg_dice_for_example(model, config, post_trans, example)
