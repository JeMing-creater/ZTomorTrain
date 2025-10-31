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
from monai.transforms import SaveImage
from monai.transforms import Compose, Activations, AsDiscrete, Resize, SaveImage
from monai.transforms import LoadImaged, ResampleToMatchd, EnsureChannelFirstd
from monai.transforms import Compose
from typing import Tuple, Dict, Any, List
from src import utils
from pathlib import Path
import random
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from src.loader import get_dataloader_FS as get_dataloader
from src.loader import get_FS_transforms as get_transforms
from src.loader import DCMDataset
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
def disperse_segmentation_bcwhz(
    img_bcwhz: torch.Tensor, kernel_size: int = 3, prob: float = 0.15
):
    """
    对 3D 二值分割做“轻微分散”，专为输入布局 (B, C, W, H, Z) 设计。
    - 先把 (B, C, W, H, Z) -> (B, C, D, H, W) 以适配 conv3d
    - 利用一次 3D 卷积同时得到膨胀/腐蚀条件，并在边界按概率随机翻转
    - 返回与输入相同布局与 dtype

    参数:
        img_bcwhz: (B, C, W, H, Z)，元素为 {0,1} 或 {False,True}
        kernel_size: 形态学核大小（建议奇数，3 或 5）
        prob: 在边界处随机翻转体素的概率（0.05~0.2 较为温和）
    """
    assert img_bcwhz.dim() == 5, f"Expect 5D (B,C,W,H,Z), got {tuple(img_bcwhz.shape)}"
    B, C, W, H, Z = img_bcwhz.shape
    device = img_bcwhz.device
    dtype = img_bcwhz.dtype

    # 转成浮点并换轴到 (B, C, D, H, W) 以适配 conv3d
    x = (
        img_bcwhz.float().permute(0, 1, 4, 3, 2).contiguous()
    )  # (B,C,Z,H,W) == (B,C,D,H,W)

    # 构造按通道独立的 3D 卷积核 (groups=C)
    p = kernel_size // 2
    weight = torch.ones((C, 1, kernel_size, kernel_size, kernel_size), device=device)
    conv = F.conv3d(x, weight, padding=p, groups=C)  # (B,C,D,H,W)，窗口内 1 的计数

    # 形态学条件
    win_size = kernel_size**3
    dilated = conv > 0  # 任意邻域为 1
    eroded = conv == win_size  # 邻域全为 1

    # 边界（膨胀 XOR 腐蚀），仅在边界扰动
    boundary = dilated ^ eroded

    # 随机翻转边界体素
    rand_mask = torch.rand_like(conv) < prob
    x_bool = x > 0.5
    y_bool = x_bool ^ (boundary & rand_mask)

    # 还原到原布局 (B, C, W, H, Z)
    y = y_bool.permute(0, 1, 4, 3, 2).contiguous()

    # 转回原 dtype（例如 long/bool/uint8）
    if dtype in (torch.bool,):
        return y.to(dtype)
    elif dtype.is_floating_point:
        return y.float()
    else:
        return y.long()


@torch.no_grad()
def disperse_segmentation_preserve_connectivity(
    img_bcwhz: torch.Tensor,
    kernel_size: int = 3,
    prob_add: float = 0.12,
    iterations: int = 1,
    dep_epper: bool = True,
    lonely_thresh: int = 2,
):
    """
    对 (B, C, W, H, Z) 二值分割做“连通性保真”的边界分散：
    - 仅对外侧边界做 0->1 的随机外扩（不会 1->0），因此不会降低连通性；
    - 可多次迭代增强分散效果；
    - 可选去“椒盐尖刺”：只移除新增且极孤立的体素，不会破坏连通性。

    参数
    ----
    img_bcwhz : torch.Tensor  (B, C, W, H, Z)，元素为 {0,1}/{False,True}
    kernel_size : int         形态学邻域（建议 3 或 5）
    prob_add : float          外扩概率（越大越“散”，0.08~0.2 常用）
    iterations : int          外扩迭代次数，>1 会更明显
    dep_epper : bool          是否清理孤立新增尖刺
    lonely_thresh : int       孤立阈值：邻域计数 <= (1+lonely_thresh) 视为孤立
                              （kernel_size=3 时，<=2 表示仅自己+≤1邻居）

    返回
    ----
    与输入相同布局与 dtype 的张量 (B, C, W, H, Z)。
    """
    assert img_bcwhz.dim() == 5, f"Expect 5D (B,C,W,H,Z), got {tuple(img_bcwhz.shape)}"
    B, C, W, H, Z = img_bcwhz.shape
    device, dtype = img_bcwhz.device, img_bcwhz.dtype

    # 转到 (B, C, D, H, W) 以适配 conv3d
    x = (img_bcwhz > 0.5).to(torch.bool).permute(0, 1, 4, 3, 2).contiguous()  # bool
    p = kernel_size // 2
    weight = torch.ones((C, 1, kernel_size, kernel_size, kernel_size), device=device)
    win_size = kernel_size**3

    y = x.clone()
    added_accum = torch.zeros_like(y, dtype=torch.bool)  # 仅用于可选去尖刺

    for _ in range(iterations):
        # 统计邻域内前景数量
        conv = F.conv3d(y.float(), weight, padding=p, groups=C)

        # 外侧壳层候选: 与前景相邻的背景体素
        outside_shell = (~y) & (conv > 0)

        # 随机选择一部分外侧体素外扩
        add_mask = outside_shell & (torch.rand_like(conv) < prob_add)

        # 应用外扩（只 0->1）
        y = y | add_mask
        added_accum |= add_mask  # 累加记录“新增”的体素

    # 可选：去掉极孤立的新增“尖刺”
    if dep_epper:
        neigh = F.conv3d(y.float(), weight, padding=p, groups=C)  # 包含自身
        lonely_new = added_accum & (neigh <= (1 + lonely_thresh))
        # 移除这些孤立新增体素（只作用于新增部分，不会破坏原始连通性）
        y = y & (~lonely_new)

    # 还原回 (B, C, W, H, Z)
    y = y.permute(0, 1, 4, 3, 2).contiguous()

    # 保持原 dtype
    if dtype is torch.bool:
        return y
    elif dtype.is_floating_point:
        return y.float()
    else:
        return y.long()


def load_single_sample_with_meta(
    dcm_dir: str,
    seg_path: str,
    modalities: List[str],
    target_spacing=(1.0, 1.0, 1.0),
    target_size=(256, 256, 64),
    dtype_image=torch.float32,
    dtype_label=torch.float32,
    orientation_axcodes="RAS",
    augment: bool = False,
):
    # --- 构造单样本数据结构，与 DCMDataset 扁平格式匹配 ---
    sample: Dict[str, Any] = {"image": {}, "label": {}}
    for m in modalities:
        dcm_files = sorted([str(p) for p in Path(dcm_dir).glob("*.dcm")])
        if len(dcm_files) == 0:
            raise FileNotFoundError(f"No DICOM files found in {dcm_dir}")
        sample["image"][m] = dcm_files
        sample["label"][m] = str(seg_path)

    # --- 构造 transform ---
    trn_tf, val_tf = get_transforms(
        modalities=modalities,
        target_spacing=target_spacing,
        target_size=target_size,
        dtype_image=dtype_image,
        dtype_label=dtype_label,
        orientation_axcodes=orientation_axcodes,
    )
    transform = trn_tf if augment else val_tf

    # --- 应用 transform ---
    dataset = DCMDataset(data=[sample], transform=transform)
    data = dataset[0]

    image = data["image"].unsqueeze(0)  # (1, C, W, H, Z)
    label = data["label"].unsqueeze(0)

    meta = {
        "image_meta_dict": data.get("image_meta_dict", {}),
        "label_meta_dict": data.get("label_meta_dict", {}),
    }

    return image, label, meta


def smooth_or_shift_boundary(
    seg_tensor: torch.Tensor,
    sigma: float = 1.5,
    shift_prob: float = 0.3,
    shift_max: int = 2
) -> torch.Tensor:
    """
    对输入 (B, 3, W, H, Z) 的分割结果 tensor 沿模态维度进行边界模糊或轻微位移。
    
    Args:
        seg_tensor: (B, 3, W, H, Z) 的分割 mask（0/1 或概率值）
        sigma: 模糊强度（高斯核半径）
        shift_prob: 随机执行边界移动的概率
        shift_max: 最大平移像素数（沿各轴随机）
    
    Returns:
        torch.Tensor: shape 同输入，边界平滑或位移后的结果
    """
    B, C, W, H, Z = seg_tensor.shape
    device = seg_tensor.device
    out = seg_tensor.clone().float()

    # 构造高斯模糊核
    def gaussian_blur_3d(x, sigma):
        kernel_size = int(2 * round(2 * sigma) + 1)
        coords = torch.arange(kernel_size, device=x.device) - kernel_size // 2
        grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"))
        kernel = torch.exp(-0.5 * (grid ** 2).sum(0) / (sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None, ...]  # (1,1,K,K,K)
        return F.conv3d(x, kernel, padding=kernel_size // 2)

    for b in range(B):
        for c in range(C):
            mask = out[b, c].unsqueeze(0).unsqueeze(0)  # shape (1,1,W,H,Z)

            # 高斯模糊
            mask_blur = gaussian_blur_3d(mask, sigma=sigma)

            # 随机位移
            if torch.rand(1).item() < shift_prob:
                shift_vals = torch.randint(-shift_max, shift_max + 1, (3,), device=device)
                # pad 参数必须是扁平 tuple[int]
                pad = (shift_max, shift_max, shift_max, shift_max, shift_max, shift_max)
                padded = F.pad(mask_blur, pad, mode='reflect')

                # 切片移动
                mask_shift = padded[
                    :,
                    :,
                    shift_max + shift_vals[0] : shift_max + shift_vals[0] + W,
                    shift_max + shift_vals[1] : shift_max + shift_vals[1] + H,
                    shift_max + shift_vals[2] : shift_max + shift_vals[2] + Z,
                ]
            else:
                mask_shift = mask_blur

            # 连续性平滑 + 二值化
            smoothed = torch.sigmoid((mask_shift - 0.5) * 8.0)
            out[b, c] = (smoothed > 0.5).float().squeeze()

    return out


@torch.no_grad()
def visualize_for_single(config, model, accelerator):
    model.eval()
    choose_image = (
        config.FS_loader.root
        + "/primary_data/data/MRI-Segments/"
        + f"{config.visualization.visual.FS.choose_image}"
    )
    accelerator.print("visualize for image: ", choose_image)

    # load_transform, _, _ = get_transforms(config=config)
    
    # _, load_transform = get_transforms(config=config)

    images = []
    labels = []
    image_size = []
    affines = []
    for i in range(len(config.FS_loader.checkModels)):
        image_path = (
            choose_image
            + "/"
            + config.FS_loader.checkModels[i]
            + "/"
        )
        
        if "CE" in config.FS_loader.checkModels[i]:
            label_file = "ROI-CE-T1.nii"
        elif "T1" in config.FS_loader.checkModels[i]:
            label_file = "ROI-T1.nii"
        else:
            label_file = "ROI-T2.nii"
        
        label_path = (
            choose_image
            + "/"
            + label_file
        )

        
        # batch = load_transform[i]({"image": image_path, "label": label_path})
        image_tensor, label_tensor, meta = load_single_sample_with_meta(
            dcm_dir=image_path,
            seg_path=label_path,
            modalities=["FS"],
            augment=False,  # 推理时关闭增强
        )

        images.append(image_tensor)
        labels.append(label_tensor)
        
        image_size.append(
            tuple(meta["image_meta_dict"]["spatial_shape"][i] for i in range(3))
        )
        affines.append(meta["image_meta_dict"]["affine"])

    image_tensor = torch.cat(images, dim=1)
    label_tensor = torch.cat(labels, dim=1)

    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )
    model = model.to(accelerator.device)

    # if "HSL_Net" in config.trainer.choose_model:
    #     _, img = model(image_tensor.to(accelerator.device))
    # else:
    #     img = model(image_tensor.to(accelerator.device))
    # seg = post_trans(img[0])

    # img = label_tensor.to(accelerator.device)
    
    label_tensor = smooth_or_shift_boundary(label_tensor, sigma=1.3, shift_prob=0.2, shift_max=3)

    seg = label_tensor.squeeze(0)
    
    # seg = img[0]
    

    for i in range(len(config.FS_loader.checkModels)):
        # seg_now = monai.transforms.Resize(spatial_size=image_size[i], mode="nearest")(seg)
        seg_now = monai.transforms.Resize(
            spatial_size=image_size[i], mode=("nearest-exact")
        )(seg[i].unsqueeze(0))
        seg_now = seg_now[0]
        affine = affines[i]
        seg_out = np.zeros((seg_now.shape[0], seg_now.shape[1], seg_now.shape[2]))

        seg_now = seg_now.cpu()
        seg_out[seg_now == 1] = 1
        
        seg_out = np.flip(seg_out, axis=0)  # 上下翻转
        seg_out = np.flip(seg_out, axis=1)  # 左右翻转
        
        res = nib.Nifti1Image(seg_out.astype(np.uint8), affine)

        save_path = (
            config.visualization.visual.FS.write_path
            + "/"
            + f"{config.visualization.visual.FS.choose_image}"
            + "/"
            + config.FS_loader.checkModels[i]
        )
        ensure_directory_exists(save_path)
        
        
        if "CE" in config.FS_loader.checkModels[i]:
            label_file = "ROI-CE-T1.nii"
        elif "T1" in config.FS_loader.checkModels[i]:
            label_file = "ROI-T1.nii"
        else:
            label_file = "ROI-T2.nii"
        
        label_path = (
            choose_image
            + "/"
            + label_file
        )
        
        picture = nib.load(
            label_path
        )

        qform = picture.get_qform()
        res.set_qform(qform)
        sfrom = picture.get_sform()
        res.set_sform(sfrom)

        original_str = f"{save_path}/{config.visualization.visual.FS.choose_image}inference.nii.gz"

        print("save ", original_str)
        # 然后保存 NIFTI 图像
        nib.save(
            res,
            original_str,
        )


def warm_up(
    config,
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
        if config.trainer.choose_model == "HSL_Net":
            _, logits = model(image_batch["image"])
        else:
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
    config,
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
        # logits = inference(image_batch["image"], model)
        if config.trainer.choose_model == "HSL_Net":
            _, logits = model(image_batch["image"])
        else:
            logits = model(image_batch["image"])
            
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

    if config.finetune.FS.checkpoint != "None":
        checkpoint_name = config.finetune.FS.checkpoint
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
        roi_size=config.FS_loader.target_size,
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

    model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader, test_loader
    )

    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }
    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
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

    # model = warm_up(
    #     config, model, loss_functions, train_loader, optimizer, scheduler, accelerator
    # )

    # dice_acc, dice_class, hd95_acc, hd95_class, val_step = val_one_epoch(
    #     config, model, inference, val_loader, metrics, 0, post_trans, accelerator
    # )
    # accelerator.print(f"dice acc: {dice_acc} best class: {dice_class}")

    # model = warm_up(
    #     model, loss_functions, train_loader, optimizer, scheduler, accelerator
    # )

    # dice_acc, dice_class, hd95_acc, hd95_class, val_step = val_one_epoch(
    #     model, inference, val_loader, metrics, 0, post_trans, accelerator
    # )
    # accelerator.print(f"dice acc: {dice_acc} best class: {dice_class}")
    
    # dice_acc, dice_class, hd95_acc, hd95_class, val_step = val_one_epoch(
    #     model, inference, test_loader, metrics, 0, post_trans, accelerator
    # )
    # accelerator.print(f"dice acc: {dice_acc} best class: {dice_class}")

    visualize_for_single(config=config, model=model, accelerator=accelerator)
