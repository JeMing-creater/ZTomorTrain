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
import matplotlib.patches as patches

from src import utils
from src.loader import get_dataloader_GCNC as get_dataloader
from src.loader import get_GCNC_transforms as get_transforms
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


class LayerActivations:
    features = None

    def __init__(self, model):
        self.hook = model.register_forward_hook(self.hook_fn)
        # 获取model.features中某一层的output

    def hook_fn(self, module, MRI_tensorut, output):
        self.features = output.cpu()

    def remove(self):  ## remove hook
        self.hook.remove()


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


def get_target_layer(model, target_layer=None):
    if target_layer is None:
        conv_list = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                conv_list.append((name, module))
            if isinstance(module, torch.nn.AdaptiveAvgPool3d):
                break  # 到池化为止
        if not conv_list:
            raise ValueError("未找到 Conv3d 层")
        target_layer = conv_list[-1][1]
        print(f"[Grad-CAM] Using target layer: {conv_list[-1][0]}")
    return target_layer


def get_last_conv3d_for_model(model):
    """
    自动找到模型中的最后一个 Conv3d 层，并返回一个可以直接用于
    model.<layer> 访问的变量（即 model.具体层名）。
    """
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            last_name = name  # 记录路径

    if last_name is None:
        raise ValueError("模型中未找到任何 Conv3d 层！")

    # 通过路径获取真正的 model.<path> 变量
    target = model
    for attr in last_name.split("."):
        target = getattr(target, attr)

    print(f"[LayerActivations] 已找到最后一个 Conv3d 层: model.{last_name}")
    return target


def get_conv3d_for_model(model):
    """
    返回模型中的倒数第三个 Conv3d 层，作为 model.<path> 对象，方便直接用 LayerActivations。
    如果模型中 Conv3d 少于 3 个，则返回最后一个。
    """
    conv_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            conv_names.append(name)

    if not conv_names:
        raise ValueError("模型中未找到任何 Conv3d 层！")

    # 选择倒数第三个，如果不足则取最后一个
    target_name = conv_names[-3] if len(conv_names) >= 3 else conv_names[-1]

    # 根据路径找到真正的 model.<path> 对象
    target = model
    for attr in target_name.split("."):
        target = getattr(target, attr)

    print(f"[LayerActivations] 已找到倒数第三个 Conv3d 层: model.{target_name}")
    return target


def write_heatmap(config, model, accelerator):
    model.eval()
    load_transform, _, _ = get_transforms(config)

    choose_image = (
        config.GCNC_loader.root
        + "/ALL/"
        + f"{config.visualization.heatmap.GCNC.choose_image}"
    )
    accelerator.print("valing heatmap for image: ", choose_image)

    images, labels, image_size, affines, MRI_array_list = [], [], [], [], []
    for i in range(len(config.GCNC_loader.checkModels)):
        image_path = (
            choose_image
            + "/"
            + config.GCNC_loader.checkModels[i]
            + "/"
            + f"{config.visualization.heatmap.GCNC.choose_image}.nii.gz"
        )
        label_path = (
            choose_image
            + "/"
            + config.GCNC_loader.checkModels[i]
            + "/"
            + f"{config.visualization.heatmap.GCNC.choose_image}seg.nii.gz"
        )

        MRI = nibabel.load(image_path)
        MRI_array = MRI.get_fdata().astype("float32")
        MRI_array_list.append(MRI_array)

        batch = load_transform[i]({"image": image_path, "label": label_path})
        images.append(batch["image"].unsqueeze(1))
        labels.append(batch["label"].unsqueeze(1))
        image_size.append(
            tuple(batch["image_meta_dict"]["spatial_shape"][i].item() for i in range(3))
        )
        affines.append(batch["label_meta_dict"]["affine"])

    input_data = torch.cat(images, dim=1).to(accelerator.device)
    # conv_out = LayerActivations(get_conv3d_for_model(model))
    conv_out = LayerActivations(model.encoder.hidden_downsample)

    _ = model(input_data)
    cam = conv_out.features
    conv_out.remove  # 取消 hook

    cam = cam.cpu().detach().numpy().squeeze()
    cam = cam[1]  # 取第 1 个通道的激活图

    count = 1
    for MRI_array in MRI_array_list:
        capi = resize(cam, MRI_array.shape, order=1, mode="reflect", anti_aliasing=True)
        heatmap = (capi - capi.min()) / (capi.max() - capi.min() + 1e-8)

        # 中间切片
        axial_idx = MRI_array.shape[-1] // 2
        coronal_idx = MRI_array.shape[-2] // 2
        sagittal_idx = MRI_array.shape[-3] // 2

        sagittal_img = MRI_array[sagittal_idx, :, :]
        axial_img = MRI_array[:, :, axial_idx]
        coronal_img = MRI_array[:, coronal_idx, :]

        sagittal_cam = heatmap[sagittal_idx, :, :]
        axial_cam = heatmap[:, :, axial_idx]
        coronal_cam = heatmap[:, coronal_idx, :]

        # --- 统一三视图尺寸 ---
        target_h = max(sagittal_img.shape[0], axial_img.shape[0], coronal_img.shape[0])
        target_w = max(sagittal_img.shape[1], axial_img.shape[1], coronal_img.shape[1])

        def resize_to_target(img):
            return resize(
                img, (target_h, target_w), order=1, mode="reflect", anti_aliasing=True
            )

        sagittal_img, sagittal_cam = resize_to_target(sagittal_img), resize_to_target(
            sagittal_cam
        )
        axial_img, axial_cam = resize_to_target(axial_img), resize_to_target(axial_cam)
        coronal_img, coronal_cam = resize_to_target(coronal_img), resize_to_target(
            coronal_cam
        )
        # ------------------------

        f, axarr = plt.subplots(3, 3, figsize=(12, 12))
        f.suptitle("CAM_3D_medical_image", fontsize=30)

        # Sagittal
        axarr[0, 0].imshow(np.rot90(sagittal_img, 1), cmap="gray")
        axarr[0, 0].axis("off")
        axarr[0, 0].set_title("Sagittal MRI", fontsize=25)
        axarr[0, 1].imshow(np.rot90(sagittal_cam, 1), cmap="jet")
        axarr[0, 1].axis("off")
        axarr[0, 1].set_title("Weight-CAM", fontsize=25)
        sagittal_overlay = cv2.addWeighted(sagittal_img, 0.3, sagittal_cam, 0.6, 0)
        axarr[0, 2].imshow(np.rot90(sagittal_overlay, 1), cmap="jet")
        axarr[0, 2].axis("off")
        axarr[0, 2].set_title("Overlay", fontsize=25)

        # Axial
        axarr[1, 0].imshow(np.rot90(axial_img, 1), cmap="gray")
        axarr[1, 0].axis("off")
        axarr[1, 0].set_title("Axial MRI", fontsize=25)
        axarr[1, 1].imshow(np.rot90(axial_cam, 1), cmap="jet")
        axarr[1, 1].axis("off")
        axarr[1, 1].set_title("Weight-CAM", fontsize=25)
        axial_overlay = cv2.addWeighted(axial_img, 0.3, axial_cam, 0.6, 0)
        axarr[1, 2].imshow(np.rot90(axial_overlay, 1), cmap="jet")
        axarr[1, 2].axis("off")
        axarr[1, 2].set_title("Overlay", fontsize=25)

        # Coronal
        axarr[2, 0].imshow(np.rot90(coronal_img, 1), cmap="gray")
        axarr[2, 0].axis("off")
        axarr[2, 0].set_title("Coronal MRI", fontsize=25)
        axarr[2, 1].imshow(np.rot90(coronal_cam, 1), cmap="jet")
        axarr[2, 1].axis("off")
        axarr[2, 1].set_title("Weight-CAM", fontsize=25)
        coronal_overlay = cv2.addWeighted(coronal_img, 0.3, coronal_cam, 0.6, 0)
        axarr[2, 2].imshow(np.rot90(coronal_overlay, 1), cmap="jet")
        axarr[2, 2].axis("off")
        axarr[2, 2].set_title("Overlay", fontsize=25)

        plt.colorbar(axarr[2, 2].images[0], shrink=0.5)
        ensure_directory_exists(
            config.visualization.heatmap.GCNC.write_path
            + "/"
            + f"{config.visualization.heatmap.GCNC.choose_image}"
        )
        plt.savefig(
            f"{config.visualization.heatmap.GCNC.write_path}/{config.visualization.heatmap.GCNC.choose_image}/Heatmap_{config.GCNC_loader.checkModels[count-1]}.png"
        )
        count += 1


def write_localized_heatmap(config, model, accelerator):
    model.eval()
    load_transform, _, _ = get_transforms(config)

    choose_image = (
        config.GCNC_loader.root
        + "/ALL/"
        + f"{config.visualization.heatmap.GCNC.choose_image}"
    )
    accelerator.print("Validating heatmap for image: ", choose_image)

    images, labels, image_size, affines, MRI_array_list, label_list = [], [], [], [], [], []
    
    # Loop through each modality to process and store data
    for i in range(len(config.GCNC_loader.checkModels)):
        image_path = (
            choose_image
            + "/"
            + config.GCNC_loader.checkModels[i]
            + "/"
            + f"{config.visualization.heatmap.GCNC.choose_image}.nii.gz"
        )
        label_path = (
            choose_image
            + "/"
            + config.GCNC_loader.checkModels[i]
            + "/"
            + f"{config.visualization.heatmap.GCNC.choose_image}seg.nii.gz"
        )

        MRI = nib.load(image_path)
        MRI_array = MRI.get_fdata().astype("float32")
        MRI_array_list.append(MRI_array)

        batch = load_transform[i]({"image": image_path, "label": label_path})
        images.append(batch["image"].unsqueeze(0))
        labels.append(batch["label"].unsqueeze(0))
        image_size.append(tuple(batch["image_meta_dict"]["spatial_shape"]))
        affines.append(batch["label_meta_dict"]["affine"])

        # Load and store the segmentation label if available
        try:
            label = nib.load(label_path).get_fdata().astype("float32")
            label_list.append(label)
        except FileNotFoundError:
            label_list.append(None)

    input_data = torch.cat(images, dim=1).to(accelerator.device)
    conv_out = LayerActivations(model.encoder.hidden_downsample)
    _ = model(input_data)
    cam = conv_out.features
    conv_out.remove()  # Remove hook

    cam = cam.cpu().detach().numpy().squeeze()

    threshold = 0.5  # Threshold to focus on the most relevant regions

    count = 1
    for idx, (MRI_array, label) in enumerate(zip(MRI_array_list, label_list)):
        capi = resize(cam[idx], MRI_array.shape, order=1, mode="reflect", anti_aliasing=True)
        heatmap = (capi - capi.min()) / (capi.max() - capi.min() + 1e-8)

        axial_idx = MRI_array.shape[-1] // 2
        coronal_idx = MRI_array.shape[-2] // 2
        sagittal_idx = MRI_array.shape[-3] // 2

        sagittal_img = MRI_array[sagittal_idx, :, :]
        axial_img = MRI_array[:, :, axial_idx]
        coronal_img = MRI_array[:, coronal_idx, :]

        sagittal_cam = heatmap[sagittal_idx, :, :]
        axial_cam = heatmap[:, :, axial_idx]
        coronal_cam = heatmap[:, coronal_idx, :]

        def create_local_heatmap(img, cam, threshold, seg_mask=None):
            # Normalize the CAM to [0, 1]
            cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            mask = cam_norm > threshold
            if seg_mask is not None:
                mask = np.logical_and(mask, seg_mask)
            local_cam = cam_norm * mask
            
            # Ensure both images are of type float32 for correct blending
            img_float = img.astype(np.float32)
            local_cam_float = local_cam.astype(np.float32)
            
            # Apply Gaussian blur to smooth the heatmap
            local_cam_blurred = gaussian_filter(local_cam_float, sigma=5)  # Adjust sigma as needed
            
            # Blend the original image with the heatmap
            overlay = cv2.addWeighted(img_float, 0.7, local_cam_blurred, 0.3, 0)
            
            return local_cam, overlay, mask

        sagittal_local_cam, sagittal_overlay, sagittal_mask = create_local_heatmap(sagittal_img, sagittal_cam, threshold, label[sagittal_idx, :, :] if label is not None else None)
        axial_local_cam, axial_overlay, axial_mask = create_local_heatmap(axial_img, axial_cam, threshold, label[:, :, axial_idx] if label is not None else None)
        coronal_local_cam, coronal_overlay, coronal_mask = create_local_heatmap(coronal_img, coronal_cam, threshold, label[:, coronal_idx, :] if label is not None else None)

        f, axarr = plt.subplots(3, 3, figsize=(12, 12))
        f.suptitle(f"Localized-CAM_3D_medical_image {config.GCNC_loader.checkModels[idx]}", fontsize=30)

        # Sagittal
        axarr[0, 0].imshow(np.rot90(sagittal_img, 1), cmap="gray")
        add_bounding_box(axarr[0, 0], sagittal_mask)
        axarr[0, 0].axis("off")
        axarr[0, 0].set_title("Sagittal MRI", fontsize=25)
        axarr[0, 1].imshow(np.rot90(sagittal_local_cam, 1), cmap="jet")
        axarr[0, 1].axis("off")
        axarr[0, 1].set_title("Local Weight-CAM", fontsize=25)
        axarr[0, 2].imshow(np.rot90(sagittal_overlay, 1), cmap="jet")
        axarr[0, 2].axis("off")
        axarr[0, 2].set_title("Overlay", fontsize=25)

        # Axial
        axarr[1, 0].imshow(np.rot90(axial_img, 1), cmap="gray")
        add_bounding_box(axarr[1, 0], axial_mask)
        axarr[1, 0].axis("off")
        axarr[1, 0].set_title("Axial MRI", fontsize=25)
        axarr[1, 1].imshow(np.rot90(axial_local_cam, 1), cmap="jet")
        axarr[1, 1].axis("off")
        axarr[1, 1].set_title("Local Weight-CAM", fontsize=25)
        axarr[1, 2].imshow(np.rot90(axial_overlay, 1), cmap="jet")
        axarr[1, 2].axis("off")
        axarr[1, 2].set_title("Overlay", fontsize=25)

        # Coronal
        axarr[2, 0].imshow(np.rot90(coronal_img, 1), cmap="gray")
        add_bounding_box(axarr[2, 0], coronal_mask)
        axarr[2, 0].axis("off")
        axarr[2, 0].set_title("Coronal MRI", fontsize=25)
        axarr[2, 1].imshow(np.rot90(coronal_local_cam, 1), cmap="jet")
        axarr[2, 1].axis("off")
        axarr[2, 1].set_title("Local Weight-CAM", fontsize=25)
        axarr[2, 2].imshow(np.rot90(coronal_overlay, 1), cmap="jet")
        axarr[2, 2].axis("off")
        axarr[2, 2].set_title("Overlay", fontsize=25)

        plt.colorbar(axarr[2, 2].images[0], shrink=0.5)
        ensure_directory_exists(
            config.visualization.heatmap.GCNC.write_path
            + "/"
            + f"{config.visualization.heatmap.GCNC.choose_image}"
        )
        plt.savefig(
            f"{config.visualization.heatmap.GCNC.write_path}/{config.visualization.heatmap.GCNC.choose_image}/LocalizedHeatmap_{config.GCNC_loader.checkModels[idx]}.png"
        )
        plt.close(f)  # Close figure to free memory
        count += 1

def add_bounding_box(ax, mask):
    coords = np.where(mask)
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

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
    model = get_model(config)

    model = load_model(model, accelerator, checkpoint_name)
    model = accelerator.prepare(model)

    accelerator.print("write heatmap...")
    write_localized_heatmap(config, model, accelerator)
