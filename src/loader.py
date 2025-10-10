from calendar import c
import os
import math
import yaml
import torch
import monai
import random
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from datetime import datetime
from easydict import EasyDict
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep
from monai.networks.utils import one_hot

sitk.ProcessObject.SetGlobalWarningDisplay(False)
from typing import Tuple, List, Mapping, Hashable, Dict
from monai.transforms import (
    Compose,
    LoadImage,
    LoadImaged,
    EnsureTyped,
    MapTransform,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ResampleToMatchd,
    ResizeWithPadOrCropd,
    Resize,
    Resized,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd
)


class ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(
    monai.transforms.MapTransform
):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        keys: monai.config.KeysCollection,
        is2019: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.is2019 = is2019

    def converter(self, img: monai.config.NdarrayOrTensor):
        # TC WT ET
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if self.is2019:
            result = [
                (img == 2) | (img == 3),
                (img == 1) | (img == 2) | (img == 3),
                (img == 2),
            ]
        else:
            # TC WT ET
            result = [
                (img == 1) | (img == 4),
                (img == 1) | (img == 4) | (img == 2),
                img == 4,
            ]
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def read_csv_for_GCM(config):
    csv_path = config.GCM_loader.root + "/" + "Classification.xlsx"
    # 定义dtype转换，将第二列（索引为1）读作str
    dtype_converters = {1: str}

    df1 = pd.read_excel(
        csv_path, engine="openpyxl", dtype=dtype_converters, sheet_name="腹膜转移分类"
    )
    df2 = pd.read_excel(
        csv_path,
        engine="openpyxl",
        dtype=dtype_converters,
        sheet_name="淋巴结同时序（手术）",
    )
    df3 = pd.read_excel(
        csv_path,
        engine="openpyxl",
        dtype=dtype_converters,
        sheet_name="淋巴结异时序（化疗后）",
    )

    # 创建空字典
    content_dict1 = {}
    content_dict2 = {}
    content_dict3 = {}
    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df1.iterrows():

        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4]  # 第4列的数据读为time，用于划分数据
        content_dict1[key] = [values, time]

    for index, row in df2.iterrows():

        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4]
        content_dict2[key] = [values, time]

    for index, row in df3.iterrows():

        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4]
        content_dict3[key] = [values, time]

    return content_dict1, content_dict2, content_dict3


def read_csv_for_GCNC(config):

    csv_path1 = config.GCNC_loader.root + "/" + "NPC_new.xlsx"

    # 定义dtype转换，将第四列（索引为3）读作str
    dtype_converters = {3: str}

    df = pd.read_excel(csv_path1, engine="openpyxl", dtype=dtype_converters)
    
    

    # 创建空列表
    content_dict = {}

    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df.iterrows():

        key = str(row["ID"])
        
        #TODO: 中心3以0开头设计ID，ID需要特殊读取避免删0
        if row["中心"] == 3:
            key = key.zfill(10)
        
        # TODO: 此处只做高低表达分类记录。后续数据补全可以添加阴阳性分类
        # 如果指标缺失，则跳过读取
        if isinstance(row["label"], (int, float)) != True:
            print(f"{key} miss label")
            continue

        # try:
        #     row["PD-L1"] = float(row["PD-L1"])
        # except:
        #     try:
        #         if "未做" in row["PD-L1"]:
        #             print(f"{key} miss PD-L1")
        #             continue
        #         elif "＜1" in row["PD-L1"]:
        #             row["PD-L1"] = 0
        #     except:
        #         print(f"{key} miss PD-L1")
        #         continue

        # label_dict = {"PD-L1": -1, "label": -1, "center": -1}
        label_dict = {"label": -1, "center": -1}

        # if row["PD-L1"] > 20:
        #     label_dict["PD-L1"] = 1
        # else:
        #     label_dict["PD-L1"] = 0

        label_dict["M"] = row["label"]

        label_dict["center"] = row["中心"]

        if  label_dict["M"] != -1:
            content_dict[key] = label_dict
        else:
            print(f"{key} miss label")
    return content_dict


def read_usedata(file_path):
    read_flas = False
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if read_flas == True:
                result = line.replace("\n", "").split(",")
                result = [data.replace(" ", "") for data in result]
                return result
            elif "Useful data" in line:
                read_flas = True
                continue


def load_MR_dataset_images(
    root, use_data, use_models, use_data_dict={}, data_choose="GCM", test_center=3
):
    images_path = os.listdir(root)
    images_list = []
    test_images_list = []
    images_lack_list = []

    for path in use_data:
        path = str(path)
        if path in images_path:
            models = os.listdir(root + "/" + path + "/")
        else:
            print(f"{path} is not in {root}. ")
            continue
        lack_flag = False
        lack_model_flag = False
        image = []
        label = []

        for model in models:
            if "WI" in model:
                check_model = model.replace("WI", "")
            elif model == "CT1":
                check_model = "T1+C"
            elif model == "T1+c":
                check_model = "T1+C"
            else:
                check_model = model

            if check_model in use_models:
                if not os.path.exists(root + "/" + path + "/" + model):
                    print(f"{path} does not have {model} file. ")
                    lack_model_flag = True
                    break
                elif not os.path.exists(
                    root + "/" + path + "/" + model + "/" + path + ".nii.gz"
                ):
                    print(f"{path} does not have {model} image file.")
                    lack_model_flag = True
                    break

                image.append(root + "/" + path + "/" + model + "/" + path + ".nii.gz")
                if not os.path.exists(
                    root + "/" + path + "/" + model + "/" + path + "seg.nii.gz"
                ):
                    print(f"Label file not found for {path} in model {model}. ")
                    label.append(
                        root + "/" + path + "/" + model + "/" + path + ".nii.gz"
                    )
                    lack_flag = True
                else:
                    label.append(
                        root + "/" + path + "/" + model + "/" + path + "seg.nii.gz"
                    )

        if image == [] or len(image) < len(use_models):
            print(f"{path} does not have image file or not enough modals. ")
            lack_model_flag = True

        if data_choose == "GCM":
            if lack_flag == False and lack_model_flag == False:
                if use_data_dict != {}:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                            "class_label": use_data_dict[path][0],
                        }
                    )
                else:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
            elif lack_model_flag == False:
                if use_data_dict != {}:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                            "class_label": use_data_dict[path][0],
                        }
                    )
                else:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
        else:
            if lack_flag == False and lack_model_flag == False:
                if use_data_dict != {}:
                    if use_data_dict[path]["center"] == test_center:
                        test_images_list.append(
                            {
                                "image": image,
                                "label": label,
                                # "pdl1_label": use_data_dict[path]["PD-L1"],
                                "m_label": use_data_dict[path]["M"],
                                "center": use_data_dict[path]["center"],
                            }
                        )
                    else:
                        images_list.append(
                            {
                                "image": image,
                                "label": label,
                                # "pdl1_label": use_data_dict[path]["PD-L1"],
                                "m_label": use_data_dict[path]["M"],
                                "center": use_data_dict[path]["center"],
                            }
                        )
                else:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
            elif lack_model_flag == False:
                if use_data_dict != {}:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                            # "pdl1_label": use_data_dict[path]["PD-L1"],
                            "m_label": use_data_dict[path]["M"],
                            "center": use_data_dict[path]["center"],
                        }
                    )
                else:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
        # print(f'{path} example has been loaded')

    if data_choose == "GCM":
        return images_list, images_lack_list
    else:
        return images_list, test_images_list, images_lack_list


def load_MR_tif_dataset_images(
    root, use_data, use_models):
    images_path = os.listdir(root)
    images_list = []

    for path in use_data:
        path = str(path)
        images = []
        labels = []
        
        if path not in images_path:
            print(f"{path} is not in {root}. ")
            continue
        
        for modal in use_models:
            image = []
            label = []
            
            for img in os.listdir(root + "/" + path):
                if (modal in img) and ("mask" not in img):
                    image.append(root + "/" + path + "/" + img)
                elif (modal in img) and ("mask" in img):
                    label.append(root + "/" + path + "/" + img)
            
            images.append(image)
            labels.append(label)
            
        images_list.append(
            {
                "image": images,
                "label": labels
            }
        )
        
    return images_list
        

def load_brats2021_dataset_images(root):
    images_path = os.listdir(root)
    images_list = []
    for path in images_path:
        image_path = root + "/" + path + "/" + path
        flair_img = image_path + "_flair.nii.gz"
        t1_img = image_path + "_t1.nii.gz"
        t1ce_img = image_path + "_t1ce.nii.gz"
        t2_img = image_path + "_t2.nii.gz"
        seg_img = image_path + "_seg.nii.gz"
        images_list.append(
            {"image": [flair_img, t1_img, t1ce_img, t2_img], "label": seg_img}
        )
    return images_list


def get_GCM_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    load_transform = []
    for model_scale in config.GCM_loader.model_scale:
        load_transform.append(
            monai.transforms.Compose(
                [
                    LoadImaged(
                        keys=["image", "label"], image_only=False, simple_keys=True
                    ),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Resized(
                        keys=["image", "label"],
                        spatial_size=config.GCM_loader.target_size,
                        mode=("trilinear", "nearest-exact"),
                    ),
                    ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,  # 输出图像的最小强度值
                        b_max=1.0,  # 输出图像的最大强度值
                        clip=True,  # 是否裁剪超出范围的值
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        )

    train_transform = monai.transforms.Compose(
        [
            # 训练集的额外增强
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            ToTensord(keys=["image", "label"]),
        ]
    )
    return load_transform, train_transform, val_transform


def get_GCNC_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    load_transform = []
    for model_scale in config.GCNC_loader.model_scale:
        load_transform.append(
            monai.transforms.Compose(
                [
                    LoadImaged(
                        keys=["image", "label"], image_only=False, simple_keys=True
                    ),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Resized(
                        keys=["image", "label"],
                        spatial_size=config.GCNC_loader.target_size,
                        mode=("trilinear", "nearest-exact"),
                    ),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,  # 输出图像的最小强度值
                        b_max=1.0,  # 输出图像的最大强度值
                        clip=True,  # 是否裁剪超出范围的值
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        )

    train_transform = monai.transforms.Compose(
        [
            # 训练集的额外增强
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            ToTensord(keys=["image", "label"]),
        ]
    )
    return load_transform, train_transform, val_transform


def get_FS_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:

    train_transform = monai.transforms.Compose(
        [
            ReadAndStackPerModalityd(
                keys=["image", "label"],
                spatial_size=config.FS_loader.target_size,
                image_mode="trilinear",   # 图像
                label_mode="nearest",     # 标签
                reader="PILReader",
                image_keys=("image",),
                label_keys=("label",),
                dtype_image=torch.float32,
                dtype_label=torch.float32,  # 若需要整数标签可改为 torch.long
            ),
            # 可选：强度归一化、插值、几何增强等放在这里（它们都支持 (C,H,W,D)）
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
            
            # 训练集的额外增强
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            ReadAndStackPerModalityd(
                keys=["image", "label"],
                spatial_size=config.FS_loader.target_size,
                image_mode="trilinear",   # 图像
                label_mode="nearest",     # 标签
                reader="PILReader",
                image_keys=("image",),
                label_keys=("label",),
                dtype_image=torch.float32,
                dtype_label=torch.float32,  # 若需要整数标签可改为 torch.long
            ),
            # 可选：强度归一化、插值、几何增强等放在这里（它们都支持 (C,H,W,D)）
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )
    return train_transform, val_transform


def get_Brats_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(
                keys=["label"], is2019=False
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.SpatialPadD(
                keys=["image", "label"],
                spatial_size=(255, 255, config.BraTS_loader.image_size),
                method="symmetric",
                mode="constant",
            ),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.CenterSpatialCropD(
                keys=["image", "label"],
                roi_size=ensure_tuple_rep(config.BraTS_loader.image_size, 3),
            ),
            # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=2,
                spatial_size=ensure_tuple_rep(config.BraTS_loader.image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            # monai.transforms.RandSpatialCropd(keys=["image", "label"], roi_size=config.model.image_size, random_size=False),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(
                keys="label", is2019=False
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
        ]
    )
    return train_transform, val_transform


class MultiModalityDataset(monai.data.Dataset):
    def __init__(
        self,
        data,
        loadforms,
        transforms,
        over_label=False,
        over_add=0,
        use_class=True,
        data_choose="GCM",
    ):
        self.data = data
        self.transforms = transforms
        self.loadforms = loadforms
        self.over_label = over_label
        self.over_add = over_add
        self.use_class = use_class
        self.data_choose = data_choose

    def extract_and_resize(self, image, label, over_add=0):
        # 获取label中值为1的点的坐标
        indices = torch.nonzero(label[0])  # 去掉batch维度，找到非零索引

        # 找到最长的维度
        max_size = max(image[0].shape[0], image[0].shape[1], image[0].shape[2])

        # 计算各个维度上的 over_add，按比例缩小较短维度的扩展量
        over_add_x = round(over_add * (image[0].shape[0] / max_size))
        over_add_y = round(over_add * (image[0].shape[1] / max_size))
        over_add_z = round(over_add * (image[0].shape[2] / max_size))

        # 获取在每个维度上的最小和最大索引
        min_x, min_y, min_z = indices.min(dim=0).values.tolist()
        max_x, max_y, max_z = indices.max(dim=0).values.tolist()

        # 计算扩展后的坐标，并限制在合法范围内
        min_x = max(0, min_x - over_add_x)
        max_x = min(image[0].shape[0] - 1, max_x + over_add_x)
        min_y = max(0, min_y - over_add_y)
        max_y = min(image[0].shape[1] - 1, max_y + over_add_y)
        min_z = max(0, min_z - over_add_z)
        max_z = min(image[0].shape[2] - 1, max_z + over_add_z)

        # 切割image和label
        cropped_image = image[
            :, min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1
        ]

        # 直接使用 interpolate 进行 resize 到 (1, 128, 128, 64)
        # resized_image = F.interpolate(cropped_image.unsqueeze(0), size=(128, 128, 64), mode='trilinear', align_corners=False).squeeze(0)
        resized_image = monai.transforms.Resize(
            spatial_size=(label.shape[1], label.shape[2], label.shape[3]),
            mode=("trilinear"),
        )(cropped_image)
        return resized_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        combined_data = {}

        for i in range(0, len(item["image"])):
            # print('Loading ', item['image'][i])
            globals()[f"data_{i}"] = self.loadforms[i](
                {"image": item["image"][i], "label": item["label"][i]}
            )

            combined_data[f"model_{i}_image"] = globals()[f"data_{i}"]["image"]
            combined_data[f"model_{i}_label"] = globals()[f"data_{i}"]["label"]

            if self.over_label == True:
                imgae = self.extract_and_resize(
                    combined_data[f"model_{i}_image"],
                    combined_data[f"model_{i}_label"],
                    self.over_add,
                )
            else:
                imgae = combined_data[f"model_{i}_image"]
            combined_data[f"model_{i}_image"] = imgae

        images = []
        labels = []

        for i in range(0, len(item["image"])):
            images.append(combined_data[f"model_{i}_image"])
            labels.append(combined_data[f"model_{i}_label"])
            image_tensor = torch.cat(images, dim=0)
            label_tensor = torch.cat(labels, dim=0)

        result = {"image": image_tensor, "label": label_tensor}
        result = self.transforms(result)

        if self.use_class == True:
            if self.data_choose == "GCM":
                class_label = item["class_label"]
                if class_label != 0:
                    class_label = 1
                return {
                    "image": result["image"],
                    "label": result["label"],
                    "class_label": torch.tensor(class_label).unsqueeze(0).long(),
                }
            else:
                # pdl1_label = item["pdl1_label"]
                # if pdl1_label != 0:
                #     pdl1_label = 1
                m_label = item["m_label"]
                if m_label != 0:
                    m_label = 1
                return {
                    "image": result["image"],
                    "label": result["label"],
                    # "pdl1_label": torch.tensor(pdl1_label).unsqueeze(0).long(),
                    "m_label": torch.tensor(m_label).unsqueeze(0).long(),
                    "center": torch.tensor(item["center"]).unsqueeze(0).long(),
                }
        else:
            return {"image": result["image"], "label": result["label"], "center": result["center"]}



class ReadAndStackPerModalityd(MapTransform):
    def __init__(
        self,
        keys,
        spatial_size,                 # (w, h, z)
        image_mode="trilinear",
        label_mode="nearest",
        reader="PILReader",
        image_keys=("image",),
        label_keys=("label",),
        dtype_image=torch.float32,
        dtype_label=torch.float32,
        rgb_to_gray="mean",
        allow_empty=False,
    ):
        super().__init__(keys)
        self.loader = LoadImage(reader=reader, image_only=True)

        self.tgt_w, self.tgt_h, self.tgt_z = spatial_size
        self.size_2d = (self.tgt_h, self.tgt_w)             # 2D -> (H,W)
        self.size_3d = (self.tgt_z, self.tgt_h, self.tgt_w)  # 3D -> (D,H,W)

        self.image_mode = image_mode
        self.label_mode = label_mode
        self.image_keys = set(image_keys)
        self.label_keys = set(label_keys)
        self.dtype_image = dtype_image
        self.dtype_label = dtype_label
        self.rgb_to_gray = rgb_to_gray
        self.allow_empty = allow_empty

    @staticmethod
    def _to_gray(arr, how="mean"):
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if how == "mean":
                return np.mean(arr, axis=2)
            idx = {"r": 0, "g": 1, "b": 2}[how]
            return arr[..., idx]
        return np.squeeze(arr)

    @torch.no_grad()
    def _resize2d(self, im_hw: np.ndarray, use_nearest: bool) -> np.ndarray:
        t = torch.from_numpy(im_hw).to(torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        mode = "nearest" if use_nearest else "bilinear"
        kwargs = {"size": self.size_2d, "mode": mode}
        if mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        t = F.interpolate(t, **kwargs)
        return t.squeeze(0).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def _resize3d_dhw(self, vol_dhw: torch.Tensor, use_nearest: bool) -> torch.Tensor:
        mode = "nearest" if use_nearest else "trilinear"
        kwargs = {"size": self.size_3d, "mode": mode}
        if mode == "trilinear":
            kwargs["align_corners"] = False
        return F.interpolate(vol_dhw, **kwargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            per_modality_paths = d[key]  # list[list[str]]

            is_label = key in self.label_keys
            use_nearest = True if is_label else False
            out_dtype = self.dtype_label if is_label else self.dtype_image

            vols_hwz = []
            for mi, mod_paths in enumerate(per_modality_paths):
                if (not mod_paths) and not self.allow_empty:
                    raise ValueError(f"{key}[modality {mi}] has 0 slices.")

                # 1) 每张切片读入 -> 灰度 -> 2D resize 到同一 (H,W)
                resized_slices = []
                for si, p in enumerate(mod_paths):
                    arr = np.asarray(self.loader(p))
                    arr = self._to_gray(arr, self.rgb_to_gray)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    arr = self._resize2d(arr, use_nearest=use_nearest)
                    resized_slices.append(arr)

                # 如果该模态没有切片，放一个全零占位
                if len(resized_slices) == 0:
                    vol_hwz = np.zeros((self.tgt_h, self.tgt_w, 1), dtype=np.float32)
                else:
                    # 2) 先 stack 成 (H,W,Zm)
                    vol_hwz = np.stack(resized_slices, axis=2)  # (H,W,Zm)

                    # 3) **在模态内部**统一到目标 Z： (H,W,Zm)->(1,1,Zm,H,W)->resize->(H,W,Zt)
                    t = torch.from_numpy(vol_hwz.transpose(2, 0, 1)).unsqueeze(0).unsqueeze(0).to(torch.float32)  # (1,1,Zm,H,W)
                    t = self._resize3d_dhw(t, use_nearest=use_nearest)  # (1,1,Zt,H,W)
                    vol_hwz = t.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,Zt)

                vols_hwz.append(vol_hwz)  # 每个元素现在都是同形状 (H,W,Zt)

            # 4) 现在 Z 已经一致，可以安全 stack 成 (C,H,W,Z)
            d[key] = np.stack(vols_hwz, axis=0).astype(np.float32)
            # 5) 类型
            if out_dtype != torch.float32:
                d[key] = torch.from_numpy(d[key]).to(out_dtype).cpu().numpy()

        return d

    
class TIFFDataset(monai.data.Dataset):
    """
    data: 形如 [{"image": [[...tif...],[...]], "label": [[...],[...]]}, ...]
    transform: MONAI Compose；需要把输出变成 (C,H,W,Z) 并转为 tensor
    """
    def __init__(self, data, transform):
        super().__init__(data=data, transform=transform)

        

def split_list(data, ratios):
    # 计算每个部分的大小
    sizes = [math.ceil(len(data) * r) for r in ratios]

    # 调整大小以确保总大小与原列表长度匹配
    total_size = sum(sizes)
    if total_size != len(data):
        sizes[-1] -= total_size - len(data)

    # 分割列表
    start = 0
    parts = []
    for size in sizes:
        end = start + size
        parts.append(data[start:end])
        start = end

    return parts


def check_example(data, tif=False):
    index = []
    for d in data:
        if tif != True:
            num = d["image"][0].split("/")[-1].split(".")[0]
        else:
            num = d["image"][0][0].split("/")[-2]
        index.append(num)
    return index


def sort_keys_by_time(data: dict) -> list:
    def parse_time(t):
        if isinstance(t, datetime):
            return t
        if isinstance(t, (float, int)):
            return datetime.fromtimestamp(t)
        if not isinstance(t, str):
            raise ValueError(f"Unsupported time format: {t}")

        t = t.strip()
        time_formats = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d  %H:%M:%S"]

        for fmt in time_formats:
            try:
                return datetime.strptime(t, fmt)
            except ValueError:
                continue

        raise ValueError(f"Unrecognized datetime format: {t}")

    return sorted(data.keys(), key=lambda k: parse_time(data[k][1]))


def split_examples_to_data(data, config, lack_flag=False, loding=False):
    def read_file_to_list(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            # 去除每行末尾的换行符
        lines = [line.strip() for line in lines]
        return lines

    def select_example_to_data(data, example_list):
        selected_data = []
        for d in data:
            num = d["image"][0].split("/")[-1].split(".")[0]
            if num in example_list:
                selected_data.append(d)
        return selected_data

    def load_example_to_data(data, example_path, loding=False):
        data_list = read_file_to_list(example_path)
        print(f"Loading examples from {example_path}")
        if loding == True:
            data_list = select_example_to_data(data, data_list)
        return data_list

    if config.trainer.choose_dataset == "GCM":
        data_root = config.GCM_loader.root
    elif config.trainer.choose_dataset == "GCNC":
        data_root = config.GCNC_loader.root
    elif config.trainer.choose_dataset == "FS":
        data_root = config.FS_loader.root
    
    train_example = data_root + "/" + "train_examples.txt"
    val_example = data_root + "/" + "val_examples.txt"
    test_example = data_root + "/" + "test_examples.txt"

    train_data, val_data, test_data = (
        load_example_to_data(data, train_example, loding),
        load_example_to_data(data, val_example, loding),
        load_example_to_data(data, test_example, loding),
    )

    if lack_flag == True:
        train_data_lack, val_data_lack, test_data_lack = (
            load_example_to_data(data, train_example, loding),
            load_example_to_data(data, val_example, loding),
            load_example_to_data(data, test_example, loding),
        )
        return (
            train_data,
            val_data,
            test_data,
            train_data_lack,
            val_data_lack,
            test_data_lack,
        )

    return train_data, val_data, test_data


def get_dataloader_GCM(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.GCM_loader.root
    datapath = root + "/" + "ALL" + "/"
    use_models = config.GCM_loader.checkModels

    # data1: 腹膜转移分类; data2: 淋巴结同时序（手术）; data3: 淋巴结异时序（化疗后）
    data1, data2, data3 = read_csv_for_GCM(config)
    if config.GCM_loader.task == "PM":
        use_data_dict = data1
    elif config.GCM_loader.task == "NL_SS":
        use_data_dict = data2
    else:
        use_data_dict = data3

    # 按时间顺序划分数据集
    use_data_list = sort_keys_by_time(use_data_dict)

    # 剔除不需要的病历号
    remove_list = config.GCM_loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]

    # 在use_data处划分数据，避免并行导致的读取问题
    if config.GCM_loader.fix_example != True:
        if config.GCM_loader.time_limit != True:
            random.shuffle(use_data)
            print("Random Loading!")
        train_use_data, val_use_data, test_use_data = split_list(
            use_data,
            [
                config.GCM_loader.train_ratio,
                config.GCM_loader.val_ratio,
                config.GCM_loader.test_ratio,
            ],
        )
        if config.GCM_loader.fusion == True:
            need_val_data = val_use_data + test_use_data
            val_use_data = need_val_data
            test_use_data = need_val_data
    else:
        train_use_data, val_use_data, test_use_data = split_examples_to_data(
            use_data, config
        )

    # 加载MR数据
    train_data, _ = load_MR_dataset_images(
        datapath, train_use_data, use_models, use_data_dict, data_choose="GCM"
    )
    val_data, _ = load_MR_dataset_images(
        datapath, val_use_data, use_models, use_data_dict, data_choose="GCM"
    )
    test_data, _ = load_MR_dataset_images(
        datapath, test_use_data, use_models, use_data_dict, data_choose="GCM"
    )

    # data, _ = load_MR_dataset_images(datapath, use_data, use_models, use_data_dict)

    load_transform, train_transform, val_transform = get_GCM_transforms(config)

    # if config.GCM_loader.fix_example == True:
    #     train_data, val_data, test_data = split_examples_to_data(data, config)
    # else:
    #     # shuffle data for objective verification
    #     if config.GCM_loader.time_limit != True:
    #         random.shuffle(data)
    #         print('Random Loading!')

    #     train_data, val_data, test_data = split_list(data, [config.GCM_loader.train_ratio, config.GCM_loader.val_ratio, config.GCM_loader.test_ratio])

    #     # if not need test, can use fusion to fuse two data
    #     if config.GCM_loader.fusion == True:
    #         need_val_data = val_data + test_data
    #         val_data = need_val_data
    #         test_data = need_val_data

    train_example = check_example(train_data)
    val_example = check_example(val_data)
    test_example = check_example(test_data)

    train_dataset = MultiModalityDataset(
        data=train_data,
        over_label=config.GCM_loader.over_label,
        over_add=config.GCM_loader.over_add,
        loadforms=load_transform,
        transforms=train_transform,
        data_choose="GCM",
    )
    val_dataset = MultiModalityDataset(
        data=val_data,
        over_label=config.GCM_loader.over_label,
        over_add=config.GCM_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        data_choose="GCM",
    )
    test_dataset = MultiModalityDataset(
        data=test_data,
        over_label=config.GCM_loader.over_label,
        over_add=config.GCM_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        data_choose="GCM",
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.GCM_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.GCM_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.GCM_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        (train_example, val_example, test_example),
    )


def get_dataloader_GCNC(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.GCNC_loader.root
    datapath = root + "/" + "ALL" + "/"
    use_models = config.GCNC_loader.checkModels

    content_dict = read_csv_for_GCNC(config)

    use_data_list = list(content_dict.keys())

    remove_list = config.GCNC_loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]

    data, test_data, data_lack = load_MR_dataset_images(
        datapath, use_data, use_models, content_dict, data_choose="GCNC", test_center=config.GCNC_loader.test_center
    )
    load_transform, train_transform, val_transform = get_GCNC_transforms(config)

    if config.GCNC_loader.fix_example == True:
        (train_data, val_data, _) = split_examples_to_data(
            data, config, lack_flag=False, loding=True
        )
    else:
        random.shuffle(data)
        print("Random Loading!")
        # TODO：使用第三中心数据作为测试，避免test数据划分
        train_data, val_data, _ = split_list(
            data,
            [
                config.GCNC_loader.train_ratio,
                config.GCNC_loader.val_ratio + config.GCNC_loader.test_ratio,
                0,
            ],
        )

        if config.GCNC_loader.fusion == True:
            need_val_data = val_data + test_data
            val_data = need_val_data
            test_data = need_val_data

    train_example = check_example(train_data)
    val_example = check_example(val_data)
    test_example = check_example(test_data)

    # train_lack_example = check_example(train_data_lack)
    # val_lack_example = check_example(val_data_lack)
    # test_lack_example = check_example(test_data_lack)

    train_dataset = MultiModalityDataset(
        data=train_data,
        over_label=config.GCNC_loader.over_label,
        over_add=config.GCNC_loader.over_add,
        loadforms=load_transform,
        transforms=train_transform,
        use_class=True,
        data_choose="GCNC",
    )
    val_dataset = MultiModalityDataset(
        data=val_data,
        over_label=config.GCNC_loader.over_label,
        over_add=config.GCNC_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        use_class=True,
        data_choose="GCNC",
    )
    test_dataset = MultiModalityDataset(
        data=test_data,
        over_label=config.GCNC_loader.over_label,
        over_add=config.GCNC_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        use_class=True,
        data_choose="GCNC",
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.GCNC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.GCNC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.GCNC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        (train_example, val_example, test_example),
    )


def get_dataloader_FS(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.FS_loader.root
    datapath = root + "/" + "images" + "/"+ "imgs" + "/"
    
    use_data_list = [d for d in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, d))]
    remove_list = config.FS_loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]
    
    data = load_MR_tif_dataset_images(
        datapath, use_data, config.FS_loader.checkModels
    )
    
    if config.FS_loader.fix_example == True:
        (train_data, val_data, test_data) = split_examples_to_data(
            data, config, lack_flag=False, loding=True
        )
    else:
        random.shuffle(data)
        print("Random Loading!")
        # TODO：使用第三中心数据作为测试，避免test数据划分
        train_data, val_data, test_data = split_list(
            data,
            [
                config.FS_loader.train_ratio,
                config.FS_loader.val_ratio,
                config.FS_loader.test_ratio,
            ],
        )

        if config.FS_loader.fusion == True:
            need_val_data = val_data + test_data
            val_data = need_val_data
            test_data = need_val_data
    
    train_example = check_example(train_data, tif=True)
    val_example = check_example(val_data, tif=True)
    test_example = check_example(test_data, tif=True)
    
    
    train_transform, val_transform = get_FS_transforms(config)
    
    
    train_dataset = TIFFDataset(
        data=train_data, transform=train_transform
    )
    
    val_dataset = TIFFDataset(
        data=val_data, transform=val_transform
    )
    
    test_dataset = TIFFDataset(
        data=test_data, transform=val_transform
    )
    
    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.FS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )
    
    
    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.FS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    
    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.FS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    
    return (
        train_loader,
        val_loader,
        test_loader,
        (train_example, val_example, test_example),
    )
    
    
    

def get_dataloader_BraTS(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_images = load_brats2021_dataset_images(config.BraTS_loader.dataPath)
    train_transform, val_transform = get_Brats_transforms(config)
    train_dataset = monai.data.Dataset(
        data=train_images[: int(len(train_images) * config.BraTS_loader.train_ratio)],
        transform=train_transform,
    )
    val_dataset = monai.data.Dataset(
        data=train_images[int(len(train_images) * config.BraTS_loader.train_ratio) :],
        transform=val_transform,
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.BraTS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.BraTS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(
            open("/workspace/GZTumor/config.yml", "r", encoding="utf-8"),
            Loader=yaml.FullLoader,
        )
    )

    # train_loader, val_loader, test_loader, _ = get_dataloader_GCM(config)
    # train_loader, val_loader, test_loader, _ = get_dataloader_GCNC(config)
    train_loader, val_loader, test_loader, _ = get_dataloader_FS(config)

    conut = 0
    f_count = 0

    pd_l1_count = 0
    pd_l1_f_count = 0

    count = 0
    for i, batch in enumerate(train_loader):
        try:
            # print(batch["image"].shape)
            # print(batch["label"].shape)
            count += 1
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue
    
    print(f"Total train batches: {count}")    
      
    count = 0    
    for i, batch in enumerate(val_loader):
        try:
            # print(batch["image"].shape)
            # print(batch["label"].shape)
            count += 1
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue
    
    print(f"Total val batches: {count}")  
    
    count = 0    
    for i, batch in enumerate(test_loader):
        try:
            # print(batch["image"].shape)
            # print(batch["label"].shape)
            count += 1
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue

    print(f"Total test batches: {count}")  