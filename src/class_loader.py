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
from easydict import EasyDict
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep
from monai.networks.utils import one_hot
sitk.ProcessObject.SetGlobalWarningDisplay(False)
from typing import Tuple, List, Mapping, Hashable, Dict
from monai.transforms import (
    LoadImaged, MapTransform, ScaleIntensityRanged, EnsureChannelFirstd, Spacingd, Orientationd,ResampleToMatchd, ResizeWithPadOrCropd, Resize, Resized, RandFlipd, NormalizeIntensityd, ToTensord,RandScaleIntensityd,RandShiftIntensityd
)

def read_csv(config):
    def change_p(use_Pathology):
        keep_pathology = [0,0,0,0,0,0]
        if 'SMI' in use_Pathology:
            keep_pathology[0] = 1
        if 'LNM' in use_Pathology:
            keep_pathology[1] = 1
        if 'VI' in use_Pathology:
            keep_pathology[2] = 1
        if 'NBI' in use_Pathology:
            keep_pathology[3] = 1
        if 'HER2' in use_Pathology:
            keep_pathology[4] = 1
        if 'KI67' in use_Pathology:
            keep_pathology[5] = 1
        return keep_pathology
    
    csv_path = config.loader.csvPath
    use_Pathology = config.loader.checkPathology
    retention_list = change_p(use_Pathology)
    # 定义dtype转换，将第二列（索引为1）读作str
    dtype_converters = {1: str}
    
    df = pd.read_excel(csv_path, engine='openpyxl', dtype=dtype_converters)
    
    # 创建空字典
    content_dict = {}
    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df.iterrows():
        # if index == 0:
        #     continue  # 跳过第一行
        
        key = row[1]  # 第2列作为键
        values = row[2:8].tolist()  # 第3-8列的数据读为列表
        
        content_dict[key] = values
    
    # 把nan值改换为0
    for key in content_dict.keys():
        new_dict = [data for data, retain in zip(content_dict[key], retention_list) if retain == 1]
        
        for i in range(len(new_dict)):
            if np.isnan(new_dict[i]):
                new_dict[i] = 0.0
        content_dict[key] = new_dict
        
    return content_dict

def read_csv_for_PM(config):
    csv_path = config.loader.csvPath
    # 定义dtype转换，将第二列（索引为1）读作str
    dtype_converters = {1: str}
    
    df1 = pd.read_excel(csv_path, engine='openpyxl', dtype=dtype_converters, sheet_name='腹膜转移分类')
    df2 = pd.read_excel(csv_path, engine='openpyxl', dtype=dtype_converters, sheet_name='淋巴结同时序（手术）')
    df3 = pd.read_excel(csv_path, engine='openpyxl', dtype=dtype_converters, sheet_name='淋巴结异时序（化疗后）')

    # 创建空字典
    content_dict1 = {}
    content_dict2 = {}
    content_dict3 = {}
    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df1.iterrows():
        
        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        
        content_dict1[key] = values
    
    for index, row in df2.iterrows():
        
        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        
        content_dict2[key] = values

    for index, row in df3.iterrows():
        
        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        
        content_dict3[key] = values

    return content_dict1, content_dict2, content_dict3

def load_MR_dataset_images(root, use_data, use_models, use_data_dict):
    images_path = os.listdir(root)
    images_list = []
    for path in use_data:
        if path in images_path:
            models = os.listdir(root + '/' + path + '/')
        else:
            continue
        
        image = []
        label = []
        
        for model in models:
            if model in use_models:
                image.append(root + '/' + path + '/' + model + '/' + path + '.nii.gz')
                label.append(root + '/' + path + '/' + model + '/' + path + 'seg.nii.gz')
        
        images_list.append({
            'image': image,
            'label': label,
            'class_label': use_data_dict[path]
            })           
    return images_list

def get_transforms(config: EasyDict) -> Tuple[
    monai.transforms.Compose, monai.transforms.Compose]:
    
    load_transform = []
    
    for model_scale in config.loader.model_scale:
        load_transform.append(
            monai.transforms.Compose([
                LoadImaged(keys=["image", "label"], image_only=False, simple_keys=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                Resized(keys=["image", "label"], spatial_size=config.loader.target_size, mode=("trilinear", "nearest-exact")),
                
                ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,            # 输出图像的最小强度值
                        b_max=1.0,            # 输出图像的最大强度值
                        clip=True             # 是否裁剪超出范围的值
                    ),
                ToTensord(keys=['image', 'label'])
            ])
        )
    
    train_transform = monai.transforms.Compose([
        # 训练集的额外增强
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=['image', 'label'])
    ])
    val_transform = monai.transforms.Compose([
        ToTensord(keys=["image", "label"]),
    ])
    return load_transform, train_transform, val_transform

def extract_and_resize(image, label, over_add=0):
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
    max_x = min(image[0].shape[0]-1, max_x + over_add_x)
    min_y = max(0, min_y - over_add_y)
    max_y = min(image[0].shape[1]-1, max_y + over_add_y)
    min_z = max(0, min_z - over_add_z)
    max_z = min(image[0].shape[2]-1, max_z + over_add_z)
    
    # 切割image和label
    cropped_image = image[:, min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    
    # 直接使用 interpolate 进行 resize 到 (1, 128, 128, 64)
    # resized_image = F.interpolate(cropped_image.unsqueeze(0), size=(128, 128, 64), mode='trilinear', align_corners=False).squeeze(0)
    resized_image = monai.transforms.Resize(spatial_size=(label.shape[1],label.shape[2],label.shape[3]), mode=("trilinear"))(cropped_image)
    return resized_image

class MultiModalityDataset(monai.data.Dataset):
    def __init__(self, data, loadforms, transforms, over_label=False, over_add=0):
        self.data = data
        self.transforms = transforms
        self.loadforms = loadforms
        self.over_label = over_label
        self.over_add = over_add
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        combined_data = {}
        
        for i in range(0, len(item['image'])):
            globals()[f'data_{i}'] = self.loadforms[i]({
                'image': item['image'][i],
                'label': item['label'][i]
            })

            combined_data[f'model_{i}_image'] = globals()[f'data_{i}']['image']
            combined_data[f'model_{i}_label'] = globals()[f'data_{i}']['label']

            imgae = extract_and_resize(combined_data[f'model_{i}_image'], combined_data[f'model_{i}_label'], self.over_add)
            combined_data[f'model_{i}_image'] = imgae
            
        images = []
        labels = []
        
        for i in range(0, len(item['image'])):
            images.append(combined_data[f'model_{i}_image'])
            labels.append(combined_data[f'model_{i}_label'])
            image_tensor = torch.cat(images, dim=0)
            label_tensor = torch.cat(labels, dim=0)
        
        result = {'image': image_tensor, 'label': label_tensor}
        result = self.transforms(result)
        return {'image': result['image'], 'label': result['label'], 'class_label': torch.tensor(item['class_label']).unsqueeze(0).long()}

def split_list(data, ratios):
    # 计算每个部分的大小
    sizes = [math.ceil(len(data) * r) for r in ratios]
    
    # 调整大小以确保总大小与原列表长度匹配
    total_size = sum(sizes)
    if total_size != len(data):
        sizes[-1] -= (total_size - len(data))
    
    # 分割列表
    start = 0
    parts = []
    for size in sizes:
        end = start + size
        parts.append(data[start:end])
        start = end
    
    return parts

def calculate_ratio(input_dict):
    total_zeros = 0
    total_ones = 0
    total_elements = 0
    
    for value in input_dict.values():
        total_zeros += value.count(0)
        total_ones += value.count(1)
        total_elements += len(value)
    
    ratio_zeros = total_zeros / total_elements if total_elements > 0 else 0
    ratio_ones = total_ones / total_elements if total_elements > 0 else 0
    
    return {'0': ratio_zeros, '1': ratio_ones}

def calculate_label_ratio(train_loader):
    total_zeros = 0
    total_ones = 0
    total_labels = 0

    # 遍历 train_loader 中的所有批次
    for i, batch in enumerate(train_loader):
        labels = batch['class_label']
        total_zeros += (labels == 0).sum().item()  # 统计标签为0的数量
        total_ones += (labels == 1).sum().item()   # 统计标签为1的数量
        # total_labels += labels.size(0)             # 统计标签的总数量

    total = total_zeros + total_ones
    a_ratio = total_zeros / total
    b_ratio = total_ones / total
    
    return [a_ratio, b_ratio]

def get_dataloader(config: EasyDict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    datapath = config.loader.dataPath
    use_models = config.loader.checkModels
    
    # data1: 腹膜转移分类; data2: 淋巴结同时序（手术）; data3: 淋巴结异时序（化疗后）
    data1, data2, data3 = read_csv_for_PM(config)
    if config.loader.task == 'PM':
        use_data_dict = data1
    elif config.loader.task == 'NL_SS':
        use_data_dict = data2
    else:
        use_data_dict = data3

    use_data_list = list(use_data_dict.keys())
    remove_list = config.loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]
    
    data = load_MR_dataset_images(datapath, use_data, use_models, use_data_dict)
    
    load_transform, train_transform, val_transform = get_transforms(config)
    
    # shuffle data for objective verification
    random.shuffle(data)

    train_data, val_data, test_data = split_list(data, [config.loader.train_ratio, config.loader.val_ratio, config.loader.test_ratio]) 

    # if not need test, can use fusion to fuse two data
    if config.loader.fusion == True:
        need_val_data = val_data + test_data
        val_data = need_val_data
        test_data = need_val_data

    train_dataset = MultiModalityDataset(data=train_data, over_label=config.loader.over_label, over_add = config.loader.over_add, 
                                         loadforms = load_transform,
                                         transforms=train_transform)
    val_dataset   = MultiModalityDataset(data=val_data, over_label=config.loader.over_label, over_add = config.loader.over_add,
                                         loadforms = load_transform,
                                         transforms=val_transform)
    test_dataset   = MultiModalityDataset(data=test_data, over_label=config.loader.over_label, over_add = config.loader.over_add,
                                         loadforms = load_transform,
                                         transforms=val_transform)
    
    train_loader = monai.data.DataLoader(train_dataset, num_workers=config.loader.num_workers,
                                         batch_size=config.trainer.batch_size, shuffle=True)
    val_loader = monai.data.DataLoader(val_dataset, num_workers=config.loader.num_workers, 
                                       batch_size=config.trainer.batch_size, shuffle=False)
    test_loader = monai.data.DataLoader(test_dataset, num_workers=config.loader.num_workers, 
                                       batch_size=config.trainer.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def check_loader(config):
    train_loader, val_loader, test_loader = get_dataloader(config)
    
    for i, batch in enumerate(train_loader):
        try:
            print(batch['image'].shape)
            print(batch['label'].shape)
            print(batch['class_label'].shape)
            # print(batch['class_label'])
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue 

    for i, batch in enumerate(val_loader):
        try:
            print(batch['image'].shape)
            print(batch['label'].shape)
            print(batch['class_label'].shape)
            # print(batch['class_label'])
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue 

    for i, batch in enumerate(test_loader):
        try:
            print(batch['image'].shape)
            print(batch['label'].shape)
            print(batch['class_label'].shape)
            # print(batch['class_label'])
        except Exception as e:
            print(f"Error occurred while loading batch {i}: {e}")
            continue


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('/workspace/Jeming/Project1/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    config.loader.task = 'PM'
    check_loader(config)

    config.loader.task = 'NL_SS'
    check_loader(config)

    config.loader.task = 'NL_DS'
    check_loader(config)