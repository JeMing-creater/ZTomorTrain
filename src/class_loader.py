import os
import math
import yaml
import torch
import monai
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from easydict import EasyDict
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

def load_MR_dataset_images(root, use_data, use_models, use_data_dict):
    root1 = root + '/' + 'NonsurgicalMR' + '/'
    root2 = root + '/' + 'SurgicalMR' + '/'
    images_path1 = os.listdir(root1)
    images_path2 = os.listdir(root2)
    images_list = []
    for path in use_data:
        if path in images_path1:
            models = os.listdir(root1 + '/' + path + '/')
            root = root1
        elif path in images_path2:
            models = os.listdir(root2 + '/' + path + '/')
            root = root2
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


class MultiModalityDataset(monai.data.Dataset):
    def __init__(self, data, loadforms, transforms):
        self.data = data
        self.transforms = transforms
        self.loadforms = loadforms
        
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
        
        images = []
        labels = []
        
        for i in range(0, len(item['image'])):
            images.append(combined_data[f'model_{i}_image'])
            labels.append(combined_data[f'model_{i}_label'])
            image_tensor = torch.cat(images, dim=0)
            label_tensor = torch.cat(labels, dim=0)
        
        result = {'image': image_tensor, 'label': label_tensor}
        result = self.transforms(result)
        return {'image': result['image'], 'label': result['label'], 'class_label': torch.tensor(item['class_label']).view(-1,1)}
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

def get_dataloader(config: EasyDict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    datapath = config.loader.dataPath
    use_models = config.loader.checkModels
    
    use_data_dict = read_csv(config)
    use_data = list(use_data_dict.keys())
    
    
    data = load_MR_dataset_images(datapath, use_data, use_models, use_data_dict)
    
    load_transform, train_transform, val_transform = get_transforms(config)
    
    train_data, val_data, test_data = split_list(data, [config.loader.train_ratio, config.loader.val_ratio, config.loader.test_ratio]) 

    train_dataset = MultiModalityDataset(data=train_data, 
                                         loadforms = load_transform,
                                         transforms=train_transform)
    val_dataset   = MultiModalityDataset(data=val_data, 
                                         loadforms = load_transform,
                                         transforms=val_transform)
    test_dataset   = MultiModalityDataset(data=test_data, 
                                         loadforms = load_transform,
                                         transforms=val_transform)
    
    train_loader = monai.data.DataLoader(train_dataset, num_workers=config.loader.num_workers,
                                         batch_size=config.trainer.batch_size, shuffle=True)
    val_loader = monai.data.DataLoader(val_dataset, num_workers=config.loader.num_workers, 
                                       batch_size=config.trainer.batch_size, shuffle=False)
    test_loader = monai.data.DataLoader(test_dataset, num_workers=config.loader.num_workers, 
                                       batch_size=config.trainer.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    config = EasyDict(yaml.load(open('/workspace/Jeming/Project1/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    train_loader, val_loader, test_loader = get_dataloader(config)
    
    for i, batch in enumerate(train_loader):
        print(batch['image'].shape)
        print(batch['label'].shape)
        print(batch['class_label'].shape)
        print(batch['class_label'])

    for i, batch in enumerate(val_loader):
        print(batch['image'].shape)
        print(batch['label'].shape)
        print(batch['class_label'].shape)
        print(batch['class_label'])
    
    for i, batch in enumerate(test_loader):
        print(batch['image'].shape)
        print(batch['label'].shape)
        print(batch['class_label'].shape)
        print(batch['class_label'])