import os
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
    csv_path = config.loader.csvPath
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # 创建空字典
    content_dict = {}
    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df.iterrows():
        if index == 0:
            continue  # 跳过第一行
        
        key = row[1]  # 第2列作为键
        values = row[2:8].tolist()  # 第3-8列的数据读为列表
        
        content_dict[key] = values
    
    return content_dict

if __name__ == '__main__':
    config = EasyDict(yaml.load(open('/workspace/Jeming/Project1/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    content_dict = read_csv(config)
    print(content_dict)