from easydict import EasyDict
import SimpleITK as sitk
import os, cv2, yaml
from src.utils import (
    get_directory_item, check_open_files, check_subdirectories_contain_files,check_slices_consistency,check_modalities_resolution, check_label_size, write_result
)
    
                       

def check(config, dataPath, checkModels):
    directory_item = get_directory_item(dataPath)

    # TODO: 0. 判断是否有打不开的nii文件
    cannot_open = {}
    # TODO: 1. 确定哪个编号空缺模态，如皆不空缺，返回(True, []), 否则返回(False, [模态名])
    loss_model = {}
    # TODO: 2. 确定同编号下，不同模态切片数量是否对齐，如皆对齐返回(True, {}), 如不对齐返回(False, {模态名：切片数量})
    unalign_model = {}
    # TODO: 3. 确定所有模态分辨率是否合理，合理返回(True, {}), 不合理返回(False, {模态名：分辨率})
    unResolution_model = {}
    # TODO: 4. 确定所有模态的病灶大小是否合理，合理返回(True, {}), 不合理返回(False, {模态名：病灶体素大小})
    unSize_model = {}
    # TODO: 5. 统筹能用的数据编号
    use_model = []
    if directory_item != []:
        for item in directory_item:
            this_data_path = dataPath + '/' + item + '/'
            all_models = get_directory_item(this_data_path)
            # TODO: 0
            check_flag, error_item = check_open_files(this_data_path, checkModels)
            if check_flag == False:
                cannot_open[item] = error_item
                continue
            # TODO: 1
            check_flag, loss_item = check_subdirectories_contain_files(this_data_path, all_models, checkModels)
            if check_flag == False:
                loss_model[item] = loss_item
            else:
                # 如果模态缺失，则无所谓对齐，如果不缺失，则考虑对齐
                # TODO: 2
                check_flag, unalign_item = check_slices_consistency(this_data_path, checkModels)
                if check_flag == False:
                    unalign_model[item] = unalign_item
                    
            # 无论模态是否缺失，判断已有模态分辨率是否达标
            # TODO: 3
            check_flag, unResolution_item = check_modalities_resolution(this_data_path, checkModels, config.lowestResolution)
            
            if check_flag  == False:
                unResolution_model[item] = unResolution_item
            
            # 无论模态是否缺失，判断已有模态病灶大小是否达标
            # TODO: 4
            check_flag, unSize_item = check_label_size(this_data_path, checkModels, config.lowestSize)

            if check_flag  == False:
                unSize_model[item] = unSize_item
            
    else:
        print('dataPath has not data!')
    if config.is_align != True:
        unalign_model = {}
    elif config.is_Resolution != True:
        unResolution_model = {}
    elif config.is_Size != True:
        unSize_model = {}
    # TODO: 5
    for item in directory_item:
        if item not in loss_model.keys() and item not in unalign_model.keys() and item not in unResolution_model.keys() and item not in unSize_model.keys():
            use_model.append(item)
  
    return cannot_open, loss_model, unalign_model, unResolution_model, unSize_model, use_model

def main(config):
    # 检查的模态
    checkModels = config.checkModels
    # 写入路径
    writePath = config.writePath
  
    # 非手术勾画
    dataPath1 = config.dataPath1
    all_result1 = check(config, dataPath1, checkModels)
    write_result(config, writePath+'/'+'NonsurgicalMR.txt', all_result1)
  
    # 手术勾画
    dataPath2 = config.dataPath2
    all_result2 = check(config, dataPath2, checkModels)
    write_result(config, writePath+'/'+'SurgicalMR.txt', all_result2)
  

if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    main(config.data_check)
    