import torch as tc
import re
import os
from matplotlib import pyplot as plt
# from Library.TensorNetwork import TensorTrain_pack, copy_from_mps_pack
# from Library.TN_ADQC import *

def search_qc(folder_path, evol_num):
    # 使用正则表达式匹配文件名中的 evol 和 temp
    pattern = re.compile(r'qc_param_sample_\d+_evol_(\d+)\.pt')

    max_temp = None
    max_temp_file = None

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            match = pattern.search(filename)
            if match:
                temp = int(match.group(1))

                # 检查 temp 小于给定的 evol_num，并且是最大的
                if temp < evol_num and (max_temp is None or temp > max_temp):
                    max_temp = temp
                    max_temp_file = filename

    return max_temp_file

path = 'GraduationProject/Data/PXP/length10/loss_multi_mags/0.7/d'
old = search_qc(path, 7)
print(old)