from ctypes.wintypes import tagRECT
import re
import os

def search_qc(folder_path, sample_num, evol_num):
    # 使用正则表达式匹配文件名中的 evol 和 sample
    pattern = re.compile(r'qc_param_sample_(\d+)_evol_(\d+)\.pt')

    max_sample = None
    max_evol = None
    target_file = None

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            match = pattern.search(filename)
            if match:
                temp_sample = int(match.group(1))
                temp_evol = int(match.group(2))

                # 检查 temp_sample 小于给定的 sample_num
                if temp_sample < sample_num:
                    # 如果 max_sample 为空或 temp_sample 大于 max_sample，更新 max_sample 和 max_evol
                    if max_sample is None or temp_sample > max_sample:
                        max_sample = temp_sample
                        max_evol = temp_evol  # 更新 max_evol
                        target_file = filename
                    # 如果 temp_sample 等于 max_sample，检查 temp_evol
                    elif temp_sample == max_sample and temp_evol < evol_num:
                        if max_evol is None or temp_evol > max_evol:
                            max_evol = temp_evol
                            target_file = filename

    return target_file

target = search_qc('/data/home/scv7454/run/GraduationProject/Data/PXP/length10/loss_fidelity/0.20/product', sample_num=5, evol_num=10)
print(target)