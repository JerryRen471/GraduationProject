from math import dist
import ADQC
import torch as tc
import numpy as np
from Library import BasicFun as bf
from Library import PhysModule as phy
from Library.BasicFun import choose_device, mkdir
from torch.utils.data import DataLoader, TensorDataset
import os
import re

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

# 通用参数
para = {'lr': 1e-2,  # 初始学习率
        'batch_size': 2000,  # batch大小
        'it_time': 1000,  # 总迭代次数
        'dtype': tc.complex128,  # 数据精度
        'device': choose_device()}  # 计算设备（cuda优先）

# ADQC参数
para_adqc = {'depth': 4, 'loss_type': 'multi_mag'}  # ADQC量子门层数，损失函数类型

para_adqc = dict(para, **para_adqc)

# 添加运行参数
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--folder', type=str, default="/XXZ_inhomo/delta1/theta0/length10/loss_multi_mags/0.1/dn")
parser.add_argument('--length', type=int, default=10)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--evol_num', type=int, default=2)
parser.add_argument('--loss_type', type=str, default='multi_mags')
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--rec_time', type=int, default=1)

args = parser.parse_args()
# para_adqc['folder'] = args.folder
# para_adqc['seed'] = args.seed
para_adqc['loss_type'] = args.loss_type
para_adqc['ini_way'] = 'identity'
para_adqc['depth'] = args.depth
para_adqc['length_in'] = args.length
para_adqc['recurrent_time'] = args.rec_time
path = 'GraduationProject/Data'+args.folder
mkdir(path)
data = np.load(path+'/train_set_sample_{:d}_evol_{:d}.npy'.format(args.sample_num, args.evol_num), allow_pickle=True)
data = data.item()

print(data['train_set'].dtype)

qc = ADQC.ADQC(para_adqc)
old_qc_path = search_qc(folder_path=path, evol_num=args.evol_num)
print(old_qc_path)
if old_qc_path != None:
    qc.load_state_dict(tc.load(path + '/' + old_qc_path), strict=False)
qc, results_adqc, para_adqc = ADQC.train(qc, data, para_adqc)
np.save(path+'/adqc_result_sample_{:d}_evol_{:d}'.format(args.sample_num, args.evol_num), results_adqc)

qc.single_state = False
E = tc.eye(2**para_adqc['length_in'], dtype=tc.complex128, device=para_adqc['device'])
shape_ = [E.shape[0]] + [2] * para_adqc['length_in']
E = E.reshape(shape_)
with tc.no_grad():
    for _ in range(args.rec_time):
        E = qc(E)
    qc_mat = E.reshape([E.shape[0], -1])

print('\nqc_mat.shape is', qc_mat.shape)
np.save(path+'/qc_mat_sample_{:d}_evol_{:d}'.format(args.sample_num, args.evol_num), qc_mat.cpu())
tc.save(qc.state_dict(), path+'/qc_param_sample_{:d}_evol_{:d}.pt'.format(args.sample_num, args.evol_num))

# A = tc.mm(qc_mat, qc_mat.T.conj())
# print(A)
# print(tc.diag(A))