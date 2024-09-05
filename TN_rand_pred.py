from math import dist
import ADQC
import torch as tc
import numpy as np
from Library import TN_ADQC, BasicFun as bf
from Library import PhysModule as phy
from Library.BasicFun import choose_device, mkdir
from torch.utils.data import DataLoader, TensorDataset

# tc.autograd.set_detect_anomaly(True)
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
parser.add_argument('--folder', type=str, default="/TN_Heis/length5/loss_multi_mags/0.1/dn")
parser.add_argument('--length', type=int, default=5)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--evol_num', type=int, default=1)
parser.add_argument('--loss_type', type=str, default='multi_mags')
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--rec_time', type=int, default=1)

args = parser.parse_args()
# para_adqc['folder'] = args.folder
# para_adqc['seed'] = args.seed
para_adqc['state_type'] = 'mps'
para_adqc['loss_type'] = args.loss_type
para_adqc['ini_way'] = 'random'
para_adqc['depth'] = args.depth
para_adqc['length_in'] = args.length
para_adqc['recurrent_time'] = args.rec_time
path = 'GraduationProject/Data'+args.folder
mkdir(path)
data = tc.load(path+'/train_set_sample_{:d}_evol_{:d}.pt'.format(args.sample_num, args.evol_num))
# data = data.item()

print(data['train_set'].dtype)

qc = TN_ADQC.ADQC(para_adqc)
# traced_qc = tc.jit.trace(qc, (data['train_set'],))
# print()
# print(traced_qc.graph)

qc, results_adqc, para_adqc = TN_ADQC.train(qc, data, para_adqc)
tc.save(results_adqc, path+'/adqc_result_sample_{:d}_evol_{:d}.pt'.format(args.sample_num, args.evol_num))

# qc.single_state = False
# E = tc.eye(2**para_adqc['length_in'], dtype=tc.complex128, device=para_adqc['device'])
# shape_ = [E.shape[0]] + [2] * para_adqc['length_in']
# E = E.reshape(shape_)
# with tc.no_grad():
#     for _ in range(args.rec_time):
#         E = qc(E)
#     qc_mat = E.reshape([E.shape[0], -1])

# print('\nqc_mat.shape is', qc_mat.shape)
tc.save(qc, path+'/qc_mat_sample_{:d}_evol_{:d}.pt'.format(args.sample_num, args.evol_num))

# A = tc.mm(qc_mat, qc_mat.T.conj())
# print(A)
# print(tc.diag(A))

