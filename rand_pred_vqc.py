from math import dist
import ADQC
import torch as tc
import numpy as np
from Library import BasicFun as bf
from Library import PhysModule as phy
from Library.BasicFun import choose_device, mkdir
from torch.utils.data import DataLoader, TensorDataset

# 通用参数
para = {'lr': 1e-2,  # 初始学习率
        # 'length_tot': 500,  # 序列总长度
        # 'order': 10,  # 生成序列的傅里叶阶数
        # 'length': 1,  # 每个样本长度
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
parser.add_argument('--folder', type=str, default="/rand_unitary/loss_mags/dn")
parser.add_argument('--train_num', type=int, default=100)
parser.add_argument('--loss_type', type=str, default='multi_mags')
args = parser.parse_args()
# para_adqc['folder'] = args.folder
# para_adqc['seed'] = args.seed
para_adqc['loss_type'] = args.loss_type
para_adqc['ini_way'] = 'identity'
path = 'GraduationProject/Data'+args.folder
mkdir(path)
data = np.load(path+'/data_num{:d}.npy'.format(args.train_num), allow_pickle=True)
data = data.item()

print(data['train_set'].dtype)

qc = ADQC.VQC(para_adqc)
# from torchsummary import summary
# summary(qc, input_size=tuple(2 for _ in range(10)))
# print(qc)
qc, results_adqc, para_adqc = ADQC.train(qc, data, para_adqc)
np.save(path+'/adqc_result_num{:d}'.format(args.train_num), results_adqc)


qc.single_state = False
E = tc.eye(2**para_adqc['length_in'], dtype=tc.complex128, device=para_adqc['device'])
shape_ = [E.shape[0]] + [2] * para_adqc['length_in']
E = E.reshape(shape_)
with tc.no_grad():
    qc_mat = qc(E).reshape([E.shape[0], -1])

print('\nqc_mat.shape is', qc_mat.shape)
np.save(path+'/qc_mat_num{:d}'.format(args.train_num), qc_mat.cpu())

# A = tc.mm(qc_mat, qc_mat.T.conj())
# print(A)
# print(tc.diag(A))

