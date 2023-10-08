import ADQC
import torch as tc
import numpy as np
from Library import BasicFun as bf
from Library import PhysModule as phy
from Library.BasicFun import choose_device
from torch.utils.data import DataLoader, TensorDataset

# 通用参数
para = {'lr': 1e-2,  # 初始学习率
        'length_tot': 500,  # 序列总长度
        'order': 10,  # 生成序列的傅里叶阶数
        'length': 1,  # 每个样本长度
        'batch_size': 2000,  # batch大小
        'it_time': 1000,  # 总迭代次数
        'dtype': tc.complex128,  # 数据精度
        'device': choose_device()}  # 计算设备（cuda优先）

# ADQC参数
para_adqc = {'depth': 4, 'loss_type': 'mag'}  # ADQC量子门层数

para_adqc = dict(para, **para_adqc)

# 添加运行参数
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--folder', type=str, default='rand_init/')
parser.add_argument('--train_num', type=int, default=100)
parser.add_argument('--loss_type', type=str, default='fidelity')
args = parser.parse_args()
# para_adqc['folder'] = args.folder
# para_adqc['seed'] = args.seed
para_adqc['loss_type'] = args.loss_type

data = np.load('GraduationProject/Data/'+args.folder+'data_num{:d}.npy'.format(args.train_num), allow_pickle=True)
data = data.item()

print(data['train_set'].dtype)

qc = ADQC.ADQC(para_adqc)
qc, results_adqc, para_adqc = ADQC.train(qc, data, para_adqc)
np.save('GraduationProject/Data/'+args.folder+'adqc_result_num{:d}'.format(args.train_num), results_adqc)