import ADQC
import torch as tc
import numpy as np
from Library import PhysModule as phy
from Library.BasicFun import choose_device, mkdir

from rand_time_sequence import main as generate_main

#=======================
# 分配运行参数
#=======================

import argparse
parser = argparse.ArgumentParser(description='manual to this script')

parser.add_argument('--Jx', type=float, default=1)
parser.add_argument('--Jy', type=float, default=0)
parser.add_argument('--Jz', type=float, default=1)
parser.add_argument('--hx', type=float, default=0.5)
parser.add_argument('--hl', type=float, default=2)
parser.add_argument('--gen_type', type=str, default='nn')
parser.add_argument('--time_tot', type=float, default=1)
parser.add_argument('--print_time', type=int, default=10)
parser.add_argument('--folder', type=str, default='rand_init/')
parser.add_argument('--evol_mat_path', type=str, default="GraduationProject/Data/evol_mat.npy")
parser.add_argument('--loss_type', type=str, default='multi_mags')
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--rec_time', type=int, default=1)
args = parser.parse_args()

# 数据生成参数
para_data = dict()
para_data['J'] = [args.Jx, args.Jy, args.Jz]
para_data['h'] = [args.hx, 0, 0]
para_data['hl'] = args.hl
para_data['gen_type'] = args.gen_type
para_data['time_tot'] = args.time_tot
para_data['print_time'] = args.print_time
para_data['seed'] = None
para_data['tau'] = 0.01
para_data['train_num'] = 1
para_data['test_num'] = 1

# 训练参数
para_train = {'lr': 1e-2,  # 初始学习率
        'batch_size': 2000,  # batch大小
        'it_time': 2000,  # 总迭代次数
        'dtype': tc.complex128,  # 数据精度
        'device': choose_device()}  
para_train['loss_type'] = args.loss_type
para_train['ini_way'] = 'identity'
para_train['depth'] = args.depth
para_train['recurrent_time'] = args.rec_time

# 通用参数
para_general = dict()
para_general['folder'] = args.folder
para_general['evol_path'] = args.evol_mat_path
path = 'GraduationProject/Data'+para_general['folder']

#=======================
# 生成训练数据
#=======================

length = 10
spin = 'half'
d = phy.from_spin2phys_dim(spin)
device = tc.device('cuda:0')
dtype = tc.complex128
para_generate = dict(para_general, **para_data)

data = generate_main(para_generate)

#=======================
# 训练adqc
#=======================
para_adqc = dict(para_general, **para_train)
qc = ADQC.ADQC(para_adqc)
qc, results_adqc, para_adqc = ADQC.train(qc, data, para_adqc)
np.save(path+'/adqc_result_num{:d}'.format(para_data['time_tot']), results_adqc)

qc.single_state = False
E = tc.eye(2**para_adqc['length_in'], dtype=tc.complex128, device=para_adqc['device'])
shape_ = [E.shape[0]] + [2] * para_adqc['length_in']
E = E.reshape(shape_)
with tc.no_grad():
    for _ in range(para_train['recurrent_time']):
        E = qc(E)
    qc_mat = E.reshape([E.shape[0], -1])

print('\nqc_mat.shape is', qc_mat.shape)
np.save(path+'/qc_mat_num{:d}'.format(para_data['time_tot']), qc_mat.cpu())
