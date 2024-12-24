import torch as tc
import numpy as np

from matplotlib import pyplot as plt

# 导入数据
states = np.load('D:\\Manual\\Code\\tests\\GraduationProject\\Data\\states.npy', allow_pickle=True)
print(states[0].shape)
data = tc.stack(list(tc.from_numpy(states)))
print(data.shape)

probs = tc.mul(data, data.conj())
print(probs.shape[0])

# 导入模型进行训练
from ADQC import *

print('设置参数')
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
para_adqc = {'depth': 4}  # ADQC量子门层数

para_adqc = dict(para, **para_adqc)

qc, results_adqc, para_adqc = \
        ADQC(probs, para_adqc)
output_adqc = tc.cat([results_adqc['train_pred'],
                    results_adqc['test_pred']], dim=0)

# 画图

x_range = probs.shape[0]
train_num = int(x_range - x_range*para_adqc['test_ratio'])
plt.plot(x_range, probs[:, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
plt.plot(x_range[0:train_num], output_adqc['train_pred'])
plt.plot(x_range, output_adqc['test_pred'])

tc.save(qc, 'D:\\Manual\\Code\\tests\\GraduationProject\\Data\\qc_probs.pth')
np.save('D:\\Manual\\Code\\tests\\GraduationProject\\Data\\probs_adqc', output_adqc)