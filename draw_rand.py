import torch as tc
import numpy as np
from Library.BasicFun import mkdir
from Library import PhysModule as phy
from matplotlib import pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--folder', type=str, default='rand_init/')
parser.add_argument('--train_num', type=int, default=100)
args = parser.parse_args()
train_num = args.train_num

data_path = 'GraduationProject/Data'+args.folder
pic_path = 'GraduationProject/pics'+args.folder
mkdir(data_path)
mkdir(pic_path)

results = np.load(data_path+'/adqc_result_num{:d}.npy'.format(args.train_num), allow_pickle=True) # results是字典, 包含'train_pred', 'test_pred', 'train_loss', 'test_loss'
results = results.item()
train_pred = results['train_pred']
test_pred = results['test_pred']

# 对所有不同损失函数训练得到的数据训练过程中的保真度的变化
train_fide = results['train_fide']
test_fide = results['test_fide']

train_loss = results['train_loss']
test_loss = results['test_loss']
x = list(range(0, len(train_loss)))

print('train_num={:d}\ntrain_loss:{:.4e}\ttest_loss:{:.4e}\ntrain_fide:{:.4e}\ttest_fide:{:.4e}\n'\
      .format(train_num, train_loss[-1], test_loss[-1], train_fide[-1], test_fide[-1]))
# 打开一个文件，如果不存在则创建，如果存在则追加内容
with open(data_path+'/fin_loss_train_num.txt', 'a') as f:
    f.write('train_num={:d}\ntrain_loss:{:.4e}\ttest_loss:{:.4e}\ntrain_fide:{:.4e}\ttest_fide:{:.4e}\n'\
            .format(train_num, train_loss[-1], test_loss[-1], train_fide[-1], test_fide[-1]))

qc_mat = np.load(data_path+'/qc_mat_num{:d}.npy'.format(args.train_num))
qc_mat = tc.from_numpy(qc_mat)
evol_mat = np.load('GraduationProject/Data/evol_mat.npy')
evol_mat = tc.from_numpy(evol_mat)
# print('\nevol_mat.shape is', evol_mat.shape)

# gate fidelity
def gate_fidelity(E, U):
    n = E.shape[0]
    trace = tc.einsum('aa', U.T.conj() @ E)
    gate_fidelity = 1/(n*(n+1))*(n + tc.abs(trace)**2)
    return gate_fidelity

gate_fidelity = gate_fidelity(qc_mat, evol_mat)
with open(data_path+'/gate_fidelity.txt', 'a') as f:
    f.write("{:.6e}\t{:d}\n".format(gate_fidelity, train_num))
    pass

'''
# similarity
x = tc.arange(0, 2*tc.pi, 0.1)
# Vectorized calculation
phases = tc.exp(tc.complex(tc.zeros(x.shape), x))
qc_terms = qc_mat.cpu() * phases[:, None, None]
difference_matrix = qc_terms - evol_mat
term1 = tc.norm(difference_matrix, dim=(1, 2))
term2 = tc.norm(evol_mat)
similarity = 1- 0.5 * term1 / term2
# for i in range(len(x)): # to be improved
#     similarity[i] = (1 - 0.5*tc.norm(qc_mat.cpu()*tc.exp(tc.complex(0,x[i])) - evol_mat)/tc.norm(evol_mat))
print('\nsimilarity with different phase is', similarity)

try:
    sim_mat = np.load(data_path+'/sim_mat.npy')
    sim_mat = np.concatenate(sim_mat, similarity.numpy(), axis=0)
except:
    sim_mat = similarity.reshape((1, len(x))).numpy()

np.save(data_path+'/sim_mat.npy', sim_mat)

# with open(path+'/similarity_phase.txt', 'a') as f:
#     f.write("{:.6e}\t{:d}\n".format(similarity, args.train_num))
#     pass
'''

legends = []
plt.plot(x, train_loss, label='train loss')
plt.plot(x, test_loss, label= 'test loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig(pic_path+'/loss_num{:d}.svg'.format(args.train_num))
plt.close()

legends = []
plt.plot(x, train_fide, label='train fidelity')
plt.plot(x, test_fide, label= 'test fidelity')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('fidelity')
plt.savefig(pic_path+'/fidelity_num{:d}.svg'.format(args.train_num))
plt.close()

# legends = []
# plt.plot(x, gate_fidelity, label='gate_fidelity')
# plt.legend()
# plt.xlabel('epochs')
# plt.ylabel('gate_fidelity')
# plt.savefig(pic_path+'/gate_fidelity_num{:d}.svg'.format(args.train_num))
# plt.close()