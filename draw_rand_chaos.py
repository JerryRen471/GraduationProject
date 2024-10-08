import os
import torch as tc
import numpy as np
from Library.BasicFun import mkdir
from Library import PhysModule as phy
from matplotlib import pyplot as plt
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--length', type=int, default=10)
parser.add_argument('--J', type=float, default=1)
parser.add_argument('--delta', type=float, default=1)
parser.add_argument('--theta', type=float, default=0)
parser.add_argument('--time_interval', type=float, default=0.01)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--evol_num', type=int, default=10)
parser.add_argument('--gen_type', type=str, default='nn')
parser.add_argument('--loss_type', type=str, default='multi_mags')
parser.add_argument('--folder', type=str, default='rand_init/')
parser.add_argument('--evol_mat_path', type=str, default="GraduationProject/Data/evol_mat.npy")
# parser.add_argument('--time_it', type=int, default="GraduationProject/Data/evol_mat.npy")
args = parser.parse_args()
evol_num = args.evol_num

def write_to_csv(data, csv_file_path, subset):
    """
    向CSV文件写入数据，可以指定接受的数据所对应的列。

    参数:
    data (dict): 要写入的数据字典，其中键为列名，值为对应的数据。
    csv_file_path (str): CSV文件的路径。
    """
    # 将数据转换为 DataFrame
    new_df = pd.DataFrame(data)

    # 检查文件是否存在
    if os.path.exists(csv_file_path):
        # 加载现有的 CSV 数据
        existing_data = pd.read_csv(csv_file_path)

        # 将新数据与现有数据合并
        combined_data = pd.concat([existing_data, new_df], ignore_index=True)
        # 去重，保留最后出现的行
        combined_data = combined_data.drop_duplicates(
            subset=subset, keep='last'
        )
        combined_data = combined_data.sort_values(subset)

    else:
        # 文件不存在，直接使用新数据
        combined_data = new_df
    
    # 保存更新后的数据到 CSV 文件
    combined_data.to_csv(csv_file_path, index=False)

data_path = 'GraduationProject/Data'+args.folder
pic_path = 'GraduationProject/pics'+args.folder
evol_mat_path = args.evol_mat_path
mkdir(data_path)
mkdir(pic_path)

results = np.load(data_path+'/adqc_result_sample_{:d}_evol_{:d}.npy'.format(args.sample_num, args.evol_num), allow_pickle=True) # results是字典, 包含'train_pred', 'test_pred', 'train_loss', 'test_loss'
os.remove(data_path+'/adqc_result_sample_{:d}_evol_{:d}.npy'.format(args.sample_num, args.evol_num))
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
      .format(evol_num, train_loss[-1], test_loss[-1], train_fide[-1], test_fide[-1]))
# 打开一个文件，如果不存在则创建，如果存在则追加内容
with open(data_path+'/fin_loss_train_num.txt', 'a') as f:
    f.write('train_num={:d}\ntrain_loss:{:.4e}\ttest_loss:{:.4e}\ntrain_fide:{:.4e}\ttest_fide:{:.4e}\n'\
            .format(evol_num, train_loss[-1], test_loss[-1], train_fide[-1], test_fide[-1]))

qc_mat = np.load(data_path+'/qc_mat_sample_{:d}_evol_{:d}.npy'.format(args.sample_num, args.evol_num))
qc_mat = tc.from_numpy(qc_mat)
evol_mat = np.load(evol_mat_path)
evol_mat = tc.from_numpy(evol_mat)
print('\nevol_mat.shape is', evol_mat.shape)
os.remove(data_path+'/qc_mat_sample_{:d}_evol_{:d}.npy'.format(args.sample_num, args.evol_num))
os.remove(evol_mat_path)
print('\nqc_mat.shape is', qc_mat.shape)

# gate fidelity
def gate_fidelity(E:tc.Tensor, U:tc.Tensor):
    n = E.shape[0]
    trace = tc.einsum('aa', U.T.conj() @ E)
    gate_fidelity = 1/(n*(n+1))*(n + tc.abs(trace)**2)
    return gate_fidelity
gate_fidelity = gate_fidelity(qc_mat, evol_mat)
with open(data_path+'/gate_fidelity.txt', 'a') as f:
    f.write("{:.6e}\t{:d}\n".format(gate_fidelity, evol_num))
    pass

# gate similarity
def similarity(E:tc.Tensor, U:tc.Tensor):
    '''
    E: circuit
    U: real process
    '''
    a = tc.norm(E - U)
    b = 2 * tc.norm(U)
    s = 1 - a/b
    return s
similarity = similarity(qc_mat, evol_mat)
with open(data_path+'/similarity.txt', 'a') as f:
    f.write("{:.6e}\t{:d}\n".format(similarity, evol_num))
    pass

def spectrum(mat:tc.Tensor):
    energy = tc.log(tc.linalg.eigvals(mat))/1.j
    energy = energy.real
    energy, ind = tc.sort(energy)
    return energy
    
qc_energy = spectrum(qc_mat)
evol_energy = spectrum(evol_mat)
diff = tc.sqrt(tc.sum(tc.square(qc_energy-evol_energy)))/qc_energy.shape[0]
with open(data_path+'/spectrum_diff.txt', 'a') as f:
    f.write("{:.6e}\t{:d}\n".format(diff, evol_num))
    pass

if args.gen_type[0] == 'd':
    train_set_type = 'product'
elif args.gen_type[0] == 'n':
    train_set_type = 'non_product'

data = {
    'length': [int(args.length)],
    'J': [float(args.J)],
    'delta': [float(args.delta)],
    'theta': [float(args.theta)],
    'time_interval': [args.time_interval],
    'evol_num': [int(args.evol_num)],
    'sample_num': [int(args.sample_num)],
    'train_set_type': [train_set_type],
    'loss': [args.loss_type],
    'gate_fidelity': [float(gate_fidelity)],
    'spectrum_diff': [float(diff)],
    'train_loss': [train_loss[-1]],
    'test_loss': [test_loss[-1]],
    'similarity': [float(similarity)]
}

subset=['J', 'delta', 'theta', 'train_set_type', 'loss', 'length', 'time_interval', 'evol_num', 'sample_num']
write_to_csv(data, csv_file_path='GraduationProject/Data/XXZ_inhomo.csv', subset=subset)

legends = []
plt.plot(x, train_loss, label='train loss')
plt.plot(x, test_loss, label= 'test loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig(pic_path+'/loss_num{:d}.svg'.format(args.evol_num))
plt.close()

legends = []
plt.plot(x, train_fide, label='train fidelity')
plt.plot(x, test_fide, label= 'test fidelity')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('fidelity')
plt.savefig(pic_path+'/fidelity_num{:d}.svg'.format(args.evol_num))
plt.close()

legends = []
plt.plot(qc_energy, label='qc')
plt.plot(evol_energy, label='Trotter')
plt.legend()
plt.xlabel('n')
plt.ylabel('E*t')
plt.savefig(pic_path+'/spectrum_num{:d}.svg'.format(args.evol_num))
plt.close()


# legends = []
# plt.plot(x, gate_fidelity, label='gate_fidelity')
# plt.legend()
# plt.xlabel('epochs')
# plt.ylabel('gate_fidelity')
# plt.savefig(pic_path+'/gate_fidelity_num{:d}.svg'.format(args.evol_num))
# plt.close()

