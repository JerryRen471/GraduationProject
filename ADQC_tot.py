import torch as tc
import numpy as np
import Library.BasicFun as bf

from torch.nn import MSELoss, NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from matplotlib import pyplot as plt
from Library.BasicFun import choose_device
from Library.MathFun import series_sin_cos
from Algorithms import ADQC_algo, LSTM_algo

from Library.ADQC import ADQC_LatentGates

def split_time_series(data, length, device=None, dtype=tc.float32):
    samples, targets = list(), list()
    device = choose_device(device)
    for n in range(length, data.shape[0]):
        samples.append(data[n-length:n].clone())
        targets.append(data[n].clone())
    return tc.cat(samples, dim=0).to(device=device, dtype=dtype), tc.stack(targets, dim=0).to(device=device, dtype=dtype)

def fidelity(psi1, psi0):
    f = 0
    for i in range(psi1.shape[0]):
        psi0_ = psi0[i]
        psi1_ = psi1[i]
        x_pos = list(range(len(psi1_.shape)))
        y_pos = x_pos
        f_ = bf.tmul(psi1_.conj(), psi0_, x_pos, y_pos)
        f += tc.sqrt((f_*f_.conj()).real)
    f = f/psi1.shape[0]
    return f

def ADQC(para=None):
    para0 = dict()  # 默认参数
    para0['test_ratio'] = 0.2  # 将部分样本划为测试集
    para0['length_in'] = 10  # 数据样本维数
    para0['length_out'] = 10
    para0['batch_size'] = 200  # 批次大小
    para0['feature_map'] = 'cossin'  # 特征映射
    para0['lattice'] = 'brick'  # ADQC链接形式（brick或stair）
    para0['depth'] = 4  # ADQC层数
    para0['ini_way'] = 'random'  # 线路初始化策略
    para0['lr'] = 2e-3  # 初始学习率
    para0['it_time'] = 1000  # 迭代次数
    para0['print_time'] = 10  # 打印间隔
    para0['device'] = 'cuda:0'
    para0['dtype'] = tc.complex128

    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数
    para['device'] = bf.choose_device(para['device'])

    qc = ADQC_LatentGates(lattice=para['lattice'], num_q=para['length_in'], depth=para['depth'], ini_way=para['ini_way'],
                          device=para['device'], dtype=para['dtype'])
    qc.single_state = False  # 切换至多个态演化模式
    return qc

def train(qc, data, para):
    para0 = dict()  # 默认参数
    para0['test_ratio'] = 0.2  # 将部分样本划为测试集
    para0['length_in'] = 10  # 数据样本维数
    para0['length_out'] = 10
    para0['batch_size'] = 200  # 批次大小
    para0['feature_map'] = 'cossin'  # 特征映射
    para0['lattice'] = 'brick'  # ADQC链接形式（brick或stair）
    para0['depth'] = 4  # ADQC层数
    para0['ini_way'] = 'random'  # 线路初始化策略
    para0['lr'] = 2e-3  # 初始学习率
    para0['it_time'] = 1000  # 迭代次数
    para0['print_time'] = 10  # 打印间隔
    para0['device'] = 'cuda:0'
    para0['dtype'] = tc.complex128
    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数

    optimizer = Adam(qc.parameters(), lr=para['lr'])

    num_train = int(data.shape[0] * (1-para['test_ratio']))
    trainset, train_lbs = split_time_series(
        data[:num_train], para['length'], para['device'], para['dtype'])
    testset, test_lbs = split_time_series(
        data[num_train-para['length']:], para['length'], para['device'], para['dtype'])
    trainloader = DataLoader(TensorDataset(trainset, train_lbs), batch_size=para['batch_size'], shuffle=False)
    testloader = DataLoader(TensorDataset(testset, test_lbs), batch_size=para['batch_size'], shuffle=False)

    loss_train_rec = list()
    loss_test_rec = list()

    for t in range(para['it_time']):
        loss_tmp = 0.0
        for n, (samples, lbs) in enumerate(trainloader):
            psi0 = samples
            psi1 = qc(psi0)
            loss = 1 - fidelity(psi1, lbs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_tmp += loss.item() * samples.shape[0]

        if (t+1) % para['print_time'] == 0:
            loss_train_rec.append(1e-4 * loss_tmp / (1-para0['test_ratio']))
            loss_tmp = 0.0
            with tc.no_grad():
                for n, (samples, lbs) in enumerate(testloader):
                    psi0 = samples
                    psi1 = qc(psi0)
                    loss = 1 - fidelity(psi1, lbs)
                    loss_tmp += loss.item() * samples.shape[0]
            loss_test_rec.append(1e-4 * loss_tmp / para0['test_ratio'])
            print('Epoch %i: train loss %g, test loss %g' %
                (t+1, loss_train_rec[-1], loss_test_rec[-1]))

    with tc.no_grad():
        results = dict()
        psi0 = trainset
        psi1 = qc(psi0)
        output = psi1
        output = tc.cat([data[:para['length']].to(dtype=output.dtype), output.to(device=data.device)], dim=0)
        results['train_pred'] = output.data
        psi0 = testset
        psi1 = qc(psi0)
        output1 = psi1
        output1 = output1.data.to(device=data.device)
        results['test_pred'] = output1
        results['train_loss'] = loss_train_rec
        results['test_loss'] = loss_test_rec
    return qc, results, para

if __name__ == '__main__':
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

    # 添加运行参数
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--time', type=float, default=1000)
    parser.add_argument('--pt_it', type=int, default=100)
    args = parser.parse_args()
    interval = args.pt_it
    time_tot = args.time

    # 导入数据
    states = np.load('GraduationProject/Data/states_dt{:d}_tot{:.0f}.npy'.format(interval, time_tot), allow_pickle=True)
    print(states[0].shape)
    data_list = []
    time_diff = 10
    for i in range(time_diff):
        data = tc.stack(list(tc.from_numpy(states[:int((i + 1)*time_tot/time_diff)])))
        print(data.shape)

        # 训练ADQC实现预测
        # trainloader, testloader = DataProcess(data, para_adqc)
        qc = ADQC(para_adqc)
        qc, results_adqc, para_adqc = \
            train(qc, data, para_adqc)
        output_adqc = tc.cat([results_adqc['train_pred'],
                            results_adqc['test_pred']], dim=0)

        # 保存数据
        tc.save(qc, 'GraduationProject/Data/qc_dt{:d}_tot{:.0f}.pth'.format(interval, (i+1)*time_tot/time_diff))
        np.save('GraduationProject/Data/output_adqc_dt{:d}_tot{:.0f}'.format(interval, (i+1)*time_tot/time_diff), output_adqc)