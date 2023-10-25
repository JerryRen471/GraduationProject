import torch as tc
import numpy as np
import Library.BasicFun as bf

from torch.nn import MSELoss, NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from matplotlib import pyplot as plt
from Library.BasicFun import choose_device
from Library.MathFun import series_sin_cos
from Library.PhysModule import mag_from_states, mags_from_states, multi_mags_from_states, spin_operators

from Library.ADQC import ADQC_LatentGates

def split_time_series(data, length, device=None, dtype=tc.float32):
    samples, targets = list(), list()
    device = choose_device(device)
    for n in range(length, data.shape[0]):
        samples.append(data[n-length:n].clone())
        targets.append(data[n].clone())
    return tc.cat(samples, dim=0).to(device=device, dtype=dtype), tc.stack(targets, dim=0).to(device=device, dtype=dtype)

def fidelity(psi1:tc.Tensor, psi0:tc.Tensor):
    f = 0
    for i in range(psi1.shape[0]):
        psi0_ = psi0[i]
        psi1_ = psi1[i]
        x_pos = list(range(len(psi1_.shape)))
        y_pos = x_pos
        f_ = bf.tmul(psi1_.conj(), psi0_, x_pos, y_pos)
        f += (f_*f_.conj()).real
    f = f/psi1.shape[0]
    return f

def loss_fid(psi1:tc.Tensor, psi0:tc.Tensor):
    return 1 - fidelity(psi1, psi0)

def loss_average_mag(psi1:tc.Tensor, psi0:tc.Tensor):
    mag_diff = mag_from_states(psi1, device=psi1.device) - mag_from_states(psi0, device=psi0.device)
    loss = tc.norm(mag_diff)/tc.sqrt(psi1.shape[0])
    return loss

def loss_mags(psi1:tc.Tensor, psi0:tc.Tensor):
    mags_diff = mags_from_states(psi1, device=psi1.device) - mags_from_states(psi0, device=psi0.device)
    loss = tc.norm(mags_diff)/tc.sqrt(psi0.shape[1]*psi1.shape[0])
    return loss

def loss_multi_mags(psi1:tc.Tensor, psi0:tc.Tensor):
    spins = spin_operators('half', device=psi1.device)
    multi_mags_diff = multi_mags_from_states(psi1, spins, device=psi1.device) - multi_mags_from_states(psi0, spins, device=psi1.device)
    loss = tc.norm(multi_mags_diff)/tc.sqrt(psi0.shape[1]*psi1.shape[0])
    return loss

def choose_loss(loss_type:str):
    if loss_type == 'fidelity':
        loss = loss_fid
    elif loss_type == 'mag':
        loss = loss_average_mag
    elif loss_type == 'mags':
        loss = loss_mags
    elif loss_type == 'multi_mags':
        loss = loss_multi_mags
    else:
        raise ValueError("the loss_type should be \'fidelity\', \'mag\', \'mags\' or \'multi_mags\'")
    return loss

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

def train(qc:ADQC_LatentGates, data:dict, para:dict):
    """用data按照para的设定对qc进行训练

    返回的results是字典, 包含'train_pred', 'test_pred', 'train_loss', 'test_loss'
    """
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

    # num_train = int(data.shape[0] * (1-para['test_ratio']))
    trainset = data['train_set']
    train_lbs = data['train_lbs']
    testset = data['test_set']
    test_lbs = data['test_lbs']
    # trainset, train_lbs = split_time_series(
    #     data[:num_train], para['length'], para['device'], para['dtype'])
    # testset, test_lbs = split_time_series(
    #     data[num_train-para['length']:], para['length'], para['device'], para['dtype'])
    trainloader = DataLoader(TensorDataset(trainset, train_lbs), batch_size=para['batch_size'], shuffle=False)
    testloader = DataLoader(TensorDataset(testset, test_lbs), batch_size=para['batch_size'], shuffle=False)

    loss_train_rec = list()
    loss_test_rec = list()
    train_fide = list()
    test_fide = list()

    loss_fun = choose_loss(para['loss_type'])

    for t in range(para['it_time']):
        loss_tmp = 0.0
        train_fide_tmp = 0.0
        train_batches = 0
        test_batches = 0
        for n, (samples, lbs) in enumerate(trainloader):
            psi0 = samples
            psi1 = qc(psi0)
            loss = loss_fun(psi1, lbs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_tmp += loss.item()
            train_fide_tmp += fidelity(psi1, lbs)
            train_batches += 1

        if (t+1) % para['print_time'] == 0:
            loss_train_rec.append(loss_tmp / train_batches)
            train_fide.append(train_fide_tmp.cpu().detach().numpy() / train_batches)
            loss_tmp = 0.0
            test_fide_tmp = 0.0
            with tc.no_grad():
                for n, (samples, lbs) in enumerate(testloader):
                    psi0 = samples
                    psi1 = qc(psi0)
                    loss = loss_fun(psi1, lbs)
                    loss_tmp += loss.item()
                    test_fide_tmp += fidelity(psi1, lbs)
                    test_batches += 1
            loss_test_rec.append(loss_tmp / test_batches)
            test_fide.append(test_fide_tmp.cpu().detach().numpy() / test_batches)
            print('Epoch %i: train loss %g, test loss %g; train fidelity %g, test fidelity %g' %
                (t+1, loss_train_rec[-1], loss_test_rec[-1], train_fide[-1], test_fide[-1]))

    with tc.no_grad():
        results = dict()
        psi0 = trainset
        psi1 = qc(psi0)
        output = psi1
        output = output.data.to(device=data['train_set'].device)
        results['train_pred'] = output
        psi0 = testset
        psi1 = qc(psi0)
        output1 = psi1
        output1 = output1.data.to(device=data['test_set'].device)
        results['test_pred'] = output1
        results['train_loss'] = loss_train_rec
        results['test_loss'] = loss_test_rec
        results['train_fide'] = train_fide
        results['test_fide'] = test_fide
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
    para_adqc = {'depth': 4, 'loss_type':'mag'}  # ADQC量子门层数

    para_adqc = dict(para, **para_adqc)

    # 添加运行参数
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--time', type=float, default=1000)
    parser.add_argument('--pt_it', type=int, default=1000)
    parser.add_argument('--folder', type=str, default='')
    args = parser.parse_args()
    interval = args.pt_it
    folder = args.folder

    # 导入数据
    data = np.load("/data/home/scv7454/run/GraduationProject/Data/mag_loss/data_num100.npy", allow_pickle=True)
    data = data.item()

    # 训练ADQC实现预测
    # trainloader, testloader = DataProcess(data, para_adqc)
    qc = ADQC(para_adqc)
    qc, results_adqc, para_adqc = train(qc, data, para_adqc)

    # 保存数据
    np.save("/data/home/scv7454/run/GraduationProject/Data/mag_loss/adqc_result_num100", results_adqc)
