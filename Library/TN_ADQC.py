from math import sqrt
import torch as tc
import numpy as np
import Library.BasicFun as bf

from torch.nn import MSELoss, NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD
from matplotlib import pyplot as plt
from Library.BasicFun import choose_device
from Library.MathFun import series_sin_cos
from Library.PhysModule import mag_from_states, mags_from_states, multi_mags_from_states, combined_mags, spin_operators, two_body_ob
from Library.TensorNetwork import copy_from_mps_pack, inner_mps_pack
from Library import TensorNetwork as TN

from Library.ADQC import ADQC_LatentGates, Variational_Quantum_Circuit

def split_time_series(data, length, device=None, dtype=tc.float32):
    samples, targets = list(), list()
    device = choose_device(device)
    for n in range(length, data.shape[0]):
        samples.append(data[n-length:n].clone())
        targets.append(data[n].clone())
    return tc.cat(samples, dim=0).to(device=device, dtype=dtype), tc.stack(targets, dim=0).to(device=device, dtype=dtype)

def fidelity(psi1:TN.TensorTrain_pack, psi0:TN.TensorTrain_pack):
    fides = TN.inner_mps_pack(psi1, psi0)
    f = tc.mean(fides)
    return f

def loss_fid(psi1:TN.TensorTrain_pack, psi0:TN.TensorTrain_pack):
    return 1 - fidelity(psi1, psi0)

def loss_zx_mags(psi1:TN.TensorTrain_pack, psi0:TN.TensorTrain_pack):
    op = spin_operators('half', device=psi1.device)
    spins = [op['sx'], op['sz']]
    multi_mags_diff = TN.multi_mags_from_mps_pack(psi1, spins) - TN.multi_mags_from_mps_pack(psi0, spins)
    loss = tc.norm(multi_mags_diff)/sqrt(psi0.node_list[0].shape[0])
    return loss

def loss_multi_mags(psi1:TN.TensorTrain_pack, psi0:TN.TensorTrain_pack):
    op = spin_operators('half', device=psi1.device, dtp=psi1.dtype)
    spins = [op['sx'], op['sy'], op['sz']]
    ################################### CHECK!!!!! #########################################
    multi_mags_diff = TN.multi_mags_from_mps_pack(psi1, spins) - TN.multi_mags_from_mps_pack(psi0, spins)
    loss = tc.sum(tc.pow(multi_mags_diff, 2))
    return loss

def log_loss_multi_mags(psi1:TN.TensorTrain_pack, psi0:TN.TensorTrain_pack):
    op = spin_operators('half', device=psi1.device)
    spins = [op['sx'], op['sy'], op['sz']]
    multi_mags_diff = TN.multi_mags_from_mps_pack(psi1, spins) - TN.multi_mags_from_mps_pack(psi0, spins)
    loss = tc.log(tc.norm(multi_mags_diff)/sqrt(psi0.node_list[0].shape[0]))
    return loss

def choose_loss(loss_type:str):
    if loss_type == 'fidelity':
        loss = loss_fid
    elif loss_type == 'xz_mags':
        loss = loss_zx_mags
    elif loss_type == 'multi_mags':
        loss = loss_multi_mags
    elif loss_type == 'log_multi_mags':
        loss = log_loss_multi_mags
    else:
        raise ValueError("the loss_type should be fidelity, xz_mags, multi_mags, log_multi_mags")
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
    para0['ini_way'] = 'identity'  # 线路初始化策略
    para0['lr'] = 2e-3  # 初始学习率
    para0['it_time'] = 1000  # 迭代次数
    para0['print_time'] = 10  # 打印间隔
    para0['device'] = 'cuda:0'
    para0['dtype'] = tc.complex128
    para0['state_type'] = 'tensor'

    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数
    para['device'] = bf.choose_device(para['device'])

    qc = ADQC_LatentGates(state_type=para['state_type'], lattice=para['lattice'], num_q=para['length_in'], depth=para['depth'], ini_way=para['ini_way'],
                          device=para['device'], dtype=para['dtype'])
    qc.single_state = False  # 切换至多个态演化模式
    return qc

def VQC(para=None):
    para0 = dict()  # 默认参数
    para0['test_ratio'] = 0.2  # 将部分样本划为测试集
    para0['length_in'] = 10  # 数据样本维数
    para0['length_out'] = 10
    para0['batch_size'] = 200  # 批次大小
    para0['feature_map'] = 'cossin'  # 特征映射
    para0['lattice'] = 'stair'  # ADQC链接形式（brick或stair）
    para0['depth'] = 4  # ADQC层数
    para0['ini_way'] = 'identity'  # 线路初始化策略
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

    qc = Variational_Quantum_Circuit(state_type=para['state_type'], lattice=para['lattice'], num_q=para['length_in'], depth=para['depth'], ini_way=para['ini_way'],
                          device=para['device'], dtype=para['dtype'])
    qc.single_state = False  # 切换至多个态演化模式
    return qc

from torchviz import make_dot

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
    para0['lr'] = 2e-2  # 初始学习率
    para0['it_time'] = 1000  # 迭代次数
    para0['print_time'] = 10  # 打印间隔
    para0['device'] = 'cuda:0'
    para0['dtype'] = tc.complex128
    para0['recurrent_time'] = 1
    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数

    optimizer = Adam(qc.parameters(), lr=para['lr'])

    # num_train = int(data.shape[0] * (1-para['test_ratio']))
    train_set = data['train_set']
    train_lbs = data['train_lbs']
    test_set = data['test_set']
    test_lbs = data['test_lbs']
    # train_set, train_lbs = split_time_series(
    #     data[:num_train], para['length'], para['device'], para['dtype'])
    # test_set, test_lbs = split_time_series(
    #     data[num_train-para['length']:], para['length'], para['device'], para['dtype'])
    # trainloader = DataLoader(TensorDataset(train_set, train_lbs), batch_size=para['batch_size'], shuffle=False)
    # testloader = DataLoader(TensorDataset(test_set, test_lbs), batch_size=para['batch_size'], shuffle=False)

    loss_train_rec = list()
    loss_test_rec = list()
    train_fide = list()
    test_fide = list()

    loss_fun = choose_loss(para['loss_type'])

    for t in range(para['it_time']):
        print(t)
        loss_tmp = 0.0
        train_fide_tmp = 0.0
        train_batches = 0
        test_batches = 0
        psi0 = copy_from_mps_pack(train_set)
        for _ in range(para['recurrent_time']):
            psi0 = qc(psi0)
        # psi1 = psi0
        loss = loss_fun(psi0, train_lbs)
        
        assert tc.isnan(loss).sum() == 0, print(loss)
        optimizer.zero_grad()
        # make_dot(loss).render("computational_graph", format="svg")
        loss.backward() # backward过程中梯度是none

        # 2. 如果loss不是nan,那么说明forward过程没问题，可能是梯度爆炸，所以用梯度裁剪试试
        tc.nn.utils.clip_grad_norm_(qc.parameters(), 3, norm_type=2)

        # 3.1 在step之前，判断参数是不是nan, 如果不是判断step之后是不是nan
        # assert tc.isnan(qc.layers).sum() == 0, print(qc.layers)
        optimizer.step()
        loss_tmp += loss.item()
        fide = fidelity(psi0, train_lbs)
        fide_sr = fide.abs()
        train_fide_tmp += fide_sr
        train_batches += 1

        if (t+1) % para['print_time'] == 0:
            loss_train_rec.append(loss_tmp / train_batches)
            train_fide.append(train_fide_tmp.cpu().detach().numpy() / train_batches)
            loss_tmp = 0.0
            test_fide_tmp = 0.0
            with tc.no_grad():
                psi0 = copy_from_mps_pack(test_set)
                for _ in range(para['recurrent_time']):
                    psi0 = qc(psi0)
                psi1 = psi0
                loss = loss_fun(psi1, test_lbs)
                loss_tmp += loss.item()
                fide = fidelity(psi1, test_lbs)
                fide_sr = fide.abs()
                test_fide_tmp += fide_sr
                test_batches += 1
            loss_test_rec.append(loss_tmp / test_batches)
            test_fide.append(test_fide_tmp.cpu().detach().numpy() / test_batches)
            print(loss_train_rec[-1])
            print('Epoch {}: train loss {}, test loss {}; train fidelity {}, test fidelity {}'.format(\
                t+1, loss_train_rec[-1], loss_test_rec[-1], train_fide[-1], test_fide[-1]))

    with tc.no_grad():
        results = dict()
        psi0 = train_set
        for _ in range(para['recurrent_time']):
            psi0 = qc(psi0)
        output = psi0
        # output = output.data.to(device=data['train_set'].device)
        results['train_pred'] = output
        psi0 = test_set
        for _ in range(para['recurrent_time']):
            psi0 = qc(psi0)
        output1 = psi0
        # output1 = output1.data.to(device=data['test_set'].device)
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
            'ini_way': 'identity',
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
    data = tc.load("/data/home/scv7454/run/GraduationProject/Data/mag_loss/data_num100.pt", allow_pickle=True)
    data = data.item()

    # 训练ADQC实现预测
    # trainloader, testloader = DataProcess(data, para_adqc)
    qc = ADQC(para_adqc)
    qc, results_adqc, para_adqc = train(qc, data, para_adqc)

    # 保存数据
    tc.save("/data/home/scv7454/run/GraduationProject/Data/mag_loss/adqc_result_num100.pt", results_adqc)
