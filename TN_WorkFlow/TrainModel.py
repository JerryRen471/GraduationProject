from copy import deepcopy
from math import sqrt
from sympy import AssumptionsContext, assuming
import torch as tc
import numpy as np
import sys
import gc
from memory_profiler import profile
import psutil
import os
from pympler import asizeof

sys.path.append('/data/home/scv7454/run/GraduationProject')

import Library.BasicFun as bf
from Library.TensorNetwork import TensorTrain_pack, inner_mps_pack, multi_mags_from_mps_pack

from torch.nn import MSELoss, NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from matplotlib import pyplot as plt
from Library.BasicFun import choose_device
from Library.MathFun import series_sin_cos
from Library.PhysModule import mag_from_states, mags_from_states, multi_mags_from_states, combined_mags, spin_operators, two_body_ob, sample_classical_shadow

from Library.ADQC import ADQC_LatentGates, Variational_Quantum_Circuit

def split_time_series(data, length, device=None, dtype=tc.float32):
    samples, targets = list(), list()
    device = choose_device(device)
    for n in range(length, data.shape[0]):
        samples.append(data[n-length:n].clone())
        targets.append(data[n].clone())
    return tc.cat(samples, dim=0).to(device=device, dtype=dtype), tc.stack(targets, dim=0).to(device=device, dtype=dtype)

def fidelity(psi1, psi0):
    """
    Calculate fidelity between two quantum states.
    Works with both regular tensors and TensorTrain_pack objects.
    """
    if isinstance(psi1, TensorTrain_pack) and isinstance(psi0, TensorTrain_pack):
        # For TensorTrain_pack objects, use inner_mps_pack
        fides = inner_mps_pack(psi1, psi0)
        f = tc.mean(tc.abs(fides))
        return f
    else:
        # For regular tensors, use the original method
        psi0_ = psi0.reshape(psi0.shape[0], -1)
        psi1_ = psi1.reshape(psi1.shape[0], -1)
        fides = tc.abs(tc.einsum('ab,ab->a', psi1_.conj(), psi0_))
        f = tc.mean(fides)
        return f

def loss_fid(psi1, psi0):
    return 1 - fidelity(psi1, psi0)

def loss_average_mag(psi1, psi0):
    if isinstance(psi1, TensorTrain_pack) and isinstance(psi0, TensorTrain_pack):
        # For TensorTrain_pack objects, calculate magnetization directly
        op = spin_operators('half', device=psi1.device)
        spins = [op['sz']]
        mag1 = multi_mags_from_mps_pack(psi1, spins)
        mag0 = multi_mags_from_mps_pack(psi0, spins)
        mag_diff = mag1 - mag0
        loss = tc.norm(mag_diff)/sqrt(psi1.number)
        return loss
    else:
        # For regular tensors, use the original method
        mag_diff = mag_from_states(psi1, device=psi1.device) - mag_from_states(psi0, device=psi0.device)
        loss = tc.norm(mag_diff)/sqrt(psi1.shape[0])
        return loss

def loss_mags(psi1, psi0):
    if isinstance(psi1, TensorTrain_pack) and isinstance(psi0, TensorTrain_pack):
        # For TensorTrain_pack objects, calculate magnetization directly
        op = spin_operators('half', device=psi1.device)
        spins = [op['sx'], op['sy'], op['sz']]
        mags1 = multi_mags_from_mps_pack(psi1, spins)
        mags0 = multi_mags_from_mps_pack(psi0, spins)
        mags_diff = mags1 - mags0
        loss = tc.norm(mags_diff)/sqrt(psi0.length*psi1.number)
        return loss
    else:
        # For regular tensors, use the original method
        mags_diff = mags_from_states(psi1, device=psi1.device) - mags_from_states(psi0, device=psi0.device)
        loss = tc.norm(mags_diff)/sqrt(psi0.shape[1]*psi1.shape[0])
        return loss

def loss_zx_mags(psi1, psi0):
    op = spin_operators('half', device=psi1.device)
    spins = [op['sx'], op['sz']]
    
    if isinstance(psi1, TensorTrain_pack) and isinstance(psi0, TensorTrain_pack):
        # For TensorTrain_pack objects, calculate magnetization directly
        multi_mags1 = multi_mags_from_mps_pack(psi1, spins)
        multi_mags0 = multi_mags_from_mps_pack(psi0, spins)
        multi_mags_diff = multi_mags1 - multi_mags0
        loss = tc.norm(multi_mags_diff)/sqrt(psi0.length*psi1.number)
        return loss
    else:
        # For regular tensors, use the original method
        multi_mags_diff = multi_mags_from_states(psi1, spins, device=psi1.device) - multi_mags_from_states(psi0, spins, device=psi1.device)
        loss = tc.norm(multi_mags_diff)/sqrt(psi0.shape[1]*psi1.shape[0])
        return loss

def loss_multi_mags(psi1, psi0):
    op = spin_operators('half', device=psi1.device)
    spins = [op['sx'], op['sy'], op['sz']]
    
    if isinstance(psi1, TensorTrain_pack) and isinstance(psi0, TensorTrain_pack):
        # For TensorTrain_pack objects, calculate magnetization directly
        multi_mags1 = multi_mags_from_mps_pack(psi1, spins)
        multi_mags0 = multi_mags_from_mps_pack(psi0, spins)
        multi_mags_diff = multi_mags1 - multi_mags0
        loss = tc.norm(multi_mags_diff)/sqrt(psi0.length*psi1.number)
        return loss
    else:
        # For regular tensors, use the original method
        multi_mags_diff = multi_mags_from_states(psi1, spins, device=psi1.device) - multi_mags_from_states(psi0, spins, device=psi1.device)
        loss = tc.norm(multi_mags_diff)/sqrt(psi0.shape[1]*psi1.shape[0])
        return loss

def log_loss_multi_mags(psi1, psi0):
    op = spin_operators('half', device=psi1.device)
    spins = [op['sx'], op['sy'], op['sz']]
    
    if isinstance(psi1, TensorTrain_pack) and isinstance(psi0, TensorTrain_pack):
        # For TensorTrain_pack objects, calculate magnetization directly
        multi_mags1 = multi_mags_from_mps_pack(psi1, spins)
        multi_mags0 = multi_mags_from_mps_pack(psi0, spins)
        multi_mags_diff = multi_mags1 - multi_mags0
        loss = tc.log(tc.norm(multi_mags_diff)/sqrt(psi0.length*psi1.number))
        return loss
    else:
        # For regular tensors, use the original method
        multi_mags_diff = multi_mags_from_states(psi1, spins, device=psi1.device) - multi_mags_from_states(psi0, spins, device=psi1.device)
        loss = tc.log(tc.norm(multi_mags_diff)/sqrt(psi0.shape[1]*psi1.shape[0]))
        return loss

def loss_combined_mags(psi1, psi0):
    op = spin_operators('half', device=psi1.device)
    spins = [op['sx'], op['sy'], op['sz']]
    combined_mags_diff = combined_mags(psi1, spins) - combined_mags(psi0, spins)
    loss = tc.norm(combined_mags_diff)/sqrt(psi1.shape[0])
    return loss

def loss_complete_combined_mags(psi1:tc.Tensor, psi0:tc.Tensor):
    diff = two_body_ob(psi1) - two_body_ob(psi0)
    loss = tc.norm(diff)/sqrt(psi1.shape[0])
    return loss

def loss_norm(psi1, psi0):
    if isinstance(psi1, TensorTrain_pack) and isinstance(psi0, TensorTrain_pack):
        # For TensorTrain_pack objects, we need to calculate the norm difference
        # This is a placeholder - you may need to implement a proper norm calculation for TensorTrain_pack
        N = psi1.number
        # Calculate the difference between the two MPS packs
        # This is a simplified approach and may need to be refined
        diff = inner_mps_pack(psi1, psi1) - inner_mps_pack(psi0, psi0)
        loss = 1/N * tc.norm(diff)**2
        return loss
    else:
        # For regular tensors, use the original method
        N = psi1.shape[0]
        loss = 1/N * tc.norm(psi0 - psi1)**2
        return loss

def loss_sample_fide(psi_real:tc.Tensor, psi_pred:tc.Tensor, num_basis:int=30, k=10, num_sample:int=100000):
    assert psi_pred.shape == psi_real.shape
    assert psi_pred.device == psi_real.device
    assert psi_pred.dtype == psi_real.dtype

    psi_pred.requires_grad_()
    num_states = psi_pred.shape[0]
    n_qubit = len(psi_real.shape) - 1
    avg_rho = 0
    number = num_basis * k
    avg_fidelity = 0
    # rho = tc.einsum('a, b -> ab', state.reshape(-1), state.reshape(-1).conj())
    for i in range(number):
        rho_sample = sample_classical_shadow(aim_states=psi_real, n_qubit=n_qubit, num_sample=num_sample)
        avg_rho = avg_rho + rho_sample
        if i % num_basis == num_basis-1:
            avg_rho = avg_rho / num_basis
            fidelity = tc.einsum('na, nab, nb ->n', psi_pred.reshape([num_states,-1]).conj(), avg_rho, psi_pred.reshape([num_states,-1]))
            avg_fidelity = avg_fidelity + fidelity
            avg_rho = 0
    avg_fidelity = avg_fidelity / k
    loss = tc.norm(tc.ones_like(avg_fidelity) - avg_fidelity)
    return loss

def choose_loss(loss_type:str):
    if loss_type == 'fidelity':
        loss = loss_fid
    elif loss_type == 'sample_fidelity':
        loss = loss_sample_fide
    elif loss_type == 'mag':
        loss = loss_average_mag
    elif loss_type == 'mags':
        loss = loss_mags
    elif loss_type == 'xz_mags':
        loss = loss_zx_mags
    elif loss_type == 'multi_mags':
        loss = loss_multi_mags
    elif loss_type == 'log_multi_mags':
        loss = log_loss_multi_mags
    elif loss_type == 'combined_mags':
        loss = loss_combined_mags
    elif loss_type == 'complete':
        loss = loss_complete_combined_mags
    elif loss_type == 'loss_norm':
        loss = loss_norm
    else:
        raise ValueError("the loss_type should be \'fidelity\', \'mag\', \'mags\', \'multi_mags\', \'log_multi_mags\', \'combined_mags\', \'complete\', \'loss_norm\'")
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

    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数
    para['device'] = bf.choose_device(para['device'])

    qc = ADQC_LatentGates(state_type='mps', lattice=para['lattice'], num_q=para['length_in'], depth=para['depth'], ini_way=para['ini_way'],
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

    qc = Variational_Quantum_Circuit(state_type='mps', lattice=para['lattice'], num_q=para['length_in'], depth=para['depth'], ini_way=para['ini_way'],
                          device=para['device'], dtype=para['dtype'])
    qc.single_state = False  # 切换至多个态演化模式
    return qc

def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if tc.cuda.is_available():
        return tc.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    return 0

def get_gpu_memory_reserved():
    """Get current GPU memory reserved in MB"""
    if tc.cuda.is_available():
        return tc.cuda.memory_reserved() / 1024 / 1024  # Convert to MB
    return 0

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if tc.cuda.is_available():
        tc.cuda.empty_cache()
    gc.collect()

def get_tensor_memory_usage(tensor):
    """Get memory usage of a tensor in MB"""
    if tensor is None:
        return 0
    if isinstance(tensor, tc.Tensor):
        return tensor.element_size() * tensor.nelement() / 1024 / 1024  # Convert to MB
    return 0

def get_variable_memory_usage(var, unit='MB'):
    """Get memory usage of any variable
    
    Args:
        var: Any Python variable
        unit: 'B' for bytes, 'KB' for kilobytes, 'MB' for megabytes
        
    Returns:
        Memory usage in specified unit
    """
    if var is None:
        return 0
        
    # Handle PyTorch tensors
    if isinstance(var, tc.Tensor):
        memory = var.element_size() * var.nelement()
    # Handle numpy arrays
    elif isinstance(var, np.ndarray):
        memory = var.nbytes
    # Handle other Python objects
    else:
        memory = asizeof.asizeof(var)
    
    # Convert to requested unit
    if unit == 'KB':
        return memory / 1024
    elif unit == 'MB':
        return memory / 1024 / 1024
    elif unit == 'GB':
        return memory / 1024 / 1024 / 1024
    else:  # bytes
        return memory

def train(qc, data:dict, para:dict):
    """用data按照para的设定对qc进行训练

    返回的results是字典, 包含'train_pred', 'test_pred', 'train_loss', 'test_loss'
    """
    para0 = dict()  # 默认参数
    para0['batch_size'] = 200  # 批次大小
    para0['loss_type'] = 'fidelity'  # 损失函数类型
    para0['lr'] = 2e-2  # 初始学习率
    para0['it_time'] = 1000  # 迭代次数
    para0['print_time'] = 20  # 打印间隔
    para0['recurrent_time'] = 1 # 循环次数
    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数

    optimizer = Adam(qc.parameters(), lr=para['lr'])

    train_set = data['train_set']
    train_lbs = data['train_label']
    test_set = data['test_set']
    test_lbs = data['test_label']

    loss_train_rec = list()
    loss_test_rec = list()
    train_fide = list()
    test_fide = list()
    gpu_memory_usage = list()
    gpu_memory_reserved = list()

    loss_fun = choose_loss(para['loss_type'])

    for t in range(para['it_time']):
        # Training step
        print(f'Training step: {t}')
        clear_gpu_memory()
        
        temp = get_gpu_memory_usage()
        psi0 = deepcopy(train_set)
        print(f'The cost of deepcopy train_set is {get_gpu_memory_usage() - temp:.2f} MB')
        print(f'GPU Memory - Allocated: {get_gpu_memory_usage():.2f} MB, Reserved: {get_gpu_memory_reserved():.2f} MB')
        for _ in range(para['recurrent_time']):
            psi0 = qc(psi0)
        psi1 = psi0
        loss = loss_fun(psi1, deepcopy(train_lbs))

        print(f'-----Before Backward-----')
        print(f'GPU Memory - Allocated: {get_gpu_memory_usage():.2f} MB, Reserved: {get_gpu_memory_reserved():.2f} MB')
        print(f'Loss memory: {get_variable_memory_usage(loss):.2f} MB')
        print(f'psi0 memory: {get_variable_memory_usage(psi0):.2f} MB')
        print(f'psi1 memory: {get_variable_memory_usage(psi1):.2f} MB')
        print(f'Optimizer memory: {get_variable_memory_usage(optimizer):.2f} MB')
        print(f'Model memory: {get_variable_memory_usage(qc):.2f} MB')

        loss.backward(retain_graph=False)

        print(f'-----After Backward-----')
        print(f'GPU Memory - Allocated: {get_gpu_memory_usage():.2f} MB, Reserved: {get_gpu_memory_reserved():.2f} MB')
        print(f'Loss memory: {get_variable_memory_usage(loss):.2f} MB')
        print(f'Gradients memory: {get_variable_memory_usage([p.grad for p in qc.parameters() if p.grad is not None]):.2f} MB')

        optimizer.step()
        optimizer.zero_grad()
        clear_gpu_memory()

        print(f'-----After Update-----')
        print(f'GPU Memory - Allocated: {get_gpu_memory_usage():.2f} MB, Reserved: {get_gpu_memory_reserved():.2f} MB')
        gpu_memory_usage.append(get_gpu_memory_usage())
        gpu_memory_reserved.append(get_gpu_memory_reserved())
        
        # with tc.no_grad():
        if (t+1) % para['print_time'] == 0:
            # Evaluation step
            with tc.no_grad():
                # Train evaluation
                fide = fidelity(psi1, train_lbs)
                fide_sr = fide.abs()
                loss_train_rec.append(loss.detach().cpu().numpy())
                train_fide.append(fide_sr.detach().cpu().numpy())
                
                # Clear GPU memory
                del psi0, loss, fide, fide_sr
                clear_gpu_memory()
                
                # Test evaluation
                psi0 = deepcopy(test_set)  # Use deepcopy for TensorTrain_pack
                for _ in range(para['recurrent_time']):
                    psi0 = qc(psi0)
                psi1 = psi0
                loss = loss_fun(psi1, test_lbs)
                fide = fidelity(psi1, test_lbs)
                fide_sr = fide.abs()
                
                loss_test_rec.append(loss.detach().cpu().numpy())
                test_fide.append(fide_sr.detach().cpu().numpy())
                
                print('Epoch %i: train loss %g, test loss %g; train fidelity %g, test fidelity %g' %
                    (t+1, loss_train_rec[-1], loss_test_rec[-1], train_fide[-1], test_fide[-1]))
                
                # Clear GPU memory again
                del psi0, psi1, loss, fide, fide_sr
                clear_gpu_memory()

    with tc.no_grad():
        results = dict()
        results['train_loss'] = loss_train_rec
        results['test_loss'] = loss_test_rec
        results['train_fide'] = train_fide
        results['test_fide'] = test_fide
        results['gpu_memory_usage'] = gpu_memory_usage
        results['gpu_memory_reserved'] = gpu_memory_reserved
    return qc, results, para

def main(qc_type:str='ADQC', init_param:dict=None, data:dict=None, nn_para:dict={}):
    if qc_type == 'ADQC':
        qc = ADQC(nn_para)
    elif qc_type == 'VQC':
        qc = VQC(nn_para)
    else:
        raise NotImplementedError
    if init_param != None:
        qc.load_state_dict(init_param, strict=False)
    print(f'---Before Train---')
    print(f'GPU Memory - Allocated: {get_gpu_memory_usage():.2f} MB, Reserved: {get_gpu_memory_reserved():.2f} MB')
    print('-'*20)
    qc, results, nn_para = train(qc, data, nn_para)
    return qc, results, nn_para