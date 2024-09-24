import torch as tc
import numpy as np
from ADQC import fidelity
from Library.BasicFun import mkdir
from Library import BasicFun as bf
from Library import PhysModule as phy
import random

def rand_states(number:int, length:int, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs)->tc.Tensor:
    number = int(number)
    shape = [number, 2 ** length]
    states = tc.randn(shape, dtype=dtype, device=device)
    shape_ = [number] + [2]*length
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    states = states / tc.sqrt(norm)
    states = states.reshape(shape_)
    return states

def rand_dir_prod_states(number:int, length:int, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs)->tc.Tensor:
    number = int(number)
    shape = [number, 2]
    states = tc.rand(shape, dtype=dtype, device=device)
    for _ in range(length - 1):
        states = tc.einsum('ij,ik->ijk', states, tc.rand(shape, dtype=dtype, device=device))
        states = states.reshape(number, -1)
    print(states.shape)
    shape_ = [number] + [2]*length
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    states = states / tc.sqrt(norm)
    states = states.reshape(shape_)
    return states

def Z2_states(number:int, length:int, device=tc.device('cuda:0'), dtype=tc.complex64, **kwargs):
    state0 = tc.zeros([number, 2], device=device, dtype=dtype)
    state0[:, 0] = tc.ones([number], device=device, dtype=dtype)
    state1 = tc.zeros([number, 2], device=device, dtype=dtype)
    state1[:, 1] = tc.ones([number], device=device, dtype=dtype)
    states = state0
    for _ in range(1, length):
        states = tc.einsum('ij,ik->ijk', states, [state0, state1][_%2])
        states = states.reshape(number, -1)
    print(states.shape)
    shape_ = [number] + [2]*length
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    states = states / tc.sqrt(norm)
    states = states.reshape(shape_)
    return states

def rq_decomposition(matrix):
    # 对matrix的后两个维度进行转置
    matrix_transposed = matrix.transpose(-2, -1)

    q, r = tc.linalg.qr(matrix_transposed)
    r = r.transpose(-2, -1)
    q = q.transpose(-2, -1)
    return r, q

def rand_entangled_states(number:int, length:int, entangle_dim:int, device=tc.device('cuda:0'), dtype=tc.complex64):
    L = length // 2
    R = length - L
    states_L = tc.randn([number, 2**L, entangle_dim], dtype=dtype, device=device)
    states_L, _ = tc.linalg.qr(states_L)
    states_R = tc.randn([number, entangle_dim, 2**R], dtype=dtype, device=device)
    _, states_R = rq_decomposition(states_R)
    states_mid = tc.eye(entangle_dim, dtype=dtype, device=device)
    states = tc.einsum('ijk,kl,iln->ijn', states_L, states_mid, states_R)
    states = states.reshape(number, -1)
    print(states.shape)
    shape_ = [number] + [2]*length
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    states = states / tc.sqrt(norm)
    states = states.reshape(shape_)
    return states

def gen_select(gen_type:str):
    if gen_type == 'd':
        gen = rand_dir_prod_states
    elif gen_type == 'n':
        gen = rand_states
    elif gen_type == 'Z2':
        gen = Z2_states
    elif gen_type == 'entangled':
        gen = rand_entangled_states
    # else:
    #     gen = None
    #     print('-'*10+'\nArgument --gen_type must be the combination of \'n\' and \'d\'!\n'+'-'*10)
    return gen

if __name__ == '__main__':
    # length = 14
    spin = 'half'
    d = phy.from_spin2phys_dim(spin)
    device = tc.device('cuda:0')
    dtype = tc.complex128

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=500)
    parser.add_argument('--entangle_dim', type=int, default=1)
    parser.add_argument('--folder', type=str, default='mixed_rand_states/')
    parser.add_argument('--gen_type', type=str, default='Z2')
    args = parser.parse_args()

    para = dict()
    para['length'] = args.length
    para['device'] = device
    para['gen_type'] = args.gen_type
    para['spin'] = 'half'
    para['d'] = d
    para['dtype'] = dtype

    if args.seed != None:
        tc.manual_seed(args.seed)

    path = 'GraduationProject/Data'+args.folder
    mkdir(path)

    import os

    gen_train = gen_select(para['gen_type'])
    trainset = gen_train(para_train['sample_num'], para['length'], device=para['device'], dtype=para['dtype'], entangle_dim=para['entangle_dim'])

    gen_test = gen_select('d')
    testset = gen_train(para_test['sample_num'], para['length'], device=para['device'], dtype=para['dtype'], entangle_dim=para['entangle_dim'])
    data = dict()
    data['train_set'] = trainset
    data['test_set'] = testset
    np.save(path+'/init_states_sample_{:d}_evol_{:d}'.format(args.sample_num, args.evol_num), data)
