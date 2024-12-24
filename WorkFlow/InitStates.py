import torch as tc
import numpy as np
import sys
sys.path.append('/data/home/scv7454/run/GraduationProject')
from Library.Tools import *

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

def Z2_states(number:int, length:int, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs):
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

def rand_entangled_states(number:int, length:int, entangle_dim:int=1, device=tc.device('cuda:0'), dtype=tc.complex64, **kwargs)->tc.Tensor:
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

def xorX_state_m(m:int, length:int, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs)->tc.Tensor:
    state = tc.zeros((2 ** length, ), dtype=dtype, device=device)
    state[-1] = 1 + 0.j
    state = state.reshape([1] + [2]*length)
    for _ in range(m):
        state = act_Q_dagger(length, state, dtype=dtype, device=device)
    state = state.reshape([1, -1])
    shape_ = [1] + [2]*length
    norm = tc.sum(state * state.conj(), dim=1, keepdim=True)
    state = state / tc.sqrt(norm)
    state = state.reshape(shape_)
    return state

def act_Q_dagger(length, state, device=tc.device('cuda:0'), dtype=tc.complex128):
    P0 = tc.zeros((2, 2), dtype=dtype, device=device)
    P0[1, 1] = 1+0.j
    sigma_up = tc.zeros((2, 2), dtype=dtype, device=device)
    sigma_up[0, 1] = 1+0.j
    op_plus = tc.kron(P0, tc.kron(sigma_up, P0)).reshape([2] * 6)
    op_minus = -tc.kron(P0, tc.kron(sigma_up, P0)).reshape([2] * 6)
    ops = [op_plus, op_minus]
    which_where = []
    for i in range(1, length-1):
        which_where.append([i%2, i-1, i, i+1])
    fin_state = tc.zeros_like(state)
    for n in range(len(which_where)):
        state_tmp = pure_states_evolution_one_gate(
            state, ops[which_where[n][0]], which_where[n][1:])
        fin_state += state_tmp
    return fin_state

def first_n_xorX_states(number:int, length:int, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs)->tc.Tensor:
    number = int(number)
    for m in range(1, number+1):
        state = xorX_state_m(m, length, dtype=dtype, device=device)
        if m == 1:
            states = state
        else:
            states = tc.cat([states, state], dim=0)
    return states

def main(init_para:dict):
    if init_para['type'] == 'non_product':
        return rand_states(**init_para)
    elif init_para['type'] == 'product':
        return rand_dir_prod_states(**init_para)
    elif init_para['type'] == 'Z2':
        return Z2_states(**init_para)
    elif init_para['type'] == 'entangled':
        return rand_entangled_states(**init_para)
    elif init_para['type'] == 'xorX':
        return first_n_xorX_states(**init_para)
    else:
        raise NotImplementedError