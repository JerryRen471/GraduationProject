import torch as tc
import numpy as np
import sys
sys.path.append('/data/home/scv7454/run/GraduationProject')
from Library.Tools import *
import Library.TensorNetwork as TN

def states_to_mps_pack(states:tc.Tensor, length:int, chi:int, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs)->TN.TensorTrain_pack:
    mps_pack = TN.TensorTrain_pack(tensor_packs=[states], length=length, phydim=2, center=-1, chi=chi, device=device, dtype=dtype, initialize=True)
    return mps_pack

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

def rand_entangled_states(number:int, length:int, entangle_dim:int=1, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs)->tc.Tensor:
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

def eig_states(number:int, length:int, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs)->tc.Tensor:
    number = int(number)
    def pauli_string(n, positions, paulis, sigma_x, sigma_z, identity, device=device):
        matrices = []
        for pos in range(1, n + 1):
            if pos in positions:
                idx = positions.index(pos)
                p = paulis[idx]
                if p == 'x':
                    mat = sigma_x.clone().to(device)
                elif p == 'z':
                    mat = sigma_z.clone().to(device)
                else:
                    raise ValueError(f"Invalid Pauli type: {p}")
            else:
                mat = identity.clone().to(device)
            matrices.append(mat)
        if not matrices:
            return tc.eye(1, dtype=dtype, device=device)
        term = matrices[0]
        for mat in matrices[1:]:
            term = tc.kron(term, mat)
        return term

    def construct_xorX(n, lamda, delta, J, device=device):
        sigma_x = tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        sigma_z = tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
        identity = tc.eye(2, dtype=dtype, device=device)
        
        H = tc.zeros((2**n, 2**n), dtype=dtype, device=device)
        
        # 第一部分：λ项（i从2到n-1）
        if n >= 3:
            for i in range(2, n):  # i的取值为2 ≤ i ≤ n-1
                term1 = pauli_string(n, [i], ['x'], sigma_x, sigma_z, identity, device)
                term2 = pauli_string(n, [i-1, i, i+1], ['z', 'x', 'z'], sigma_x, sigma_z, identity, device)
                H += lamda * (term1 - term2)
        
        # 第二部分：Δ项（i从1到n）
        for i in range(1, n + 1):
            term = pauli_string(n, [i], ['z'], sigma_x, sigma_z, identity, device)
            H += delta * term
        
        # 第三部分：J项（i从1到n-1）
        for i in range(1, n):
            term = pauli_string(n, [i, i+1], ['z', 'z'], sigma_x, sigma_z, identity, device)
            H += J * term
        
        return H
    
    H = construct_xorX(n=length, lamda=kwargs['lamda'], delta=kwargs['delta'], J=kwargs['J'], device=device)
    _, vecs = tc.linalg.eig(H)
    states = vecs.T.reshape([2**length] + [2]*length).to(device)
    if number < 2**length:
        states = states[:number]
    return states

import math
def RK_initial_state(length:int, ksi:float, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs):
    ksi = tc.tensor([ksi], dtype=dtype, device=device)
    Lb = length // 2
    state = tc.zeros((2 ** length, ), dtype=dtype, device=device)
    state[-1] = 1 + 0.j
    state = state.reshape([1] + [2]*length)
    for n in range(Lb):
        state = state + xorX_state_m(n, length, device, dtype) * tc.pow(ksi, n) * math.comb(length - n - 1, n)
    state = state.reshape([1, -1])
    shape_ = [1] + [2]*length
    norm = tc.sum(state * state.conj(), dim=1, keepdim=True)
    state = state / tc.sqrt(norm)
    state = state.reshape(shape_)
    return state

def RK_states(number:int, length:int, ksi_bar:float, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs):
    if number > 1:
        ksi_list = np.linspace(0, ksi_bar, number)
    elif number == 1:
        ksi_list = np.array([ksi_bar])
    state_list = []
    for ksi in ksi_list:
        state = RK_initial_state(length, ksi, device, dtype)
        state_list.append(state)
    states = tc.cat(state_list, dim=0)
    return states

def linear_comb_of_scar_states(number:int, length:int, device=tc.device('cuda:0'), dtype=tc.complex128, **kwargs):
    coefficients = np.random.random(size=[number, length//2]) + 1.j * np.random.random(size=[number, length//2])
    norm = np.sum(coefficients * coefficients.conj(), axis=1, keepdims=True).real
    coefficients = coefficients / np.sqrt(norm)
    coefficients = tc.from_numpy(coefficients).to(device=device, dtype=dtype)
    states = tc.zeros(size=[number]+[2]*length, device=device, dtype=dtype)
    for i in range(length // 2):
        states = states + tc.einsum('ij, jk...l->ik...l', coefficients[:, i].unsqueeze(1), xorX_state_m(i, length, device, dtype))
        pass
    return states

def main(init_para:dict):
    if init_para['type'] == 'non_product':
        return states_to_mps_pack(rand_states(**init_para), **init_para)
    elif init_para['type'] == 'product':
        return TN.rand_prod_mps_pack(**init_para)
    elif init_para['type'] == 'Z2':
        return states_to_mps_pack(Z2_states(**init_para), **init_para)
    elif init_para['type'] == 'entangled':
        return states_to_mps_pack(rand_entangled_states(**init_para), **init_para)
    elif init_para['type'] == 'xorX':
        return states_to_mps_pack(first_n_xorX_states(**init_para), **init_para)
    elif init_para['type'] == 'eig':
        return states_to_mps_pack(eig_states(**init_para), **init_para)
    elif init_para['type'] == 'RK':
        return states_to_mps_pack(RK_states(**init_para), **init_para)
    elif init_para['type'] == 'linear_scar':
        return states_to_mps_pack(linear_comb_of_scar_states(**init_para), **init_para)
    else:
        raise NotImplementedError