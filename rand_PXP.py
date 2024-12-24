from matplotlib import pyplot as plt
import torch as tc
import numpy as np
from ADQC import fidelity
from Library.BasicFun import mkdir
from Library import BasicFun as bf
from Library import PhysModule as phy
import random

def pure_states_evolution(states:tc.Tensor, gates:list, which_where:list)->tc.Tensor:
    """Evolve the state by several gates0.

    :param state: initial state
    :param gates: quantum gates
    :param which_where: [which gate, which spin, which spin]
    :return: evolved state
    Example: which_where = [[0, 1, 2], [1, 0, 1]] means gate 0 on spins
    1 and 2, and gate 1 on spins 0 and 1
    """
    def pure_states_evolution_one_gate(v, g, pos):
        ind = list(range(len(pos), 2*len(pos)))
        pos = [_+1 for _ in pos]
        # print("v.dtype",v.dtype)
        # print("g.dtype",g.dtype)
        v = tc.tensordot(v, g, [pos, ind]) # 对每个pos加一
        ind0 = list(range(v.ndimension()))
        for nn in range(len(pos)):
            ind0.remove(pos[nn])
        ind0 += pos
        order = np.argsort(ind0)
        return v.permute(tuple(order))

    for n in range(len(which_where)):
        states = pure_states_evolution_one_gate(
            states, gates[which_where[n][0]], which_where[n][1:])
    return states

def PXP_mul_states_evl(states, para=None):
    para_def = dict()
    para_def['length'] = 10
    para_def['time_tot'] = 1
    para_def['tau'] = 0.01
    para_def['print_time'] = para_def['time_tot']
    para_def['device'] = None
    para_def['dtype'] = tc.complex64
    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['time_it'] = round(para['time_tot'] / para['tau'])
    para['print_time'] = para['print_time'] // para['tau']
    para['device'] = bf.choose_device(para['device'])
    P = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    P[0, 0] = 1+0.j
    sigma_x = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    sigma_x[0, 1] = sigma_x[1, 0] = 1+0.j
    hamilt = tc.kron(P, tc.kron(sigma_x, P))
    gate = tc.matrix_exp(- 1j * para['tau'] * hamilt).reshape([2] * 6).to(dtype=para['dtype'])
    gate_list = [gate]
    which_where = []
    for i in range(1, para['length'] - 1):
        # gate_list.append(gate)
        which_where.append([0, i-1, i, i+1])
    list_states = list()
    for t in range(para['time_it']):
        states = pure_states_evolution(states, gate_list, which_where)
        if t % para['print_time'] == para['print_time']-1:
            list_states.append(states)
    states = tc.stack(list_states, dim=1)
    return states

def domain_wall(states, device=tc.device('cpu'), dtype=tc.complex64):
    length = states.dim() - 1
    from Library.PhysModule import n_combined_mags
    sigma_z = tc.zeros([2, 2], device=device, dtype=dtype)
    sigma_z[0, 0] = 1 + 0.j
    sigma_z[1, 1] = -1 + 0.j
    mags_zz = n_combined_mags(states, 2, sigma_z)
    domain_wall = tc.ones(mags_zz.shape[0], device=device, dtype=dtype) * (length-1)/2 + tc.sum(mags_zz, dim=[1, 2]) / 2
    return domain_wall

def fidelity_0_t(states):
    time_it = states.shape[0]
    state_0 = states[0]
    fidelity_list = []
    for i in range(time_it):
        fidelity = tc.einsum('a, a->', state_0.reshape(-1).conj(), states[i].reshape(-1))
        fide_sqare = fidelity.real**2 + fidelity.imag**2
        fidelity_list.append(fide_sqare)
    return tc.tensor(fidelity_list)

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
    if gen_type == 'product' or gen_type == 'd':
        gen = rand_dir_prod_states
    elif gen_type == 'n' or gen_type == 'non_product':
        gen = rand_states
    elif gen_type == 'Z2':
        gen = Z2_states
    elif gen_type == 'entangled':
        gen = rand_entangled_states
    # else:
    #     gen = None
    #     print('-'*10+'\nArgument --gen_type must be the combination of \'n\' and \'d\'!\n'+'-'*10)
    return gen

def label_generate(data, para):
    label = PXP_mul_states_evl(data, para)
    input = tc.cat((data.unsqueeze(1), label[:, :-1]), dim=1)
    merge = lambda x: x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])
    return merge(input), merge(label)

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
    parser.add_argument('--evol_num', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=500)
    parser.add_argument('--entangle_dim', type=int, default=1)
    parser.add_argument('--folder', type=str, default='mixed_rand_states/')
    parser.add_argument('--evol_mat_path', type=str, default="GraduationProject/Data/evol_mat.npy")
    parser.add_argument('--gen_type', type=str, default='Z2')
    parser.add_argument('--time_interval', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.02)
    args = parser.parse_args()

    para = dict()
    para['length'] = args.length
    para['device'] = device
    para['gen_type'] = args.gen_type
    para['spin'] = 'half'
    para['d'] = d
    para['dtype'] = dtype
    para['tau'] = args.tau

    if args.seed != None:
        tc.manual_seed(args.seed)

    path = 'GraduationProject/Data'+args.folder
    mkdir(path)
    para_evol = dict(para)
    para_evol['time_tot'] = args.time_interval
    para_evol['print_time'] = args.time_interval

    para_train = dict(para)
    para_train['time_tot'] = args.time_interval*args.evol_num
    para_train['print_time'] = args.time_interval
    para_train['sample_num'] = args.sample_num

    para_test = dict(para)
    para_test['time_tot'] = args.time_interval
    para_test['print_time'] = args.time_interval
    para_test['sample_num'] = 100

    import os
    evol_mat_path = args.evol_mat_path
    # if os.access(evol_mat_path, os.F_OK):
    #     pass
    # else:
    E = tc.eye(2**para['length'], dtype=para['dtype'], device=para['device'])
    shape_ = [E.shape[0]] + [2] * para['length']
    E = E.reshape(shape_)
    evol_mat = PXP_mul_states_evl(E, para_evol).reshape(E.shape[0], -1)
    print('evol_mat.shape is', evol_mat.shape)
    np.save(evol_mat_path, evol_mat.cpu().T)

    gen_train = gen_select(para['gen_type'])
    trainset = gen_train(para_train['sample_num'], para['length'], device=para['device'], dtype=para['dtype'], entangle_dim=args.entangle_dim)
    trainset, train_lbs = label_generate(trainset, para_train)

    gen_test = gen_select('d')
    testset = gen_test(para_test['sample_num'], para['length'], device=para['device'], dtype=para['dtype'], entangle_dim=args.entangle_dim)
    testset, test_lbs = label_generate(testset, para_test)
    data = dict()
    data['train_set'] = trainset
    data['train_lbs'] = train_lbs
    data['test_set'] = testset
    data['test_lbs'] = test_lbs
    
    np.save(path+'/train_set_sample_{:d}_evol_{:d}'.format(args.sample_num, args.evol_num), data)

    print(f'trainset.shape = {trainset.shape}')

    mkdir('GraduationProject/pics'+args.folder+f'/mag_evol{args.evol_num}')

    fidelity_evol = fidelity_0_t(trainset)
    legend = []
    time_it = len(fidelity_evol)
    x = np.array([_ for _ in range(time_it)]) * args.time_interval
    y = np.array(fidelity_evol.cpu())
    plt.plot(x, y, label='|\inner{\psi_0}{\psi}|^2')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('fidelity^2')
    plt.title(f'Average Domain-Wall')
    plt.savefig('GraduationProject/pics'+args.folder+f'/mag_evol{args.evol_num}/fidelity_0_t.svg')
    plt.close()
    domain_walls = domain_wall(trainset, device=para['device'], dtype=para['dtype'])
    print(f'domain_walls.shape = {domain_walls.shape}')
    legend = []
    time_it = domain_walls.shape[0]
    x = np.array([_ for _ in range(time_it)]) * args.time_interval
    y = np.array(domain_walls.cpu())
    plt.plot(x, y, label='s')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Domain Wall Number')
    plt.title(f'Average Domain-Wall')
    plt.savefig('GraduationProject/pics'+args.folder+f'/mag_evol{args.evol_num}/AverageDomainWall.svg')
    plt.close()