import torch as tc
import numpy as np
from matplotlib import pyplot as plt
from Library import PhysModule as phy
from Library import BasicFun as bf
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

def Quantum_Sun_evol(states, para=None):
    para_def = {
        'dot_size': 3,
        'length': 10,
        'alpha': 1,
        'spin': 'half',
        'time_tot': 1,
        'tau': 0.01,
        'print_time': 1,
        'device': None,
        'dtype': tc.complex64
    }
    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['d'] = phy.from_spin2phys_dim(para['spin'])
    para['time_it'] = round(para['time_tot'] / para['tau'])
    para['print_time'] = para['print_time'] // para['tau']
    para['device'] = bf.choose_device(para['device'])

    gates = []

    A = tc.randn((2**para['dot_size'], 2**para['dot_size']), device=para['device'], dtype=tc.float32)
    A = A.to(para['dtype'])
    H_dot = 1/pow(2**para['dot_size'] + 1, 0.5) * 1/pow(2, 0.5) * (A + A.T)
    ops = phy.spin_operators(spin=para['spin'], device=para['device'], dtp=para['dtype'])
    SxSx = bf.kron(ops['sx'], ops['sx'])
    Sz = ops['sz']

    u = [j + 0.4*(tc.rand(1, device=para['device'])-0.5) for j in range(para['out_size'])]
    u[0] = 0
    h = [0.5 + tc.rand(1, device=para['device']) for _ in range(para['out_size'])]

    list_states = list()
    for t in range(para['time_it']):
        # states = pure_states_evolution(states, [gate, gate_l, gate_r], which_where)
        gate_dot = tc.matrix_exp(- 1j * para['tau'] * H_dot).reshape([para['d']] * 6)
        states = pure_states_evolution(states, [gate_dot], [[0, 0, 1, 2]])

        for j in range(para['out_size']):
            site_pos = para['dot_size'] + j
            dot_pos = random.randint(0, para['dot_size']-1)

            gate_interact = tc.matrix_exp(- 1j * para['tau'] * para['alpha']**u[j] * SxSx).reshape([para['d']] * 4)
            pure_states_evolution(states, [gate_interact], [[0, dot_pos, site_pos]])

            gate_mag = tc.matrix_exp(- 1j * para['tau'] * h[j] * Sz).reshape([para['d']] * 2)
            pure_states_evolution(states, [gate_mag], [[0, site_pos]])

        if t % para['print_time'] == para['print_time']-1:
            list_states.append(states)
    states = tc.stack(list_states, dim=1)
    return states

'''用循环生成n个[1, m]的随机张量和直接生成一个[n, m]的随机张量是一样的'''
def rand_states_(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [number, 2 ** length]
    states = tc.randn(shape, dtype=tc.complex128, device=device)
    sigma = tc.rand([number,1], dtype=tc.float64, device=device)
    mu_real = (tc.rand([number,1], dtype=tc.float64, device=device)-0.5)*6
    mu_img = (tc.rand([number,1], dtype=tc.float64, device=device)-0.5)*6
    mu = tc.complex(mu_real, mu_img)
    # print('mu=',mu)
    states = states*sigma+mu
    shape_ = [number] + [2]*length
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    # print('norm=',norm)
    states = states / tc.sqrt(norm.real)
    states = states.reshape(shape_)
    return states

def qr_Haar(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [number, 2**length, 2**length]
    real = tc.randn(shape, dtype=tc.float64, device=device)
    imag = tc.randn(shape, dtype=tc.float64, device=device)
    A = tc.complex(real, imag)
    Q, R = tc.linalg.qr(A, 'complete')
    Lambda = tc.diagonal(R, dim1=1, dim2=2)/tc.abs(tc.diagonal(R, dim1=1, dim2=2))
    Haar_mat = tc.einsum('nij,nj->nij', Q, Lambda)
    return Haar_mat

def Haar_rand_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [2 ** length]
    state1 = tc.zeros(shape, dtype=tc.complex128, device=device)
    state1[0] = 1
    print(state1)
    U = qr_Haar(number, length, device=device)
    states = tc.einsum('nij,j->ni', U, state1)
    shape_ = [number] + [2]*length
    states = states.reshape(shape_)
    return states

def Haar_random_product_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    product_states = Haar_rand_states(number, 1, device=device)
    for _ in range(length-1):
        temp = Haar_rand_states(number, 1, device=device)
        product_states = tc.einsum("n...i,nj->n...ij", product_states, temp)
    return product_states

def rand_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [number, 2 ** length]
    states = tc.randn(shape, dtype=tc.complex128, device=device)
    shape_ = [number] + [2]*length
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    states = states / tc.sqrt(norm)
    states = states.reshape(shape_)
    return states

def rand_dir_prod_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [number, 2]
    states = tc.rand(shape, dtype=tc.complex128, device=device)
    for _ in range(length - 1):
        states = tc.einsum('ij,ik->ijk', states, tc.rand(shape, dtype=tc.complex128, device=device))
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
    elif gen_type == 'h':
        gen = Haar_random_product_states
    # else:
    #     gen = None
    #     print('-'*10+'\nArgument --gen_type must be the combination of \'n\' and \'d\'!\n'+'-'*10)
    return gen

def label_generate(data, para):
    label = Quantum_Sun_evol(data, para)
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
    parser.add_argument('--dot_size', type=int, default=3)
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--evol_num', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=500)
    parser.add_argument('--folder', type=str, default='mixed_rand_states/')
    parser.add_argument('--evol_mat_path', type=str, default="GraduationProject/Data/evol_mat.npy")
    parser.add_argument('--gen_type', type=str, default='nn')
    parser.add_argument('--time_interval', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.01)
    args = parser.parse_args()

    para = dict()
    para['dot_size'] = args.dot_size
    para['length'] = args.length
    para['out_size'] = args.length - args.dot_size
    para['alpha'] = args.alpha
    para['device'] = device
    para['gen_type'] = args.gen_type
    para['spin'] = 'half'
    para['d'] = d
    para['dtype'] = dtype

    if args.seed != None:
        tc.manual_seed(args.seed)
        tc.cuda.manual_seed(args.seed)

    path = 'GraduationProject/Data'+args.folder
    from Library.BasicFun import mkdir
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
    para_test['sample_num'] = args.sample_num

    import os
    evol_mat_path = args.evol_mat_path
    # if os.access(evol_mat_path, os.F_OK):
    #     p555555
    # else:
    E = tc.eye(2**para['length'], dtype=dtype, device=para['device'])
    shape_ = [E.shape[0]] + [2] * para['length']
    E = E.reshape(shape_)
    evol_mat = Quantum_Sun_evol(E, para_evol).reshape(E.shape[0], -1)
    print('evol_mat.shape is', evol_mat.shape)
    np.save(evol_mat_path, evol_mat.cpu().T)

    gen_train = gen_select(para['gen_type'][0])
    trainset = gen_train(args.sample_num, para['length'], device=para['device'])
    trainset, train_lbs = label_generate(trainset, para_train)

    gen_test = gen_select(para['gen_type'][1])
    testset = gen_train(1, para['length'], device=para['device'])
    testset, test_lbs = label_generate(testset, para_test)
    data = dict()
    data['train_set'] = trainset
    data['train_lbs'] = train_lbs
    data['test_set'] = testset
    data['test_lbs'] = test_lbs
    
    np.save(path+'/train_set_sample_{:d}_evol_{:d}'.format(args.sample_num, args.evol_num), data)
