from cProfile import label
import torch as tc
import numpy as np
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

def n_body_unitaries(n:int, l:int, d:int, noise:float=1e-3, dtype:tc.dtype=tc.complex128, device=tc.device('cuda:0')):
    m_dim = 2**n
    gates_di = phy.rand_id([d, l-n+1], m_dim, dtype=dtype, device=device)\
          + tc.randn([d, l-n+1, m_dim, m_dim], dtype=dtype, device=device)*noise
    gates_di, _ = tc.linalg.qr(gates_di)
    shape = [d, l-n+1]+[2]*n+[2]*n
    return gates_di.reshape(shape)

def n_body_evol_states(states, gates):
    '''
    states.shape is [N] + [2]*l
    gates.shape is [d, m] + [2]*l + [2]*l (m = l-n+1)
    '''

    l = len(states.shape)-1
    d = gates.shape[0]
    m = gates.shape[1]
    n = l-m+1
    which_where = list()
    for i in range(m):
        which_where.append([i]+list(range(i, i+n)))
    for i in range(d):
        states = pure_states_evolution(states, gates[i], which_where)
    return states

def random_uni_evl(data, evol_mat):
    shape_ = data.shape
    data = data.reshape([shape_[0], -1])
    labels = tc.einsum('ij,aj->ai', evol_mat, data)
    labels = labels.reshape(shape_)
    return labels


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
        gen = Haar_rand_states
    # else:
    #     gen = None
    #     print('-'*10+'\nArgument --gen_type must be the combination of \'n\' and \'d\'!\n'+'-'*10)
    return gen

if __name__ == '__main__':
    length = 10
    spin = 'half'
    d = phy.from_spin2phys_dim(spin)
    device = tc.device('cuda:0')
    dtype = tc.complex128

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--train_num', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=50)
    parser.add_argument('--folder', type=str, default="/rand_unitary/loss_mags/dn")
    parser.add_argument('--gen_type', type=str, default='dn')
    parser.add_argument('--n_body_uni', type=int, default=2)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--evol_mat_path', type=str, default="GraduationProject/Data/evol_mat.npy")
    parser.add_argument('--noise', type=float, default=1e-3)
    args = parser.parse_args()

    para = dict()
    para['length'] = length
    para['spin'] = 'half'
    para['d'] = d
    para['device'] = device
    para['dtype'] = dtype
    para['folder'] = args.folder
    para['seed'] = args.seed

    path = 'GraduationProject/Data'+para['folder']
    mkdir(path)
    
    import os
    evol_mat_path = args.evol_mat_path
    if os.access(evol_mat_path, os.F_OK):
        evol_mat = np.load(evol_mat_path, allow_pickle=True)
        evol_mat = tc.from_numpy(evol_mat).cuda()
    else:
        gates = n_body_unitaries(args.n_body_uni, para['length'], d=args.depth, noise=args.noise, dtype=para['dtype'])
        one = tc.eye(2**para['length'], dtype=para['dtype'], device=tc.device('cuda:0'))
        one = one.reshape([2**para['length']]+[2]*para['length'])
        evol_mat = n_body_evol_states(one, gates)
        evol_mat = evol_mat.reshape([2**para['length'], 2**para['length']])
        np.save(evol_mat_path, evol_mat.cpu().T)

    if para['seed'] != None:
        tc.manual_seed(para['seed'])
    gen_train = gen_select(args.gen_type[0])
    trainset = gen_train(args.train_num, length, device=para['device'])
    print(trainset.dtype)
    train_lbs = random_uni_evl(trainset, evol_mat)

    gen_test = gen_select(args.gen_type[1])
    testset = gen_test(args.test_num, length, device=para['device'])
    test_lbs = random_uni_evl(testset, evol_mat)
    data = dict()
    data['train_set'] = trainset
    data['train_lbs'] = train_lbs
    data['test_set'] = testset
    data['test_lbs'] = test_lbs

    
    np.save(path+'/data_num{:d}'.format(args.train_num), data)


    