# from cv2 import mean
import numpy as np
import torch as tc

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

def n_body_unitaries(n:int, l:int, d:int, dtype:tc.dtype):
    m_dim = 2**n
    gates_di = tc.randn([d, l-n+1, m_dim, m_dim], dtype=dtype)
    gates_di, _ = tc.linalg.qr(gates_di)
    shape = [d, l-n+1]+[2]*n+[2]*n
    return gates_di.reshape(shape)

def n_body_evol_states(states, n:int, d:int):
    '''
    states.shape is [N] + [2]*l
    gates.shape is [d, m] + [2]*l + [2]*l (m = l-n+1)
    '''
    l = len(states.shape)-1
    m = l-n+1
    dtype = states.dtype
    gates = n_body_unitaries(n, l, d, dtype)
    which_where = list()
    for i in range(m):
        which_where.append([i]+list(range(i, i+n)))
    for i in range(d):
        states = pure_states_evolution(states, gates[i], which_where)
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

states = rand_states(3, 4, device=tc.device('cpu'))
print(tc.norm(states)**2)
new_states = n_body_evol_states(states, n=3, d=2)
print(new_states.shape)
print(tc.norm(new_states)**2)