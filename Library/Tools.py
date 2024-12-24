import torch as tc
import numpy as np

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

def pure_states_evolution(states:tc.Tensor, gates:list, which_where:list)->tc.Tensor:
    """Evolve the state by several gates0.

    :param state: initial state
    :param gates: quantum gates
    :param which_where: [which gate, which spin, which spin]
    :return: evolved state
    Example: which_where = [[0, 1, 2], [1, 0, 1]] means gate 0 on spins
    1 and 2, and gate 1 on spins 0 and 1
    """
    for n in range(len(which_where)):
        states = pure_states_evolution_one_gate(
            states, gates[which_where[n][0]], which_where[n][1:])
    return states