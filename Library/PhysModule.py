import torch as tc
import numpy as np
from Library import BasicFun as bf


def spin_operators(spin, if_list=False, device='cpu', dtp=tc.complex128):
    op = dict()
    if spin == 'half':
        op['id'] = tc.eye(2, device=device, dtype=dtp)
        op['sx'] = tc.zeros((2, 2), device=device, dtype=dtp)
        op['sy'] = tc.zeros((2, 2), device=device, dtype=dtp)
        op['sz'] = tc.zeros((2, 2), device=device, dtype=dtp)
        op['su'] = tc.zeros((2, 2), device=device, dtype=dtp)
        op['sd'] = tc.zeros((2, 2), device=device, dtype=dtp)
        op['sx'][0, 1] = 0.5
        op['sx'][1, 0] = 0.5
        op['sy'][0, 1] = 0.5 * 1j
        op['sy'][1, 0] = -0.5 * 1j
        op['sz'][0, 0] = 0.5
        op['sz'][1, 1] = -0.5
        op['su'][0, 1] = 1.0
        op['sd'][1, 0] = 1.0
        # op['sy'] = tc.from_numpy(op['sy'])
    elif spin == 'one':
        op['id'] = tc.eye(3, device=device, dtype=dtp)
        op['sx'] = tc.zeros((3, 3), device=device, dtype=dtp)
        op['sy'] = tc.zeros((3, 3), dtype=tc.complex128)
        op['sz'] = tc.zeros((3, 3), device=device, dtype=dtp)
        op['sx'][0, 1] = 1.0
        op['sx'][1, 0] = 1.0
        op['sx'][1, 2] = 1.0
        op['sx'][2, 1] = 1.0
        op['sy'][0, 1] = -1.0j
        op['sy'][1, 0] = 1.0j
        op['sy'][1, 2] = -1.0j
        op['sy'][2, 1] = 1.0j
        op['sz'][0, 0] = 1.0
        op['sz'][2, 2] = -1.0
        op['sx'] /= 2 ** 0.5
        op['sy'] /= 2 ** 0.5
        # op['sy'] = tc.from_numpy(op['sy'])
        op['su'] = op['sx'] + 1j * op['sy']
        op['sd'] = op['sx'] - 1j * op['sy']
    if if_list:
        for key in op:
            op[key] = [op[key].real, op[key].imag]
    return op


def from_spin2phys_dim(spin, is_type='spin'):
    if is_type == 'spin':
        if spin == 'half':
            return 2
        elif spin == 'one':
            return 3
    elif is_type == 'fermion':
        if spin == 'one_half':
            return 4
        elif spin == 'zero':
            return 2

def hamiltonian_NN_NNN(spin, t1, t2, V1, V2, device, dtype):
    op = spin_operators(spin, if_list=False, device=device, dtp=dtype)
    hamilt = -t1 * (bf.kron(op['su'], op['sd']) - bf.kron(op['sd'], op['su']))\
            + V1 * (bf.kron(op['su']@op['sd']-0.5*op['id'], op['id'])\
                    @ bf.kron(op['id'], op['su']@op['sd']-0.5*op['id']))
    hamilt = bf.kron(hamilt, op['id'])
    hamilt -= t2 * (bf.kron(bf.kron(op['su'], op['sz']), op['sd']) - bf.kron(bf.kron(op['sd'], op['sz']), op['su']))
    hamilt += V2 * (bf.kron(bf.kron(op['su']@op['sd']-0.5*op['id'], op['id']), op['id']))\
                    @ (bf.kron(bf.kron(op['id'], op['id']), op['su']@op['sd']-0.5*op['id']))
    return hamilt

def hamiltonian_heisenberg(spin, jx, jy, jz, hx, hy, hz, device, dtype):
    op = spin_operators(spin, if_list=False, device=device, dtp=dtype)
    hamilt = jx*bf.kron(op['sx'], op['sx']) + jy*bf.kron(op['sy'], op['sy']) + jz*bf.kron(
        op['sz'], op['sz'])
    hamilt -= (hx[0] * bf.kron(op['sx'], op['id']) + hx[1] * bf.kron(op['id'], op['sx']))
    hamilt -= (hy[0] * bf.kron(op['sy'], op['id']) + hy[1] * bf.kron(op['id'], op['sy']))
    hamilt -= (hz[0] * bf.kron(op['sz'], op['id']) + hz[1] * bf.kron(op['id'], op['sz']))
    return hamilt


def observe_single_site(state, op, pos):
    state1 = tc.tensordot(state, op, [[pos], [1]])
    ind = list(range(0, pos)) + [state.ndimension()-1] + list(range(pos, state.ndimension()-1))
    state1 = state1.permute(ind).reshape(-1)
    return tc.dot(state.conj().reshape(-1), state1)


def reduced_density_matrix(state, pos):
    ind = list(range(state.ndimension()))
    if type(pos) is int:
        ind.remove(pos)
        dim = state.shape[pos]
    else:
        for n in pos:
            ind.remove(n)
        dim = 1
        for n in pos:
            dim *= state.shape[n]
    rho = tc.tensordot(state, state.conj(), [ind, ind])
    return rho.reshape(dim, dim)


def magnetizations(state:tc.Tensor, which_ops=None)->tc.Tensor:
    if which_ops is None:
        op = spin_operators('half', if_list=True, device=state.device)
        which_ops = [op['sx'], op['sy'], op['sz']]
    if type(which_ops) in [list, tuple]:
        num_ops = len(which_ops)
    else:
        num_ops = 1
        which_ops = [which_ops]
    length = state.ndimension()
    mag = tc.zeros((num_ops, length), dtype=tc.complex128)
    for n in range(length):
        rho = reduced_density_matrix(state, n)
        for s in range(num_ops):
            # print(rho)
            # print(which_ops[s])
            mag[s, n] = tc.einsum('ab,ba->', rho.type(tc.complex64), which_ops[s].type(tc.complex64))
    return mag


def mag_from_states(states, device='cpu'):
    spin = spin_operators('half', device=device)
    mag_z = list()
    for _ in range(states.shape[0]):
        mag_z.append(magnetizations(states[_], [spin['sz']]))
    mag_z = tc.cat(mag_z, dim=0)
    mag_z_tot = mag_z.sum(dim=1) / mag_z.shape[1] # 不同时刻链的z方向总磁矩对链长做平均
    return mag_z_tot


def combined_mag(state, which_ops=None):
    if which_ops is None:
        op = spin_operators('half', if_list=True, device=state.device)
        which_ops = [op['sx'], op['sy'], op['sz']]
    if type(which_ops) in [list, tuple]:
        num_ops = len(which_ops)
    else:
        num_ops = 1
        which_ops = [which_ops]
    length = state.ndimension()
    mag = tc.zeros((num_ops, length))
    for n in range(length-1):
        rho = reduced_density_matrix(state, [n, n+1])
        for s in range(num_ops):
            # print(rho)
            # print(which_ops[s])
            obs = tc.kron(which_ops[s], which_ops[s])
            mag[s, n] = tc.einsum('ab,ba->', rho.type(tc.complex64), obs.type(tc.complex64))
    return mag


def observe_one_bond_energy(state, h, where):
    '''
    return = <phi_n|h|phi_n> is a number
    '''
    state1 = tc.tensordot(state, h, [where, list(range(len(h.shape)//2, len(h.shape)))])

    # 将 state 和 h 通过 tensordot 作用后得到的张量的指标交换回作用前 state 的指标顺序
    ind = list(range(state.ndimension()))
    for n in where:
        ind.remove(n)
    ind += where
    ind = tuple(np.argsort(ind))
    state1 = state1.permute(ind).reshape(-1, )

    return tc.einsum('a,a->', state1, state.reshape(-1, )).item()


def bond_energies(state, hamiltonians, which_where):
    energy = list()
    for n in range(len(which_where)):
        ener = observe_one_bond_energy(state, hamiltonians[
            which_where[n][0]], which_where[n][1:])
        energy.append(ener)
    return energy


def entanglement_entropy(lm):
    '''
    calculate entaglement entropy through the singular spectrum
    '''
    lm1 = lm ** 2 + 1e-12
    if type(lm1) is tc.Tensor:
        return tc.dot(-1 * lm1, tc.log(lm1))
    else:
        return np.inner(-1 * lm1, np.log(lm1))

