import torch as tc
import numpy as np
from Library import BasicFun as bf
from qiskit.quantum_info import Clifford, random_clifford

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
        op['sy'][0, 1] = -0.5 * 1j
        op['sy'][1, 0] = 0.5 * 1j
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


def reduced_density_matrces(states, pos):
    ind = list(range(1, states.ndimension()))
    if type(pos) is int:
        ind.remove(pos+1)
        dim = states.shape[pos+1]
    else:
        for n in pos:
            ind.remove(n+1)
        dim = 1
        for n in pos:
            dim *= states.shape[n+1]
    rho_ = tc.tensordot(states, states.conj(), [ind, ind])
    reduced_len = rho_.ndimension()
    rho__ = tc.diagonal(rho_, dim1=0, dim2=int(reduced_len/2))
    rho = rho__.permute([reduced_len-2]+list(range(reduced_len-2)))
    return rho.reshape(rho.shape[0], dim, dim)


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


def op_dir_prod_n_times(op:tc.Tensor, n:int):
    temp_op = op
    new_op = temp_op
    for _ in range(n):
        new_op = temp_op
        temp_op = tc.kron(temp_op, op)
    return new_op

def n_combined_mags(states, n, which_ops=None):
    '''
    para:: states states.shape = [num_of_states, shape_of_each_state]
    '''
    if which_ops is None:
        op = spin_operators('half', device=states.device)
        which_ops = [op['sx'], op['sy'], op['sz']]
    if type(which_ops) in [list, tuple]:
        num_ops = len(which_ops)
    else:
        num_ops = 1
        which_ops = [which_ops]
    length = states.ndimension()-1
    mags = tc.zeros((states.shape[0], num_ops, length-n+1), device=states.device, dtype=states.dtype)
    for i in range(length-n+1):
        rhos = reduced_density_matrces(states, list(range(i, i+n)))
        for s in range(num_ops):
            # print(rho)
            # print(which_ops[s])
            obs = op_dir_prod_n_times(which_ops[s], n)
            mags[:, s, i] = tc.einsum('abc,cb->a', rhos.type(tc.complex64), obs.type(tc.complex64))
    return mags


def mags_from_states(states, device='cpu'):
    spin = spin_operators('half', device=device)
    mag_z = n_combined_mags(states, n=1, which_ops=[spin['sz']])
    return mag_z

def mag_from_states(states, device='cpu'):
    mag_z = mags_from_states(states=states, device=device)
    mag_z_tot = mag_z.sum(dim=1) / mag_z.shape[1] # 不同时刻链的z方向总磁矩对链长做平均
    return mag_z_tot

def multi_mags_from_states(states, spins=None, device='cpu'):
    multi_mags = list()
    multi_mags = n_combined_mags(states, n=1, which_ops=spins)
    return multi_mags

def measure(state):
    shape = state.shape
    state = state.reshape(-1)
    probabilities = tc.abs(state) ** 2
    # 根据概率分布进行测量
    measured_index = tc.multinomial(probabilities, 1).item()  # 进行测量
    measured_state = tc.zeros_like(state)
    measured_state[measured_index] = 1  # 设置对应的量子态为1

    return measured_state.reshape(shape)

# def multi_mags_from_states_sample(states, sample_time:int, spins=None, device='cpu'):
#     count_state = tc.zeros_like(states)
#     for i in range(sample_time):
#         measured_state = measure(states)
#         count_state = count_state + measured_state
#     mean_mags = [cal_mean_mag_on_site_index(count_state=count_state, i=k) for k in range(n_qubit)]

#     return multi_mags

def combined_mags(states, which_ops=None):
    '''
    para:: states states.shape = [num_of_states, shape_of_each_state]
    '''
    mags = n_combined_mags(states, n=2, which_ops=which_ops)
    return mags


def complete_two_dir_prod_op(ops1, ops2):
    dir_prod_list = list()
    for opi in ops1:
        for opj in ops2:
            op_temp = tc.kron(opi, opj)
            dir_prod_list.append(op_temp)
    return dir_prod_list


def two_body_ob(states, ops=None):
    if ops == None:
        op = spin_operators('half', if_list=False, device=states.device)
        which_ops = [op['id'],op['sx'], op['sy'], op['sz']]
        ops = complete_two_dir_prod_op(which_ops, which_ops)
        ops.pop(0)
    if type(ops) in [list, tuple]:
        num_ops = len(ops)
    else:
        num_ops = 1
        ops = list(ops)
    length = states.ndimension()-1
    n = 2
    mags = tc.zeros((states.shape[0], num_ops, length-n+1), dtype=tc.complex64)
    for i in range(length-n+1):
        rhos = reduced_density_matrces(states, list(range(i, i+n)))
        for s in range(num_ops):
            # print(rho)
            # print(which_ops[s])
            obs = ops[s]
            mags[:, s, i] = tc.einsum('abc,cb->a', rhos.type(tc.complex64), obs.type(tc.complex64))
    return mags


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

def fidelity(psi1:tc.Tensor, psi0:tc.Tensor):
    f = 0
    for i in range(psi1.shape[0]):
        psi0_ = psi0[i]
        psi1_ = psi1[i]
        x_pos = list(range(len(psi1_.shape)))
        y_pos = x_pos
        f_ = bf.tmul(psi1_.conj(), psi0_, x_pos, y_pos)
        f += (f_*f_.conj()).real
    f = f/psi1.shape[0]
    return f

def process_fide(U1:tc.Tensor, U0:tc.Tensor):
    '''
    U1 and U0 must be square matrices with the same shape
    '''
    M = tc.mm(U0.T.conj(), U1)
    n = U0.shape[0]
    f = 1/(n*(n+1)) * (n + tc.abs(tc.einsum('ii->',M))**2)
    return f

def rand_id(shape, dim, dtype, device):
    id = tc.eye(dim, dtype=dtype, device=device)
    for n in reversed(shape):
        id = tc.stack(list(id for _ in range(n)))
    return id

def get_quantum_gates_and_qubits(qc):
    gates_info = []
    for gate in qc.data:
        gate_name = gate.name  # 获取量子门的名称
        qubits = gate.qubits  # 获取作用的比特
        gates_info.append((gate_name, [qc.qubits.index(qubit) for qubit in qubits]))  # 记录量子门及其作用的比特索引
    return gates_info

def convert_qiskit_circuit_to_usual_gates(qiskit_circuit, dtype=tc.complex64, device=tc.device('cpu')):
    x_gate = tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    y_gate = tc.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    z_gate = tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    s_gate = tc.tensor([[1, 0], [0, 1j]], dtype=dtype, device=device)
    sdg_gate = tc.tensor([[1, 0], [0, -1j]], dtype=dtype, device=device)
    h_gate = tc.tensor([[1/(2**0.5), 1/(2**0.5)], [1/(2**0.5), -1/(2**0.5)]], dtype=dtype, device=device)

    cnot_gate = tc.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=dtype, device=device).reshape(2, 2, 2, 2)
    swap_gate = tc.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype, device=device).reshape(2, 2, 2, 2)

    gate_dict = {
        'x': x_gate,
        'y': y_gate,
        'z': z_gate,
        's': s_gate,
        'sdg': sdg_gate,
        'h': h_gate,
        'cx': cnot_gate,
        'swap': swap_gate
    }

    gates_info = get_quantum_gates_and_qubits(qiskit_circuit)
    gate_list = list(gate_dict.values())
    gate_names = list(gate_dict.keys())

    which_where = []
    for gate_name, qubit_indices in gates_info:
        # print(f"Gate: {gate_name}, Qubits: {qubit_indices}")
        which_where.append([gate_names.index(gate_name)] + qubit_indices)
    
    return gate_names, gate_list, which_where

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

def measure(state):
    shape = state.shape
    state = state.reshape(-1)
    probabilities = tc.abs(state) ** 2
    # 根据概率分布进行测量
    measured_index = tc.multinomial(probabilities, 1, replacement=True).item()  # 进行测量
    measured_state = tc.zeros_like(state)
    measured_state[measured_index] = 1  # 设置对应的量子态为1

    return measured_state.reshape(shape)

def measure_n_times(states, number:int):
    '''
    返回测量n次得到的平均密度矩阵
    '''
    shape = states.shape
    states = states.reshape([shape[0], -1])
    probabilities = tc.abs(states) ** 2
    # 根据概率分布进行多次测量
    measured_indices = tc.multinomial(probabilities, number, replacement=True)  # 进行多次测量
    # 统计每个测量结果的出现次数
    # counts = tc.bincount(measured_indices, minlength=states.shape[1])
    counts = []
    for row in measured_indices:
        # 使用 bincount 统计每一行的元素出现次数
        count = tc.bincount(row, minlength=states.shape[1])  # minlength 确保包含所有可能的值
        counts.append(count)
    counts = tc.stack(counts)  # 将结果堆叠成一个张量
    pd = counts / tc.sum(counts, dim=1, keepdim=True)
    avg_state = tc.sqrt(pd)
    # avg_rho = tc.diag(pd)
    return avg_state.reshape(shape).to(dtype=states.dtype)

def sample_classical_shadow(aim_states, n_qubit, num_sample=10000)->tc.Tensor:
    '''
    state的形状为[number_of_states]+[2]*n_qubit;
    
    返回一个形状为[number_of_states]+[2**n_qubit, 2**n_qubit]的密度矩阵
    '''
    n = aim_states.shape[0]
    cliff = random_clifford(n_qubit)
    qc = cliff.to_circuit()
    gate_names, gate_list, which_where = convert_qiskit_circuit_to_usual_gates(qc, dtype=aim_states.dtype, device=aim_states.device)
    U_states = pure_states_evolution(aim_states, gate_list, which_where)
    # b_state = measure(U_state)
    b_states = measure_n_times(U_states, num_sample)

    inverse_qc = qc.inverse()
    gate_names, gate_list, which_where = convert_qiskit_circuit_to_usual_gates(inverse_qc, dtype=aim_states.dtype, device=aim_states.device)
    sigma_states = pure_states_evolution(b_states, gate_list, which_where)
    sigmas = tc.einsum('na, nb -> nab', sigma_states.reshape([n, -1]), sigma_states.reshape([n, -1]).conj())
    Eye = tc.eye(2**n_qubit, dtype=aim_states.dtype, device=aim_states.device)
    Eye_tensor = Eye.unsqueeze(0).expand(n, -1, -1)
    rho_samples = (2**n_qubit + 1) * sigmas - Eye_tensor
    return rho_samples