import torch as tc
import numpy as np
from Library.TensorNetwork import n_body_gate_to_mpo

def dagger_gate(gate):
    shape = list(gate.shape)
    phy_dim = shape[0]
    new_gate = gate.reshape([phy_dim ** (len(shape)//2), phy_dim ** (len(shape)//2)]).T.conj().reshape(shape)
    return new_gate

def fill_seq(pos:list):
    delta_pos = list(i for i in range(pos[0], pos[-1]+1))
    for i in pos:
        delta_pos.remove(i)
    return delta_pos
def step_function(i:int, pos:list):
    f = lambda x: 0 if x < 0 else 1
    y = 0
    for j in pos[1:]:
        y = y + f(i - j)
    return y

def process_mpo_tensors(mpo_t_list, pos, mpo, device, dtype):
    """
    Process MPO tensors and update the mpo list.
    
    Args:
        mpo_t_list: List of MPO tensors
        pos: List of positions where gates are applied
        mpo: Current MPO list
        device: Device for tensor operations
        dtype: Data type for tensors
    
    Returns:
        Updated mpo list
    """
    delta_dim_list = [mpo_t_list[step_function(i, pos)].shape[-1] for i in range(pos[0], pos[-1]+1)]
    delta_pos = fill_seq(pos)

    gate_idx = 0
    for i in range(pos[0], pos[-1]+1):
        mpo_i = mpo[i]
        
        # 判断要作用的是 mpo 还是 delta 张量
        if i in pos:
            gate = mpo_t_list[gate_idx]
            gate_idx += 1
        elif i in delta_pos:
            delta_dim = delta_dim_list[i]
            delta = tc.einsum('il, jk -> ijkl', tc.eys(delta_dim, device=device, dtype=dtype), tc.eys(2, device=device, dtype=dtype))
            gate = delta
        # 判断之前的位置是否已经有张量
        # 如果没有，则直接把要作用的 gate 赋值
        if mpo_i == None:
            mpo[i] = gate
        # 否则，计算要作用的 gate 和已经有的张量的收缩，并在收缩完成后 reshape 为四个脚的张量
        else:
            mpo_i_tmp = tc.einsum('ijkl, akcd-> iajcld', gate, mpo_i)
            un_flatten_shape = mpo_i_tmp.shape
            flatten_shape = [un_flatten_shape[0]*un_flatten_shape[1], un_flatten_shape[2], un_flatten_shape[3], un_flatten_shape[4]*un_flatten_shape[5]]
            mpo[i] = mpo_i_tmp.reshape(flatten_shape)
    
    return mpo

def mpo_act_gates(gates, which_where, n_qubit, mpo=None, device=tc.device('cpu'), dtype=tc.complex64):
    if mpo == None:
        mpo = list(tc.eye(2, device=device, dtype=dtype).reshape([1, 2, 2, 1]) for _ in range(n_qubit))
    for i_pos in which_where:
        gate = gates[i_pos[0]]
        pos = i_pos[1:]
        mpo_t_list = n_body_gate_to_mpo(gate=gate, n=len(pos), device=device, dtype=dtype)
        # print(len(gate_list))
        mpo = process_mpo_tensors(mpo_t_list, pos, mpo, device, dtype)
    return mpo

def inverse_circuit(gates, which_where):
    new_gates = []
    new_which_where = []
    for gate in gates[:]:
        new_gates.append(dagger_gate(gate))
    for pos in which_where[::-1]:
        new_which_where.append(pos[:])
    return new_gates, new_which_where

def trace_mpo(mpo):
    tmp = mpo[0]
    for mpo_i in mpo[1:]:
        assert len(mpo_i.shape) == 4
        tmp = tc.einsum('ijjl, lbcd -> ibcd', tmp, mpo_i)
    trace = tc.einsum('ijjl -> il', tmp).squeeze()
    return trace

def cal_circuit_fidelity(gates_1, which_where_1, gates_2, which_where_2, n_qubit, device=tc.device('cpu'), dtype=tc.complex64):
    mpo = mpo_act_gates(gates_1, which_where=which_where_1, n_qubit=n_qubit, device=device, dtype=dtype)
    gates_2, which_where_2 = inverse_circuit(gates_2, which_where_2)
    mpo = mpo_act_gates(gates=gates_2, which_where=which_where_2, n_qubit=n_qubit, mpo=mpo, device=device, dtype=dtype)
    trace = trace_mpo(mpo)
    print(trace)
    n = 2 ** n_qubit
    fidelity = 1 / (n * (n + 1)) * (n + tc.abs(trace)**2)
    return fidelity

if __name__ == "__main__":
    # Test dagger_gate
    gate = tc.tensor([[1, 2], [3, 4]], dtype=tc.complex64)
    dag = dagger_gate(gate)
    assert np.allclose(dag.resolve_conj().numpy(), \
                       gate.numpy().T.conj()), "dagger_gate failed"

    # Test fill_seq
    assert fill_seq([1, 3, 5]) == [2, 4], "fill_seq failed"
    assert fill_seq([0, 2]) == [1], "fill_seq failed"

    # Test step_function
    assert step_function(3, [1, 2, 4]) == 1, "step_function failed"
    assert step_function(5, [1, 2, 4]) == 2, "step_function failed"
    assert step_function(0, [0, 1, 2]) == 0, "step_function failed"

    # Test process_mpo_tensors (basic shape test)
    # Use dummy mpo_t_list and mpo
    mpo_t_list = [tc.ones((2, 2, 2, 2), dtype=tc.complex64), tc.ones((2, 2, 2, 2), dtype=tc.complex64)]
    pos = [0, 1]
    mpo = list(tc.eye(2, device=tc.device('cpu'), dtype=tc.complex64).reshape([1, 2, 2, 1]) for _ in range(3))
    device = tc.device('cpu')
    dtype = tc.complex64
    out_mpo = process_mpo_tensors(mpo_t_list, pos, mpo, device, dtype)
    assert isinstance(out_mpo, list) and len(out_mpo) == 3, "process_mpo_tensors failed"

    # Test mpo_act_gates (basic call)
    from Library.TensorNetwork import n_body_gate_to_mpo
    gate = tc.eye(4, dtype=tc.complex64).reshape(2,2,2,2)
    gates = [gate]
    which_where = [(0, 0, 1)]
    n_qubit = 3
    mpo = mpo_act_gates(gates, which_where, n_qubit)
    assert isinstance(mpo, list) and len(mpo) == 3, "mpo_act_gates failed"

    # Test inverse_circuit
    gates = [tc.eye(2, dtype=tc.complex64)]
    which_where = [(0, 1)]
    new_gates, new_which_where = inverse_circuit(gates, which_where)
    assert np.allclose(new_gates[0].resolve_conj().numpy(), gates[0].numpy().T.conj()), "inverse_circuit failed on gates"
    assert new_which_where == which_where[::-1], "inverse_circuit failed on which_where"

    # Test cal_circuit_fidelity (basic call, just checks no error)
    a = tc.rand(size=(4, 4), dtype=tc.complex64)
    h = a + a.T.conj()
    rand_gate = tc.matrix_exp(1.j * h).reshape(2,2,2,2)
    gates_1 = [tc.eye(4, dtype=tc.complex64).reshape(2,2,2,2), rand_gate]
    which_where_1 = [(0, 1, 2), (1, 0, 1)]
    gates_2 = [tc.eye(4, dtype=tc.complex64).reshape(2,2,2,2)]
    which_where_2 = [(0, 0, 1), (0, 1, 2)]
    n_qubit = 3
    try:
        fidelity = cal_circuit_fidelity(gates_1, which_where_1, gates_1, which_where_1, n_qubit)
        print(fidelity)
    except Exception as e:
        assert False, f"cal_circuit_fidelity raised an error: {e}"
    print("All tests passed.")
