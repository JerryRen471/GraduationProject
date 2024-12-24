from shutil import which
from qiskit import QuantumCircuit, assemble
from qiskit.visualization import plot_histogram, plot_bloch_vector
from qiskit.quantum_info import Clifford, random_clifford
from math import sqrt, pi
import time
import torch as tc
from Library.Tools import *
import numpy as np

# t1 = time.time()
# cliff = random_clifford(10)
# qc = cliff.to_circuit()
# t2 = time.time()
# print(t2-t1)
# print(qc)

def get_quantum_gates_and_qubits(qc):
    gates_info = []
    for gate in qc.data:
        gate_name = gate.name  # 获取量子门的名称
        qubits = gate.qubits  # 获取作用的比特
        gates_info.append((gate_name, [qc.qubits.index(qubit) for qubit in qubits]))  # 记录量子门及其作用的比特索引
    return gates_info

def convert_qiskit_circuit_to_usual_gates(qiskit_circuit):
    x_gate = tc.tensor([[0, 1], [1, 0]], dtype=tc.complex64)
    y_gate = tc.tensor([[0, -1j], [1j, 0]], dtype=tc.complex64)
    z_gate = tc.tensor([[1, 0], [0, -1]], dtype=tc.complex64)
    s_gate = tc.tensor([[1, 0], [0, 1j]], dtype=tc.complex64)
    sdg_gate = tc.tensor([[1, 0], [0, -1j]], dtype=tc.complex64)
    h_gate = tc.tensor([[1/(2**0.5), 1/(2**0.5)], [1/(2**0.5), -1/(2**0.5)]], dtype=tc.complex64)

    cnot_gate = tc.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=tc.complex64).reshape(2, 2, 2, 2)
    swap_gate = tc.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=tc.complex64).reshape(2, 2, 2, 2)

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

def generate_unitary_2_design_states(number:int, n_qubit:int):
    state = tc.zeros((2**n_qubit), dtype=tc.complex64)
    state[0] = 1 + 0.j
    state = state.reshape([2] * n_qubit)
    state = state.unsqueeze(0)
    state_list = list()
    for i in range(number):
        cliff = random_clifford(n_qubit)
        qc = cliff.to_circuit()
        gate_names, gate_list, which_where = convert_qiskit_circuit_to_usual_gates(qc)
        rand_state = pure_states_evolution(state, gate_list, which_where)
        state_list.append(rand_state)
    u2d_states = tc.cat(state_list, dim=0)
    return u2d_states

def measure(state):
    shape = state.shape
    state = state.reshape(-1)
    probabilities = tc.abs(state) ** 2
    # 根据概率分布进行测量
    measured_index = tc.multinomial(probabilities, 1).item()  # 进行测量
    measured_state = tc.zeros_like(state)
    measured_state[measured_index] = 1  # 设置对应的量子态为1

    return measured_state.reshape(shape)

def sample_classical_shadow(aim_state, n_qubit):
    cliff = random_clifford(n_qubit)
    qc = cliff.to_circuit()
    gate_names, gate_list, which_where = convert_qiskit_circuit_to_usual_gates(qc)
    U_state = pure_states_evolution(aim_state, gate_list, which_where)
    b_state = measure(U_state)

    inverse_qc = qc.inverse()
    gate_names, gate_list, which_where = convert_qiskit_circuit_to_usual_gates(inverse_qc)
    sigma_state = pure_states_evolution(b_state, gate_list, which_where)
    sigma = tc.einsum('a, b -> ab', sigma_state.reshape(-1), sigma_state.reshape(-1).conj())
    Eye = tc.eye(2**n_qubit, dtype=tc.complex64)
    rho_sample = (2**n_qubit + 1) * sigma - Eye
    return rho_sample

def matrix_sqrt(A, tol=1e-6):
    """
    计算复数矩阵 A 的平方根，使用特征值分解。
    
    参数：
    A (torch.Tensor): 复数矩阵。
    
    返回：
    torch.Tensor: 复数矩阵 A 的平方根。
    """
    # 进行特征值分解
    eigenvalues, eigenvectors = tc.linalg.eigh(A)
    
    # 对特征值进行裁剪：小于 tol 的特征值设置为零
    eigenvalues = tc.where(tc.abs(eigenvalues) < tol, tc.tensor(0.0, dtype=eigenvalues.dtype), eigenvalues)

    # 计算特征值的平方根
    eigenvalues_sqrt = tc.sqrt(eigenvalues)
    
    # 构造平方根矩阵
    A_sqrt = eigenvectors @ tc.diag(eigenvalues_sqrt).to(dtype=tc.complex64) @ eigenvectors.T.conj()
    
    return A_sqrt

def fidelity_density_matrices(rho, sigma):
    """
    计算两个复数密度矩阵 rho 和 sigma 之间的保真度。
    
    参数：
    rho (torch.Tensor): 复数密度矩阵 rho。
    sigma (torch.Tensor): 复数密度矩阵 sigma。
    
    返回：
    float: 两个密度矩阵之间的保真度。
    """
    # 计算 sqrt(rho)
    sqrt_rho = matrix_sqrt(rho)  # 使用自定义的 matrix_sqrt 函数计算复数矩阵的平方根
    
    # 计算 sqrt(rho) * sigma * sqrt(rho)
    product = tc.matmul(tc.matmul(sqrt_rho, sigma), sqrt_rho)
    
    # 计算 sqrt(sqrt(rho) * sigma * sqrt(rho))
    sqrt_product = matrix_sqrt(product)  # 计算 sqrt(sqrt(rho) * sigma * sqrt(rho))
    
    # 计算保真度 F = Tr(sqrt(sqrt(rho) * sigma * sqrt(rho)))
    return tc.abs(tc.trace(sqrt_product))**2

def multi_mags_from_rho(rho, length):
    # length = int(np.log2(rho.shape[0]))
    rho = rho.reshape([2] * length *2)
    sigma_x = tc.tensor([[0, 1], [1, 0]], dtype=rho.dtype)
    sigma_y = tc.tensor([[0, -1j], [1j, 0]], dtype=rho.dtype)
    sigma_z = tc.tensor([[1, 0], [0, -1]], dtype=rho.dtype)
    which_ops = [sigma_x, sigma_y, sigma_z]
    mags = tc.zeros((3, length), device=rho.device)
    for i in range(length):
        perm = [_ for _ in range(2 * length)]
        perm[0] = i
        perm[i] = 0
        perm[length] = i + length
        perm[length + i] = length
        rho_tmp = rho.permute(perm)
        rho_tmp = rho_tmp.reshape([2, 2**(length-1), 2, 2**(length-1)])
        reduced_rho = tc.einsum('iaja->ij', rho_tmp)
        for s in range(3):
            # print(rho)
            # print(which_ops[s])
            obs = which_ops[s]
            mags[s, i] = tc.einsum('bc,cb->', reduced_rho.type(tc.complex64), obs.type(tc.complex64)).real
    return mags


n_qubit = 5
# state = tc.rand((2**n_qubit), dtype=tc.complex64)
state = tc.rand(2, dtype=tc.complex64)
for i in range(n_qubit-1):
    state = tc.einsum('a, b -> ab', state, tc.rand(2, dtype=tc.complex64))
    state = state.reshape(-1)
state = state / tc.norm(state)
state = state.reshape([2] * n_qubit)
state = state.unsqueeze(0)
print(state.shape)

from Library.PhysModule import multi_mags_from_states
multi_mags_real = multi_mags_from_states(state)

rho_sample = 0
avg_rho = 0
number = 800*10
K = 800
mean_fidelity_list = []
multi_mags_list = []
t1 = time.time()
# rho = tc.einsum('a, b -> ab', state.reshape(-1), state.reshape(-1).conj())
for i in range(number):
    rho_sample = sample_classical_shadow(state, n_qubit)
    avg_rho = avg_rho + rho_sample
    if i % K == K-1:
        avg_rho = avg_rho / K
        fidelity = tc.einsum('a, ab, b ->', state.reshape(-1).conj(), avg_rho, state.reshape(-1)).real
        mean_fidelity_list.append(fidelity)
        multi_mags = multi_mags_from_rho(avg_rho, length=n_qubit)
        multi_mags_list.append(multi_mags)
        avg_rho = 0
t2 = time.time()
print(t2 - t1)
print()
print(mean_fidelity_list)
median_fidelity = np.median(mean_fidelity_list)
print("Median of mean fidelity list:", median_fidelity)
multi_mags_sample = tc.stack(multi_mags_list, dim=0)
print(multi_mags_sample.shape)
median_multi_mags, _ = tc.median(multi_mags_sample, dim=0)
print(median_multi_mags - multi_mags_real)
# t1 = time.time()
# states = generate_unitary_2_design_states(100, 10)
# print(states.shape)
# t2 = time.time()
# print(t2 - t1)

# # 使用示例
# gates_info = get_quantum_gates_and_qubits(qc)
# for gate_name, qubit_indices in gates_info:
#     print(f"Gate: {gate_name}, Qubits: {qubit_indices}")
# print(len(gates_info))

# x_gate = tc.tensor([[0, 1], [1, 0]], dtype=tc.complex64)
# y_gate = tc.tensor([[0, -1j], [1j, 0]], dtype=tc.complex64)
# z_gate = tc.tensor([[1, 0], [0, -1]], dtype=tc.complex64)
# s_gate = tc.tensor([[1, 0], [0, 1j]], dtype=tc.complex64)
# h_gate = tc.tensor([[1/(2**0.5), 1/(2**0.5)], [1/(2**0.5), -1/(2**0.5)]], dtype=tc.complex64)

# cnot_gate = tc.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=tc.complex64)
# swap_gate = tc.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=tc.complex64)

# gate_dict = {
#     'x': x_gate,
#     'y': y_gate,
#     'z': z_gate,
#     's': s_gate,
#     'h': h_gate,
#     'cx': cnot_gate,
#     'swap': swap_gate
# }

# for key, value in gate_dict.items():
#     print(key, value)
#     print()

# gate_list = list(gate_dict.values())
# gate_names = list(gate_dict.keys())

# which_where = []
# for gate_name, qubit_indices in gates_info:
#     print(f"Gate: {gate_name}, Qubits: {qubit_indices}")
#     which_where.append([gate_names.index(gate_name)] + qubit_indices)



