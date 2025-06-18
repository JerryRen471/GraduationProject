# Test cal_gate_fidelity_from_mpo function

import torch as tc
import numpy as np
from DataProcess import cal_gate_fidelity_from_mpo
from Library.TensorNetwork import rand_mps_pack

# Test 1: Same gates (should give fidelity = 1)
cnot = tc.tensor([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=tc.complex64).reshape([2, 2, 2, 2])
qc_gates = [cnot]
evol_gates = [cnot]
qc_which_where = [[0, 0, 1]]
evol_which_where = [[0, 0, 1]]
fidelity = cal_gate_fidelity_from_mpo(qc_gates, evol_gates, qc_which_where, evol_which_where, num_basis=10)
print(f"Fidelity between identical gates: {fidelity:.6f}")

# Test 2: Different gates
swap = tc.tensor([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=tc.complex64).reshape([2, 2, 2, 2])
qc_gates = [cnot]
evol_gates = [swap]
fidelity = cal_gate_fidelity_from_mpo(qc_gates, evol_gates, qc_which_where, evol_which_where, num_basis=10)
print(f"Fidelity between CNOT and SWAP gates: {fidelity:.6f}")

# Test 3: Test with different number of basis states
basis_sizes = [i * 10 for i in range(1, 11)]
for num_basis in basis_sizes:
    fidelity = cal_gate_fidelity_from_mpo(qc_gates, evol_gates, qc_which_where, evol_which_where, num_basis=num_basis)
    print(f"Fidelity with {num_basis} basis states: {fidelity:.6f}")

# Test 4: Test with multiple gates
hadamard = tc.tensor([[1, 1],
                      [1, -1]], dtype=tc.complex64) / np.sqrt(2)
qc_gates = [hadamard, cnot]
evol_gates = [hadamard, cnot]
qc_which_where = [[0, 0], [1, 0, 1]]
evol_which_where = [[0, 0], [1, 0, 1]]
fidelity = cal_gate_fidelity_from_mpo(qc_gates, evol_gates, qc_which_where, evol_which_where, num_basis=10)
print(f"Fidelity for multiple gates: {fidelity:.6f}")