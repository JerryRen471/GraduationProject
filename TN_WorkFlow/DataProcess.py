import os
import torch as tc
import numpy as np
import scipy.linalg
from copy import deepcopy

from matplotlib import pyplot as plt
import pandas as pd
import Library.TensorNetwork as TN
from Library.TensorNetwork import TensorTrain, inner_mps, multi_mags_from_mps, inner_product, rand_mps_pack
from Library.PhysModule import spin_operators

def merge_TN_pack(TN_pack_list):
    '''
    The TN_pack in TN_pack_list must have the same structure, including length, chi, center, device, dtype, etc.
    '''
    new_node_list = deepcopy(TN_pack_list[0].node_list)
    for i in range(len(TN_pack_list) - 1):
        assert TN_pack_list[i].length == TN_pack_list[i+1].length
        assert TN_pack_list[i].chi == TN_pack_list[i+1].chi
        assert TN_pack_list[i].center == TN_pack_list[i+1].center
        assert TN_pack_list[i].device == TN_pack_list[i+1].device
        assert TN_pack_list[i].dtype == TN_pack_list[i+1].dtype
        for j, node in enumerate(TN_pack_list[i+1].node_list):
            new_node_list[j] = tc.cat([new_node_list[j], node], dim=0)
    merged_TN = TN.TensorTrain_pack(tensor_packs=new_node_list, length=TN_pack_list[0].length, chi=TN_pack_list[0].chi, center=TN_pack_list[0].center, device=TN_pack_list[0].device, dtype=TN_pack_list[0].dtype, initialize=False)
    return merged_TN

def cal_gate_fidelity(E:tc.Tensor, U:tc.Tensor):
    """
    Calculate gate fidelity between two quantum circuits.
    
    Args:
        E: Circuit matrix
        U: Target evolution matrix
    
    Returns:
        Gate fidelity
    """
    n = E.shape[0]
    trace = tc.einsum('aa', U.T.conj() @ E)
    gate_fidelity = 1/(n*(n+1))*(n + tc.abs(trace)**2)
    return gate_fidelity

def cal_gate_fidelity_from_mpo(qc_gates, evol_gates, which_where, num_basis=10):
    """
    Calculate gate fidelity between two quantum circuits represented as gate lists
    using basis states.
    
    Args:
        qc_gates: List of gates representing the learned quantum circuit
        evol_gates: List of gates representing the target evolution
        which_where: List specifying which gates act on which spins
                    Format: [[which_gate, spin1, spin2, ...], ...]
        num_basis: Number of basis states to use
    
    Returns:
        Gate fidelity
    """
    # Get device and dtype from the first gate
    device = qc_gates[0].device
    dtype = qc_gates[0].dtype
    
    # Determine the number of qubits from the which_where list
    n_qubits = max([max(pos[1:]) for pos in which_where]) + 1
    
    # Create computational basis states
    basis_states = []
    for i in range(min(num_basis, 2**n_qubits)):
        # Create basis state
        basis_state = rand_mps_pack(1, n_qubits, chi=10, device=device, dtype=dtype)
        basis_states.append(basis_state)
    
    # Apply both circuits to the basis states
    qc_outputs = []
    evol_outputs = []
    for state in basis_states:
        # Apply learned circuit
        qc_state = deepcopy(state)
        for n in range(len(which_where)):
            qc_state.act_n_body_gate(qc_gates[which_where[n][0]], which_where[n][1:])
        qc_outputs.append(qc_state)
        
        # Apply target evolution
        evol_state = deepcopy(state)
        for n in range(len(which_where)):
            evol_state.act_n_body_gate(evol_gates[which_where[n][0]], which_where[n][1:])
        evol_outputs.append(evol_state)
    
    # Calculate overlaps between outputs
    overlaps = []
    for i in range(len(basis_states)):
        overlap = inner_mps_pack(qc_outputs[i], evol_outputs[i])
        overlaps.append(tc.abs(overlap).item())
    
    # Calculate average overlap
    avg_overlap = np.mean(overlaps)
    
    # Calculate gate fidelity using the formula: F_gate = (d*F_avg^2 - 1)/(d-1)
    # where d is the dimension of the Hilbert space (2^n_qubits)
    d = 2**n_qubits
    gate_fidelity = (d * avg_overlap**2 - 1) / (d - 1)
    
    return gate_fidelity

def cal_similarity(E:tc.Tensor, U:tc.Tensor):
    '''
    Calculate similarity between two quantum circuits.
    
    Args:
        E: Circuit matrix
        U: Target evolution matrix
    
    Returns:
        Similarity measure
    '''
    a = tc.norm(E - U)
    b = 2 * tc.norm(U)
    s = 1 - a/b
    return s

def cal_similarity_from_mpo(qc_mpo, evol_mpo, num_basis=10):
    """
    Calculate similarity between two MPO representations of quantum circuits
    using basis states.
    
    Args:
        qc_mpo: TensorTrain representing the learned quantum circuit
        evol_mpo: TensorTrain representing the target evolution
        num_basis: Number of basis states to use
    
    Returns:
        Similarity measure
    """
    n_qubits = qc_mpo.length
    device = qc_mpo.device
    dtype = qc_mpo.dtype
    
    # Create computational basis states
    basis_states = []
    for i in range(min(num_basis, 2**n_qubits)):
        # Create basis state
        basis_state = TN.rand_mps(1, n_qubits, qc_mpo.chi, device=device, dtype=dtype)
        basis_states.append(basis_state)
    
    # Apply both circuits to the basis states
    qc_outputs = []
    evol_outputs = []
    for state in basis_states:
        qc_outputs.append(qc_mpo(state))
        evol_outputs.append(evol_mpo(state))
    
    # Calculate differences between outputs
    differences = []
    for i in range(len(basis_states)):
        # Calculate norm of difference between outputs
        diff = qc_outputs[i] - evol_outputs[i]
        diff_norm = tc.norm(diff).item()
        differences.append(diff_norm)
    
    # Calculate average difference
    avg_diff = np.mean(differences)
    
    # Calculate similarity
    # We need to estimate the norm of the evolution operator
    # This is a rough estimate based on the outputs
    evol_norms = []
    for i in range(len(basis_states)):
        evol_norm = tc.norm(evol_outputs[i]).item()
        evol_norms.append(evol_norm)
    
    avg_evol_norm = np.mean(evol_norms)
    b = 2 * avg_evol_norm
    
    # Calculate similarity
    s = 1 - avg_diff/b
    
    return s

def normalize_pi(n):
    return n - tc.div(n + tc.pi, 2*tc.pi, rounding_mode='trunc') * 2*tc.pi

def cal_spectrum(mat:tc.Tensor):
    """
    Calculate the spectrum of a matrix.
    
    Args:
        mat: Input matrix
    
    Returns:
        Sorted energy spectrum
    """
    energy = tc.log(tc.linalg.eigvals(mat))/1.j
    energy = energy.real
    energy = normalize_pi(energy)
    energy, ind = tc.sort(energy)
    return energy

def cal_spectrum_from_mpo(mpo, num_basis=10, time_interval=1.0):
    """
    Estimate the spectrum of an MPO operator using basis states.
    
    Args:
        mpo: TensorTrain representing the operator
        num_basis: Number of basis states to use
        time_interval: Time interval for the evolution
    
    Returns:
        Estimated energy spectrum
    """
    n_qubits = mpo.length
    device = mpo.device
    dtype = mpo.dtype
    
    # Create computational basis states
    basis_states = []
    for i in range(min(num_basis, 2**n_qubits)):
        # Create basis state
        basis_state = TN.rand_mps(1, n_qubits, mpo.chi, device=device, dtype=dtype)
        basis_states.append(basis_state)
    
    # Apply the operator to the basis states
    outputs = []
    for state in basis_states:
        outputs.append(mpo(state))
    
    # Calculate overlaps between input and output states
    overlaps = []
    for i in range(len(basis_states)):
        overlap = inner_product(basis_states[i], outputs[i])
        overlaps.append(overlap.item())
    
    # Convert overlaps to energies
    energies = np.angle(overlaps) / time_interval
    
    # Sort energies
    energies = np.sort(energies)
    
    return energies

def cal_hamiltonian_from_mpo(mpo, num_basis=10, time_interval=1.0):
    """
    Estimate the Hamiltonian of an MPO operator using basis states.
    
    Args:
        mpo: TensorTrain representing the operator
        num_basis: Number of basis states to use
        time_interval: Time interval for the evolution
    
    Returns:
        Estimated Hamiltonian matrix (sparse representation)
    """
    n_qubits = mpo.length
    device = mpo.device
    dtype = mpo.dtype
    
    # Create a small set of basis states
    basis_size = min(num_basis, 2**n_qubits)
    basis_states = []
    for i in range(basis_size):
        # Create basis state
        basis_state = TN.rand_mps(1, n_qubits, mpo.chi, device=device, dtype=dtype)
        basis_states.append(basis_state)
    
    # Apply the operator to the basis states
    outputs = []
    for state in basis_states:
        outputs.append(mpo(state))
    
    # Calculate matrix elements
    H = np.zeros((basis_size, basis_size), dtype=np.complex128)
    for i in range(basis_size):
        for j in range(basis_size):
            # Calculate matrix element <i|H|j>
            overlap = inner_product(basis_states[i], outputs[j])
            H[i, j] = overlap.item() / time_interval
    
    # Make the matrix Hermitian
    H = (H + H.conj().T) / 2
    
    return H

def write_to_csv(data, csv_file_path, subset):
    """
    向CSV文件写入数据，可以指定接受的数据所对应的列。

    参数:
    data (dict): 要写入的数据字典，其中键为列名，值为对应的数据。
    csv_file_path (str): CSV文件的路径。
    """
    # 将数据转换为 DataFrame
    new_df = pd.DataFrame(data)

    # 检查文件是否存在
    if os.path.exists(csv_file_path):
        # 加载现有的 CSV 数据
        existing_data = pd.read_csv(csv_file_path)

        # 将新数据与现有数据合并
        combined_data = pd.concat([existing_data, new_df], ignore_index=True)
        combined_data = combined_data.sort_values(subset)

        # 去重，保留最后出现的行
        # combined_data = combined_data.drop_duplicates()
        #     subset=subset, keep='last'
        # )
    else:
        # 文件不存在，直接使用新数据  
        combined_data = new_df
    
    # 保存更新后的数据到 CSV 文件
    combined_data.to_csv(csv_file_path, index=False, mode='w')

def cal_mag_evolution(qc_tn, evol_tn, initial_state, time_steps=10, time_interval=0.1):
    """
    Calculate the time evolution of magnetizations starting from a given state,
    using both the quantum circuit and evolution separately.
    
    Args:
        qc_tn: TensorTrain_pack representing the learned quantum circuit
        evol_tn: TensorTrain_pack representing the target evolution
        initial_state: Initial state as a TensorTrain_pack
        time_steps: Number of time steps to evolve
        time_interval: Time interval between steps
    
    Returns:
        Dictionary containing:
        - 'qc_mags': List of magnetization values for the quantum circuit
        - 'evol_mags': List of magnetization values for the target evolution
        - 'time_points': List of time points
    """
    from Library.TensorNetwork import inner_mps_pack
    from Library.PhysModule import spin_operators
    
    # Get device and dtype from the initial state
    device = initial_state.device
    dtype = initial_state.dtype
    
    # Get spin operators
    op = spin_operators('half', device=device)
    spins = [op['sx'], op['sy'], op['sz']]
    
    # Initialize lists to store results
    qc_mags = []
    evol_mags = []
    time_points = []
    
    # Create copies of the initial state
    qc_state = deepcopy(initial_state)
    evol_state = deepcopy(initial_state)
    
    # Calculate initial magnetizations
    qc_mags.append(multi_mags_from_mps(qc_state, spins))
    evol_mags.append(multi_mags_from_mps(evol_state, spins))
    time_points.append(0.0)
    
    # Evolve the states for the specified number of time steps
    for t in range(1, time_steps + 1):
        # Apply quantum circuit to the state
        for n in range(len(qc_tn.which_where)):
            qc_state.act_n_body_gate(qc_tn.gates[qc_tn.which_where[n][0]], qc_tn.which_where[n][1:])
        
        # Apply target evolution to the state
        for n in range(len(evol_tn.which_where)):
            evol_state.act_n_body_gate(evol_tn.gates[evol_tn.which_where[n][0]], evol_tn.which_where[n][1:])
        
        # Calculate magnetizations
        qc_mags.append(multi_mags_from_mps(qc_state, spins))
        evol_mags.append(multi_mags_from_mps(evol_state, spins))
        time_points.append(t * time_interval)
    
    # Convert to numpy arrays for easier plotting
    qc_mags = [mag.numpy() for mag in qc_mags]
    evol_mags = [mag.numpy() for mag in evol_mags]
    
    return {
        'qc_mags': qc_mags,
        'evol_mags': evol_mags,
        'time_points': time_points
    }

def plot_mag_evolution(evolution_data, save_path=None):
    """
    Plot the magnetization evolution data.
    
    Args:
        evolution_data: Dictionary returned by cal_mag_evolution
        save_path: Path to save the plot (if None, the plot is not saved)
    """
    import matplotlib.pyplot as plt
    
    qc_mags = evolution_data['qc_mags']
    evol_mags = evolution_data['evol_mags']
    time_points = evolution_data['time_points']
    
    # Create figure with subplots for each magnetization component
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Magnetization Evolution')
    
    # Plot each magnetization component
    for i, (ax, label) in enumerate(zip(axes, ['Sx', 'Sy', 'Sz'])):
        # Extract the i-th component from each magnetization tensor
        qc_values = [mag[i] for mag in qc_mags]
        evol_values = [mag[i] for mag in evol_mags]
        
        # Plot the values
        ax.plot(time_points, qc_values, 'b-', label='Quantum Circuit')
        ax.plot(time_points, evol_values, 'r--', label='Target Evolution')
        
        # Add labels and legend
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{label} Magnetization')
        ax.legend()
        ax.grid(True)
    
    # Adjust layout and save if path is provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    return fig

def main(
    qc_tn=None, 
    evol_tn=None,
    qc_mat=None, 
    evol_mat=None, 
    results=None,
    pic_path=None,
    csv_file_path=None,
    **para):
    
    """
    Process and analyze quantum circuit data, supporting both tensor network and matrix representations.
    
    Args:
        qc_tn: TensorTrain representing the learned quantum circuit
        evol_tn: TensorTrain representing the target evolution
        qc_mat: Matrix representation of the learned quantum circuit (if available)
        evol_mat: Matrix representation of the target evolution (if available)
        results: Dictionary containing training results
        pic_path: Path to save plots
        csv_file_path: Path to save CSV data
        **para: Additional parameters
    """
    
    # Initialize data dictionary with parameters
    data = dict(para)
    
    # Process training results if available
    if results is not None:
        train_fide = results['train_fide']
        test_fide = results['test_fide']
        train_loss = results['train_loss']
        test_loss = results['test_loss']
        x = list(range(0, len(train_loss)))

        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(x, train_loss, label='train loss')
        plt.plot(x, test_loss, label='test loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training and Test Loss')
        plt.savefig(pic_path+'/loss_num{:d}.svg'.format(para['evol_num']))
        plt.close()

        # Plot fidelity curves
        plt.figure(figsize=(10, 6))
        plt.plot(x, train_fide, label='train fidelity')
        plt.plot(x, test_fide, label='test fidelity')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('fidelity')
        plt.title('Training and Test Fidelity')
        plt.savefig(pic_path+'/fidelity_num{:d}.svg'.format(para['evol_num']))
        plt.close()

        # Add final metrics to data dictionary
        data['train_loss'] = [float(train_loss[-1])]
        data['test_loss'] = [float(test_loss[-1])]
        data['train_fide'] = [float(train_fide[-1])]
        data['test_fide'] = [float(test_fide[-1])]
    
    # Initialize return list
    return_list = []
    
    # Calculate metrics based on available representations
    if qc_tn is not None and evol_tn is not None:
        # Use tensor network representation
        print("Using tensor network representation for analysis...")
        
        # Calculate gate fidelity
        gate_fidelity = cal_gate_fidelity_from_mpo(qc_tn.gates, evol_tn.gates, qc_tn.which_where, num_basis=10)
        data['gate_fidelity'] = [float(gate_fidelity)]
        return_list.append(gate_fidelity)
        print(f"Gate fidelity: {gate_fidelity:.6f}")
        
        # Calculate similarity
        similarity = cal_similarity_from_mpo(qc_tn, evol_tn)
        data['similarity'] = [float(similarity)]
        return_list.append(similarity)
        print(f"Similarity: {similarity:.6f}")
        
        # Calculate magnetization evolution if requested
        if para.get('calc_mag_evolution', False):
            # Create an initial state
            n_qubits = qc_tn.length
            initial_state = rand_mps_pack(1, n_qubits, chi=10, device=qc_tn.device, dtype=qc_tn.dtype)
            
            # Calculate magnetization evolution
            time_steps = para.get('time_steps', 10)
            time_interval = para.get('time_interval', 0.1)
            evolution_data = cal_mag_evolution(qc_tn, evol_tn, initial_state, time_steps, time_interval)
            
            # Plot and save the evolution
            if pic_path:
                plot_mag_evolution(evolution_data, save_path=pic_path+'/mag_evolution_num{:d}.svg'.format(para['evol_num']))
            
            # Calculate average magnetization difference over time
            mag_diffs = []
            for i in range(len(evolution_data['time_points'])):
                qc_mag = evolution_data['qc_mags'][i]
                evol_mag = evolution_data['evol_mags'][i]
                mag_diff = np.mean(np.abs(qc_mag - evol_mag))
                mag_diffs.append(mag_diff)
            
            # Add average magnetization difference to data
            avg_mag_diff = np.mean(mag_diffs)
            data['avg_mag_diff'] = [float(avg_mag_diff)]
            return_list.append(avg_mag_diff)
            print(f"Average magnetization difference: {avg_mag_diff:.6f}")
        
        # Calculate spectrum if time interval is provided
        if para.get('time_interval', 0) != 0:
            time_interval = para['time_interval']
            
            # Calculate energy spectra
            qc_energy = cal_spectrum_from_mpo(qc_tn, time_interval=time_interval)
            evol_energy = cal_spectrum_from_mpo(evol_tn, time_interval=time_interval)
            
            # Calculate spectrum difference
            spectrum_diff = tc.var(tc.tensor(qc_energy - evol_energy))
            data['spectrum_diff'] = [float(spectrum_diff)]
            return_list.append(spectrum_diff)
            print(f"Spectrum difference: {spectrum_diff:.6f}")
            
            # Calculate Hamiltonians
            H_qc = cal_hamiltonian_from_mpo(qc_tn, time_interval=time_interval)
            H_evol = cal_hamiltonian_from_mpo(evol_tn, time_interval=time_interval)
            
            # Plot Hamiltonian heatmaps
            plt.figure(figsize=(10, 8))
            plt.imshow(np.abs(H_qc), cmap='Greys', interpolation='nearest', vmin=np.min(np.abs(H_qc)), vmax=np.max(np.abs(H_evol)))
            plt.colorbar(label='Absolute Value')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            plt.title('Heatmap of H_qc (MPO-based)')
            plt.savefig(pic_path+'/H_qc_heatmap_num{:d}.svg'.format(para['evol_num']))
            plt.close()
            
            plt.figure(figsize=(10, 8))
            plt.imshow(np.abs(H_evol), cmap='Greys', interpolation='nearest', vmin=np.min(np.abs(H_qc)), vmax=np.max(np.abs(H_evol)))
            plt.colorbar(label='Absolute Value')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            plt.title('Heatmap of H_evol (MPO-based)')
            plt.savefig(pic_path+'/H_evol_heatmap_num{:d}.svg'.format(para['evol_num']))
            plt.close()
            
            # Calculate Hamiltonian difference
            abs_diff = np.abs(H_qc - H_evol)
            plt.figure(figsize=(10, 8))
            plt.imshow(abs_diff, cmap='Greys', interpolation='nearest')
            plt.colorbar(label='Absolute Difference Value')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            plt.title('Heatmap of Absolute Difference (MPO-based)')
            plt.savefig(pic_path+'/abs_diff_heatmap_num{:d}.svg'.format(para['evol_num']))
            plt.close()
            
            H_diff = np.mean(abs_diff)
            data['H_diff'] = [float(H_diff)]
            return_list.append(H_diff)
            print(f"Hamiltonian difference: {H_diff:.6f}")
            
            # Calculate magnetization differences
            op = spin_operators('half', device=qc_tn.device)
            spins = [op['sx'], op['sy'], op['sz']]
            
            # Create a few basis states
            n_qubits = qc_tn.length
            basis_states = []
            
            # Create computational basis states
            for i in range(min(10, 2**n_qubits)):
                # Create basis state
                basis_state = TN.rand_mps(1, n_qubits, qc_tn.chi, device=qc_tn.device, dtype=qc_tn.dtype)
                basis_states.append(basis_state)
            
            # Calculate magnetization differences
            mag_diffs = []
            
            for state in basis_states:
                # Apply both circuits
                qc_output = qc_tn(state)
                evol_output = evol_tn(state)
                
                # Calculate magnetization
                qc_mags = multi_mags_from_mps(qc_output, spins)
                evol_mags = multi_mags_from_mps(evol_output, spins)
                
                # Calculate difference
                mag_diff = tc.norm(qc_mags - evol_mags).item()
                mag_diffs.append(mag_diff)
            
            # Average magnetization difference
            avg_mag_diff = np.mean(mag_diffs)
            data['mag_diff'] = [float(avg_mag_diff)]
            return_list.append(avg_mag_diff)
            print(f"Average magnetization difference: {avg_mag_diff:.6f}")
    
    elif qc_mat is not None and evol_mat is not None:
        # Use matrix representation
        print("Using matrix representation for analysis...")
        
        # Calculate gate fidelity
        gate_fidelity = cal_gate_fidelity(qc_mat, evol_mat)
        data['gate_fidelity'] = [float(gate_fidelity)]
        return_list.append(gate_fidelity)
        print(f"Gate fidelity: {gate_fidelity:.6f}")
        
        # Calculate similarity
        similarity = cal_similarity(qc_mat, evol_mat)
        data['similarity'] = [float(similarity)]
        return_list.append(similarity)
        print(f"Similarity: {similarity:.6f}")
        
        # Calculate spectrum if time interval is provided
        if para.get('time_interval', 0) != 0:
            time_interval = para['time_interval']
            
            # Calculate energy spectra
            qc_energy = cal_spectrum(qc_mat) / time_interval
            evol_energy = cal_spectrum(evol_mat) / time_interval
            
            # Calculate spectrum difference
            spectrum_diff = tc.var(qc_energy - evol_energy)
            data['spectrum_diff'] = [float(spectrum_diff)]
            return_list.append(spectrum_diff)
            print(f"Spectrum difference: {spectrum_diff:.6f}")
            
            # Calculate Hamiltonians
            qc_mat_np = qc_mat.numpy()
            evol_mat_np = evol_mat.numpy()
            H_qc = scipy.linalg.logm(qc_mat_np)/ 1.j / time_interval
            H_evol = scipy.linalg.logm(evol_mat_np)/ 1.j / time_interval
            
            # Plot Hamiltonian heatmaps
            plt.figure(figsize=(10, 8))
            plt.imshow(np.abs(H_qc), cmap='Greys', interpolation='nearest', vmin=np.min(np.abs(H_qc)), vmax=np.max(np.abs(H_evol)))
            plt.colorbar(label='Absolute Value')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            plt.title('Heatmap of H_qc')
            plt.savefig(pic_path+'/H_qc_heatmap_num{:d}.svg'.format(para['evol_num']))
            plt.close()
            
            plt.figure(figsize=(10, 8))
            plt.imshow(np.abs(H_evol), cmap='Greys', interpolation='nearest', vmin=np.min(np.abs(H_qc)), vmax=np.max(np.abs(H_evol)))
            plt.colorbar(label='Absolute Value')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            plt.title('Heatmap of H_evol')
            plt.savefig(pic_path+'/H_evol_heatmap_num{:d}.svg'.format(para['evol_num']))
            plt.close()
            
            # Calculate Hamiltonian difference
            abs_diff = np.abs(H_qc - H_evol)
            plt.figure(figsize=(10, 8))
            plt.imshow(abs_diff, cmap='Greys', interpolation='nearest')
            plt.colorbar(label='Absolute Difference Value')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            plt.title('Heatmap of Absolute Difference')
            plt.savefig(pic_path+'/abs_diff_heatmap_num{:d}.svg'.format(para['evol_num']))
            plt.close()
            
            H_diff = np.mean(abs_diff)
            data['H_diff'] = [float(H_diff)]
            return_list.append(H_diff)
            print(f"Hamiltonian difference: {H_diff:.6f}")
    
    # Write data to CSV
    if csv_file_path is not None:
        write_to_csv(data, csv_file_path, subset=list(para.keys()))
    
    return tuple(return_list)