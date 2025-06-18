import Library.TensorNetwork as TN
import torch as tc
import numpy as np
from typing import List, Dict, Union, Any, Tuple

def multiply_list(list: List[Union[int, float]]) -> Union[int, float]:
    """Multiply all elements in a list together.
    
    Args:
        list: A list of numbers to multiply.
        
    Returns:
        The product of all numbers in the list.
    """
    res = 1
    for i in list:
        res *= i
    return res

def TEBD(gate_dict: Dict[str, List[Any]], tau: float, time_tot: float, print_time: float, 
         init_mps: TN.TensorTrain_pack, obs: List[List[Tuple[Any, Union[int, List[int]]]]]) -> Tuple[List[TN.TensorTrain_pack], List[List[complex]]]:
    """Time Evolution Block Decimation (TEBD) algorithm implementation.
    
    Args:
        gate_dict: Dictionary containing gate information with keys:
            - 'gate_i': List of quantum gates
            - 'pos': List of positions where gates are applied
        tau: Time step size
        time_tot: Total evolution time
        print_time: Time interval for saving results
        init_mps: Initial Matrix Product State
        obs: List of observables to measure, each in format [[operator, position], ...]
            where position can be single integer or list of integers for multi-site observables
            
    Returns:
        Tuple containing:
        - List of evolved MPS states at specified time intervals
        - List of observable expectation values at specified time intervals
    """
    gate_list = gate_dict['gate_i']
    pos = gate_dict['pos']
    total_t_list = [i for i in range(1, int(time_tot // tau) + 1)]
    print_t = int(print_time // tau)
    mps_evol = []
    obs_evol = []

    for t in total_t_list:
        for i, gate_i in enumerate(gate_list):
            init_mps = init_mps.act_n_body_gate_sequence(gate_i, pos[i], set_center=0)
        if t % print_t == 0:
            mps_evol.append(TN.copy_from_mps_pack(init_mps))
            # Calculate expected values of observables
            obs_values = []
            for op, op_pos in obs:
                if isinstance(op_pos, int):
                    # Single-site observable
                    obs_value = TN.inner_mps_pack(init_mps, init_mps.act_single_site_op(op, op_pos))
                else:
                    # Multi-site observable
                    obs_value = TN.inner_mps_pack(init_mps, init_mps.act_n_body_gate(op, op_pos))
                obs_values.append(obs_value)
            obs_evol.append(obs_values)
    return mps_evol, obs_evol

def gen_1d_pos_sequence(length: int, single_pos: List[int], repeat_interval: int = 1) -> List[List[int]]:
    """Generate a sequence of positions for 1D quantum gates.
    
    Args:
        length: Total length of the quantum system
        single_pos: Initial positions of gates
        repeat_interval: Interval between repeated gate applications
        
    Returns:
        List of position sequences, where each sequence is a list of positions
        for applying gates
    """
    tmp_pos = single_pos[:]
    pos_sequence = []
    while length > tmp_pos[-1]:
        pos_sequence.append(tmp_pos)
        tmp_pos = [i + repeat_interval for i in tmp_pos]
    # rev_seq = list(pos_sequence.__reversed__())
    # pos_sequence = pos_sequence[:] + rev_seq[:]
    return pos_sequence

def gen_1d_gate_dict(length: int, hamilt_list: List[tc.Tensor], tau: float, 
                     single_pos_list: List[List[int]], repeat_interval: int = 1, 
                     bc: str = 'open') -> Dict[str, List[Any]]:
    """Generate a dictionary of quantum gates and their positions for 1D systems.
    
    Args:
        length: Total length of the quantum system
        hamilt_list: List of Hamiltonian terms
        tau: Time step size
        single_pos_list: List of initial positions for each Hamiltonian term
        repeat_interval: Interval between repeated gate applications
        bc: Boundary conditions ('open' or 'periodic')
        
    Returns:
        Dictionary containing:
        - 'gate_i': List of quantum gates (time-evolved Hamiltonian terms)
        - 'pos': List of position sequences for each gate
    """
    return_pos = []
    gate_list = []
    for i, hamilt in enumerate(hamilt_list):
        shape = list(hamilt.shape)
        mid = len(shape)//2
        mat_shape = [multiply_list(shape[:mid]), multiply_list(shape[mid:])]
        gate_i = tc.matrix_exp(-1j * hamilt.reshape(mat_shape) * tau / 2).reshape(shape)
        gate_list.append(gate_i)
        return_pos.append(gen_1d_pos_sequence(length, single_pos_list[i], repeat_interval))
    return {'gate_i':gate_list, 'pos':return_pos}