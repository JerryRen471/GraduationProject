import numpy as np

def get_swap_gate():
    """Returns the 2x2 Swap gate matrix."""
    return np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])

def get_identity_gate():
    """Returns the 2x2 Identity gate matrix."""
    return np.array([[1, 0],
                    [0, 1]])

def get_n_state_swap_gate(n_states):
    """
    Returns the swap gate matrix for n-state sites.
    For n states, this creates an n^2 × n^2 matrix.
    
    Args:
        n_states (int): Number of states per site (e.g., 3 for 0/1/2)
    
    Returns:
        numpy.ndarray: The swap gate matrix
    """
    dim = n_states ** 2
    swap = np.zeros((dim, dim))
    
    # For each possible combination of states
    for i in range(n_states):
        for j in range(n_states):
            # Original state index
            orig_idx = i * n_states + j
            # Swapped state index
            swap_idx = j * n_states + i
            # Set the swap operation
            swap[orig_idx, swap_idx] = 1
    
    return swap

def get_swap_operator(i, j, total_sites, n_states=2):
    """
    Creates a swap operator that swaps sites i and j in a system of total_sites.
    Only works for adjacent sites (|i-j| = 1).
    Positions are counted from left to right.
    
    Args:
        i (int): First site index (1-based, from left)
        j (int): Second site index (1-based, from left)
        total_sites (int): Total number of sites in the system
        n_states (int): Number of states per site (default: 2 for 0/1)
    
    Returns:
        numpy.ndarray: The swap operator matrix
    """
    if abs(i - j) != 1:
        raise ValueError("Swap operation only allowed between adjacent sites")
    
    # Convert to 0-based indexing and reverse the position counting
    i = total_sites - i  # Convert from left-to-right to right-to-left
    j = total_sites - j  # Convert from left-to-right to right-to-left
    
    # Create identity matrix for the entire system
    dim = n_states ** total_sites
    operator = np.eye(dim)
    
    # For each basis state
    for state in range(dim):
        # Get the states for sites i and j
        i_state = (state // (n_states ** i)) % n_states
        j_state = (state // (n_states ** j)) % n_states
        
        # If states are different, swap them
        if i_state != j_state:
            # Calculate the new state after swapping
            # First clear both positions
            new_state = state - (i_state * (n_states ** i)) - (j_state * (n_states ** j))
            # Then set them to their swapped values
            new_state += (j_state * (n_states ** i)) + (i_state * (n_states ** j))
            
            # Swap the amplitudes
            operator[state, state] = 0
            operator[state, new_state] = 1
            operator[new_state, new_state] = 0
            operator[new_state, state] = 1
    
    return operator

def apply_swap_sequence(sequence, total_sites, n_states=2):
    """
    Applies a sequence of adjacent swap operations.
    
    Args:
        sequence (list): List of tuples, each tuple contains two adjacent site indices
        total_sites (int): Total number of sites in the system
        n_states (int): Number of states per site (default: 2 for 0/1)
    
    Returns:
        numpy.ndarray: The combined operator matrix
    """
    # Start with identity matrix
    dim = n_states ** total_sites
    result = np.eye(dim)
    
    # Apply each swap operation in sequence
    for i, j in sequence:
        swap_op = get_swap_operator(i, j, total_sites, n_states)
        result = swap_op @ result
    
    return result

def test_apply_swap_sequence():
    # Example usage
    # Define a sequence of adjacent swaps: (1,2)(2,3)(3,4)
    sequence = [(1,2), (2,3), (3,4)]
    total_sites = 4
    
    # Calculate the combined effect
    result = apply_swap_sequence(sequence, total_sites)
    
    print("Sequence of swaps:", sequence)
    print("\nCombined operator matrix:")
    print(result)
    
    # Verify that the matrix is unitary
    is_unitary = np.allclose(result @ result.T, np.eye(2**total_sites))
    print("\nIs the operator unitary?", is_unitary)

def main():
    # Test 2-state system
    print("Testing 2-state system:")
    s_o1 = [(1, 2)]
    s_o2 = [(2, 3)]
    s_o3 = [(1, 2), (2, 3), (1, 2)]
    s_e2 = [(1, 2), (2, 3)]
    s_e3 = [(2, 3), (1, 2)]
    
    s_o1_op = apply_swap_sequence(s_o1, 3, 3)
    s_o2_op = apply_swap_sequence(s_o2, 3, 3)
    s_o3_op = apply_swap_sequence(s_o3, 3, 3)
    s_e2_op = apply_swap_sequence(s_e2, 3, 3)
    s_e3_op = apply_swap_sequence(s_e3, 3, 3)
    
    print(s_o1_op + s_o2_op + s_o3_op - s_e2_op - s_e3_op)
    
    # Test 3-state system
    print("\nTesting 3-state system:")
    swap_3state = get_n_state_swap_gate(3)
    print("3-state swap gate:")
    print(swap_3state)
    
    # Test swap operation in 3-state system
    swap_op_3state = get_swap_operator(1, 2, 2, n_states=3)
    print("\n3-state swap operator for 2 sites:")
    print(swap_op_3state)

if __name__ == "__main__":
    main()