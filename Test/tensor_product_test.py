import numpy as np
from scipy.linalg import block_diag

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

def group_conjugation(A, B):
    """
    Calculates the group conjugation of matrix A by matrix B: B^(-1)AB
    
    Args:
        A (numpy.ndarray): The matrix to be conjugated
        B (numpy.ndarray): The conjugating matrix
    
    Returns:
        numpy.ndarray: The result of B^(-1)AB
    """
    # Check if matrices have compatible dimensions
    if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1] or A.shape[0] != B.shape[0]:
        raise ValueError("Matrices must be square and have the same dimensions")
    
    # Calculate B^(-1)
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix B is not invertible")
    
    # Calculate B^(-1)AB
    return B_inv @ A @ B

def character_table_S3():
    """
    Returns the character table of the symmetric group S3.
    
    Returns:
        numpy.ndarray: The character table of S3
    """
    # S3 has 3 conjugacy classes: [e], [(12), (23), (13)], [(123), (132)]
    # And 3 irreducible representations: trivial, sign, and 2D
    return np.array([
        [1, 1, 1],  # Trivial representation
        [1, -1, 1],  # Sign representation
        [2, 0, -1]   # 2D representation
    ])

def find_similarity_transform(representation_matrices, multiplicities):
    """
    Finds a similarity transform that brings a representation to block diagonal form.
    
    Args:
        representation_matrices (list): List of matrices forming the representation
        multiplicities (numpy.ndarray): Multiplicities of irreducible representations
    
    Returns:
        numpy.ndarray: The similarity transform matrix P
    """
    # Get the dimension of the representation
    dim = representation_matrices[0].shape[0]
    
    # Construct the projection operators for each irrep
    # For S3, we have 3 irreps: trivial (1D), sign (1D), and 2D
    projection_operators = []
    
    # Trivial representation (1D)
    if multiplicities[0] > 0:
        # Projection operator for trivial irrep
        P_triv = np.zeros((dim, dim))
        for matrix in representation_matrices:
            P_triv += matrix
        P_triv /= 6  # |G| = 6 for S3
        projection_operators.append((P_triv, multiplicities[0]))
    
    # Sign representation (1D)
    if multiplicities[1] > 0:
        # Projection operator for sign irrep
        P_sign = np.zeros((dim, dim))
        # Identity and 3-cycles have character 1, 2-cycles have character -1
        for i, matrix in enumerate(representation_matrices):
            if i == 0 or i == 3:  # Identity or 3-cycle
                P_sign += matrix
            else:  # 2-cycles
                P_sign -= matrix
        P_sign /= 6
        projection_operators.append((P_sign, multiplicities[1]))
    
    # 2D representation
    if multiplicities[2] > 0:
        # Projection operator for 2D irrep
        P_2d = np.zeros((dim, dim))
        # Identity has character 2, 2-cycles have character 0, 3-cycles have character -1
        for i, matrix in enumerate(representation_matrices):
            if i == 0:  # Identity
                P_2d += 2 * matrix
            elif i == 3:  # 3-cycle
                P_2d -= matrix
        P_2d /= 6
        projection_operators.append((P_2d, multiplicities[2]))
    
    # Find eigenvectors of projection operators
    # These eigenvectors form a basis for the invariant subspaces
    basis_vectors = []
    
    for P, mult in projection_operators:
        # Find eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(P)
        
        # Sort eigenvalues and eigenvectors by magnitude
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Select eigenvectors with eigenvalue close to 1
        # These span the invariant subspace
        selected = 0
        for i, val in enumerate(eigvals):
            if np.abs(val - 1) < 1e-10 and selected < mult:
                # Ensure the vector is normalized
                vec = eigvecs[:, i]
                vec = vec / np.linalg.norm(vec)
                basis_vectors.append(vec)
                selected += 1
        
        # If we didn't find enough eigenvectors with eigenvalue 1,
        # take the eigenvectors with largest eigenvalues
        while selected < mult:
            for i, val in enumerate(eigvals):
                if np.abs(val - 1) >= 1e-10 and selected < mult:
                    vec = eigvecs[:, i]
                    vec = vec / np.linalg.norm(vec)
                    basis_vectors.append(vec)
                    selected += 1
    
    # Convert basis vectors to a matrix
    P = np.column_stack(basis_vectors)
    
    # Ensure P is invertible by adding linearly independent vectors if needed
    rank = np.linalg.matrix_rank(P)
    if rank < dim:
        # Add standard basis vectors that are linearly independent with current basis
        remaining = dim - rank
        standard_basis = np.eye(dim)
        for i in range(dim):
            if remaining <= 0:
                break
            # Check if this standard basis vector is linearly independent
            temp_P = np.column_stack([P, standard_basis[:, i]])
            if np.linalg.matrix_rank(temp_P) > rank:
                P = temp_P
                rank += 1
                remaining -= 1
    
    # Ensure P is invertible
    if np.linalg.matrix_rank(P) < dim:
        raise ValueError("Could not construct an invertible similarity transform")
    
    return P

def reduce_representation(representation_matrices):
    """
    Reduces a representation of a permutation group into its irreducible components.
    
    Args:
        representation_matrices (list): List of matrices forming the representation
    
    Returns:
        tuple: (block_diagonal_form, similarity_transform)
    """
    # Get the dimension of the representation
    dim = representation_matrices[0].shape[0]
    
    # Calculate the character of the representation
    character = np.array([np.trace(matrix) for matrix in representation_matrices])
    
    # Get the character table of S3
    char_table = character_table_S3()
    
    # Calculate the multiplicities of each irreducible representation
    # For S3, we have 3 irreps: trivial (1D), sign (1D), and 2D
    multiplicities = np.zeros(3)
    
    # For each irrep
    for i in range(3):
        # Calculate multiplicity using orthogonality of characters
        # For S3: |G| = 6, and we have 3 conjugacy classes with sizes 1, 3, 2
        class_sizes = np.array([1, 3, 2])
        
        # Map the 6 group elements to their 3 conjugacy classes
        # Identity -> class 0
        # (12), (23), (13) -> class 1
        # (123), (132) -> class 2
        class_indices = np.array([0, 1, 1, 1, 2, 2])
        
        # Calculate the character for each conjugacy class
        class_characters = np.zeros(3)
        for j in range(3):
            # Get the indices of elements in this conjugacy class
            class_elements = np.where(class_indices == j)[0]
            # Average the characters of elements in this class
            class_characters[j] = np.mean(character[class_elements])
        
        # Calculate multiplicity using orthogonality of characters
        multiplicities[i] = np.sum(class_characters * np.conj(char_table[i]) * class_sizes) / 6
    
    # Round to nearest integer (should be integers for finite groups)
    multiplicities = np.round(multiplicities).astype(int)
    
    # Construct the block diagonal form
    blocks = []
    for i, mult in enumerate(multiplicities):
        if mult > 0:
            if i == 0:  # Trivial representation
                blocks.extend([np.array([[1]]) for _ in range(mult)])
            elif i == 1:  # Sign representation
                blocks.extend([np.array([[1]]) for _ in range(mult)])
            else:  # 2D representation
                blocks.extend([np.array([[1, 0], [0, 1]]) for _ in range(mult)])
    
    block_diagonal = block_diag(*blocks)
    
    # Find the similarity transform
    P = find_similarity_transform(representation_matrices, multiplicities)
    
    return block_diagonal, multiplicities, P

def test_representation_reduction():
    """
    Test the reduction of a representation of S3.
    """
    # Create a representation of S3 using swap operators
    s_o1 = [(1, 2)]
    s_o2 = [(2, 3)]
    s_o3 = [(1, 2), (2, 3), (1, 2)]
    s_e1 = []
    s_e2 = [(1, 2), (2, 3)]
    s_e3 = [(2, 3), (1, 2)]
    
    # Create the representation matrices
    rep_matrices = [
        apply_swap_sequence(s_e1, 3, n_states=2),  # Identity
        apply_swap_sequence(s_o1, 3, n_states=2),  # (12)
        apply_swap_sequence(s_o2, 3, n_states=2),  # (23)
        apply_swap_sequence(s_o3, 3, n_states=2),  # (123)
        apply_swap_sequence(s_e2, 3, n_states=2),  # (12)(23)
        apply_swap_sequence(s_e3, 3, n_states=2)   # (23)(12)
    ]
    
    # Reduce the representation
    block_diagonal, multiplicities, P = reduce_representation(rep_matrices)
    
    print("Multiplicities of irreducible representations:")
    print(f"Trivial (1D): {multiplicities[0]}")
    print(f"Sign (1D): {multiplicities[1]}")
    print(f"2D: {multiplicities[2]}")
    
    print("\nBlock diagonal form:")
    print(block_diagonal)
    
    print("\nSimilarity transform P:")
    print(P)
    
    # Verify that P^(-1)AP is block diagonal for each matrix A
    P_inv = np.linalg.inv(P)
    for i, A in enumerate(rep_matrices):
        transformed = P_inv @ A @ P
        print(f"\nTransformed matrix {i+1}:")
        print(transformed)
    
    # Verify that the reduction is correct
    # The sum of squares of multiplicities times dimensions should equal the original dimension
    original_dim = rep_matrices[0].shape[0]
    reduced_dim = sum(multiplicities * np.array([1, 1, 2]))
    
    print(f"\nOriginal dimension: {original_dim}")
    print(f"Reduced dimension: {reduced_dim}")
    print(f"Dimensions match: {original_dim == reduced_dim}")

def main():
    # Test representation reduction
    test_representation_reduction()

if __name__ == "__main__":
    main()