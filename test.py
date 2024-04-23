import torch as tc
from torch.distributions import uniform

def haar_random_state(dimension):
    """
    Generate a Haar random state for a quantum system of given dimension.
    
    Parameters:
        dimension (int): The dimension of the quantum system.
        
    Returns:
        torch.Tensor: A Haar random state represented as a complex torch tensor.
    """
    # Generate a random unitary matrix from the Haar measure
    re = tc.randn(dimension, dimension)
    im = tc.randn(dimension, dimension)
    q, _ = tc.linalg.qr(re + 1j * im, 'complete')

    # Create a random vector in the appropriate space
    random_vector = tc.zeros(dimension) + 1j * tc.zeros(dimension)
    random_vector[0] = 1+0j

    # Apply the random unitary to the random vector to get the Haar random state
    haar_random_state = tc.matmul(q, random_vector)

    return haar_random_state

def haar_random_product_states(number, length):
    """
    Generate Haar random product states for multiple quantum systems.

    Parameters:
        number (int): Number of Haar random product states to generate.
        length (int): Length of the qubit chain.

    Returns:
        torch.Tensor: A tensor containing multiple Haar random product states represented as complex torch tensors.
    """
    # Generate Haar random product states
    product_states = []
    for _ in range(number):
        # Generate Haar random states for each qubit in the chain
        random_states = [haar_random_state(2) for _ in range(length)]
        # Compute the tensor product of all the random states
        product_state = random_states[0]
        for state in random_states[1:]:
            product_state = tc.kron(product_state, state)
        product_states.append(product_state)

    return tc.stack(product_states)

# Example usage:
number = 5  # Number of Haar random product states to generate
length = 3  # Length of the qubit chain
random_product_states = haar_random_product_states(number, length)
print("Haar random product states:")
print(random_product_states.shape)
print(tc.einsum("ij,ij->i", random_product_states, random_product_states.conj()))