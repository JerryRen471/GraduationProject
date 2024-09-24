import torch as tc

from rand_PXP import rand_entangled_states

def test_rand_entangled_states():
    number = 3
    length = 4
    entangle_dim = 2
    states = rand_entangled_states(number, length, entangle_dim, device=tc.device('cpu'))
    states = states.reshape(number, 2**2, 2**2)
    u, s, v = tc.linalg.svd(states)
    print(s)

    # assert states.shape == (number, 2, 2, 2, 2)
    # print(tc.sum(states * states.conj(), dim=1))
    # assert tc.allclose(tc.sum(states * states.conj(), dim=1), tc.ones((number, 1, 2, 2, 2), dtype=tc.complex64))

test_rand_entangled_states()