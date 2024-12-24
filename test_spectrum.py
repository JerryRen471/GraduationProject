# from draw_rand_PXP import spectrum
from cv2 import norm
import torch as tc
import scipy as sp

def normalize_pi(n):
    return n - (tc.div(n + tc.pi, 2*tc.pi, rounding_mode='trunc')) * 2*tc.pi

def spectrum(mat:tc.Tensor):
    energy = tc.log(tc.linalg.eigvals(mat))/1.j
    energy = energy.real
    energy = normalize_pi(energy)
    energy, ind = tc.sort(energy)
    return energy

a = tc.rand((4, 4), dtype=tc.complex64)
a = a.mm(a.conj().t())
b = sp.linalg.expm(a*1j)
b = tc.from_numpy(b)
print(b.mm(b.conj().t()))

eigval, eigvec = tc.linalg.eig(b)
eigval = eigval.type(tc.complex64)

print(tc.linalg.eigvals(b) - eigval)
print(tc.abs(eigval))
print(eigval)
output = spectrum(b)
expected = tc.linalg.eigvals(a).real

print(output, '\n', normalize_pi(expected))
print(output - normalize_pi(expected))
# print(output - expected.real)

n = 3*tc.pi
print(normalize_pi(n))
# normalize_pi = lambda n: n - ((n + tc.pi) // (2*tc.pi)) * 2*tc.pi
# print(n - ((n + tc.pi) // (2*tc.pi)) * 2*tc.pi - normalize_pi(n))
# print(n + 4*tc.pi)
# print(b - eigvec@tc.diag(eigval)@eigvec.conj().t())

# c = eigvec @ tc.diag(tc.log(eigval)/1.j) @ eigvec.conj().t()
# print(a - c)
