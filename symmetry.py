import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import linalg
import matplotlib.pyplot as plt

def get_state_index(state: int, index: int) -> int:
    """获得一个整数某位置处的二进制值"""
    mask = 1 << index
    return (state & mask) >> index


def flip_state(state: int, index: int) -> int:
    """翻转一个整数某位置处的二进制值"""
    mask = 1 << index
    return state ^ mask

def standard_Hal(N, Nz, J):
    state_list = []
    for a in range(2 ** N):
        if bin(a).count("1") == Nz:
            state_list.append(a)

    H = np.zeros((len(state_list),) * 2)

    for new_a, a in enumerate(state_list):
        for i in range(N-1):
            j = i + 1
            ai = get_state_index(a, i)
            aj = get_state_index(a, j)

            # Sz
            new_b = new_a
            if ai == aj:
                H[new_a, new_b] += J
            else:
                H[new_a, new_b] += -J

            # SxSx + SySy
            if ai != aj:
                b = flip_state(a, i)
                b = flip_state(b, j)
                new_b = state_list.index(b)
                H[new_a, new_b] += 1 / 2 * J
    return H

def Halm_sym(N, Nz, J, Jz, hz):
    state_list = []
    for a in range(2 ** N):
        if bin(a).count("1") == Nz:
            state_list.append(a)
    
    H = np.zeros((len(state_list),) * 2)
    print(H.shape)

    for new_a, a in enumerate(state_list):
        for i in range(N-1):
            j = i + 1
            ai = get_state_index(a, i)
            aj = get_state_index(a, j)

            # SzSz
            new_b = new_a
            if ai == aj:
                H[new_a, new_b] += Jz[i] / 4
            else:
                H[new_a, new_b] += -Jz[i] / 4

            # SxSx + SySy
            if ai != aj:
                b = flip_state(a, i)
                b = flip_state(b, j)
                new_b = state_list.index(b)
                H[new_a, new_b] += J / 2
        
        for i in range(N):
            ai = get_state_index(a, i)
            _ = (-1)**(ai) * hz[i]/2
            H[new_a, new_a] += _
    return H

def unfolding(spectrum):
    x = spectrum
    y = np.array([i for i in range(1, len(spectrum)+1)])
    pn = np.polyfit(x, y, deg=15)
    new_spectrum = np.polyval(pn, x)
    return new_spectrum

N = 14
J = 1
theta = 0.5
delta = 1
Jz = np.array([(delta+theta*(2*i-N)/(N-2)) for i in range(1, N)])
# Jz = np.ones((N))
print(Jz)
# hz = (np.random.rand((N)) - 0.5)
hz = np.zeros((N))
hl = 0
hz[0] += hl
print(hz)
energy = []
# 这里的总自旋写2是因为总的因子 1/2 被省略了
for Nz in range(7, 8):
    H112 = Halm_sym(N, Nz, J, Jz, hz)
    print(H112.shape)
    eigvals, eigvecs = linalg.eigh(H112)
    energy.append(eigvals)

energy = np.sort(np.concatenate(energy))
np.save('/data/home/scv7454/run/GraduationProject/Data/spectrum', energy)
print(energy.shape)
# print(energy.shape)

# from Library.PhysModule import hamiltonian_heisenberg
# import torch as tc
# hamilt = hamiltonian_heisenberg('half', 1, 1, 1, [0, 0], [0, 0], [0, 0], device=tc.device('cpu'), dtype=tc.complex64)
# energy = tc.linalg.eigvals(hamilt)
# print(energy)

energy_fold = unfolding(energy)
cut = np.ceil(len(energy)*0.1)
start = int(cut)
end = len(energy) - int(cut)
energy_fold = energy_fold[start:end]

plt.hist(energy, bins=50, density=True)
plt.xlabel('E')
plt.ylabel('rho(E)')
plt.savefig('/data/home/scv7454/run/GraduationProject/pics/rho_E.svg')
plt.close()

plt.plot(energy, [i for i in range(1, len(energy)+1)], 'g-', label='data')
plt.plot(energy[start:end], energy_fold, 'r-', label='fit')

plt.xlabel('E')
plt.ylabel('N(E)')
plt.legend()
plt.savefig('/data/home/scv7454/run/GraduationProject/pics/fold.svg')
plt.close()

spacing = np.diff(energy_fold)
# 归一化能级间隔
# normalized_spacings = spacing / mean_spacing
# normalized_spacings = normalized_spacings[normalized_spacings<3]

# 绘制归一化能级间隔的直方图
plt.hist(spacing, bins=50, density=True, label='normalized_spacings')

# 绘制泊松分布
s = np.linspace(0, 3, 100)
poisson_dist = np.exp(-s)
plt.plot(s, poisson_dist, 'r-', label='Poisson')

# 绘制GOE分布
goe_dist = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
plt.plot(s, goe_dist, 'g-', label='GOE')

plt.xlabel('s')
plt.ylabel('P(s)')
plt.legend()
plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/XXZ_inhomo/ProbDis(delta={delta}, theta={theta}).svg')
plt.close()

Sn = spacing[:-1]
Snn = spacing[1:]
r = np.minimum(Sn, Snn)/np.maximum(Sn, Snn)
r_mean = np.mean(r)
print(r_mean)
plt.hist(r, bins=50, density=True, alpha=0.7, color='blue')
plt.title(f'<r>={r_mean}')
plt.xlabel('Difference')
plt.ylabel('Probability Density')
plt.grid(True)
plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/XXZ_inhomo/ProbDis(r)(delta={delta}, theta={theta}).svg')
plt.show()