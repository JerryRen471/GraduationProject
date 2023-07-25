import pylops
import torch as tc
import numpy as np

from pylops import LinearOperator

qc = tc.load('GraduationProject/Data/qc.pth')
qc.single_state = True
n = 2
length = 10
loc = 2 ** int(length-n) - 1
state = tc.zeros((2 ** length, ), device='cuda:0', dtype=tc.complex128)
state[loc] = 1.0
state = state.reshape([2] * length)
print(state.shape)
print(qc(state).shape)

class ExpH(LinearOperator):
    r"""
    
    """
    def __init__(self, length, dtype=tc.complex128, device='cuda:0'):
        self.length = length
        self.d = 2**int(length)
        self.dtype = dtype
        self.device = device
        super().__init__(dtype=dtype, shape=(self.d, self.d))

    def _matvec(self, x):
        x_ = tc.from_numpy(x)
        x_ = x_.reshape([2] * self.length)
        with tc.no_grad():
            y_ = qc(x_.to(self.device)).reshape(-1)
        y = y_.cpu().numpy()
        print(y)
        return y

    def _matmat(self, B):
        C = np.zeros(B.shape, dtype=np.complex128)
        for i in range(B.shape[1]):
            C[:, i] = self._matvec(B[:, i])
        self.mat = C
        return C

    def _rmatvec(self, x):
        return self.C.T * x

if __name__ == '__main__':
    A = ExpH(length = 10)
    A_ = A._matmat(np.eye(2**10, dtype=np.complex128))
    print(A_.shape)
    eigvals = np.linalg.eigvals(A_)
    print(eigvals)