import pylops
import torch as tc
import numpy as np

from pylops import LinearOperator

from Library import BasicFun as bf
from Library import PlotFun as pf
from Library import PhysModule as phy

# # 添加运行参数
# import argparse
# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--time', type=float, default=100)
# parser.add_argument('--pt_it', type=int, default=10)
# args = parser.parse_args()
# interval = args.pt_it

# qc = tc.load('GraduationProject/Data/qc_dt{:d}.pth'.format(interval))
# qc.single_state = True
# n = 2
# length = 10
# loc = 2 ** int(length-n) - 1
# state = tc.zeros((2 ** length, ), device='cuda:0', dtype=tc.complex128)
# state[loc] = 1.0
# state = state.reshape([2] * length)
# print(state.shape)
# print(qc(state).shape)

class QC2Mat(LinearOperator):
    r"""
    
    """
    def __init__(self, para):
        para_def = dict()
        para_def['length'] = 10
        para_def['device'] = None
        para_def['dtype'] = tc.complex128
        if para is None:
            para = dict()
        para = dict(para_def, **para)
        para['device'] = bf.choose_device(para['device'])
        self.para = para
        self.qc = tc.load('GraduationProject/Data/qc_dt{:d}_tot{:.0f}.pth'.format(para['print_time_it'], para['time_tot']))
        super().__init__(dtype=para['dtype'], shape=(2**para['length'], 2**para['length']))

    def _matvec(self, x):
        self.qc.single_state = True
        x_ = tc.from_numpy(x)
        x_ = x_.reshape([2] * self.para['length'])
        with tc.no_grad():
            y_ = self.qc(x_.to(self.para['device'])).reshape(-1)
        y = y_.cpu().numpy()
        # print(y)
        return y

    def _matmat(self, B):
        self.qc.single_state = False
        B = tc.from_numpy(B)
        shape_ = list(B.shape[0]) + [2] * self.para['length']
        B = B.reshape(shape_)
        with tc.no_grad():
            C = self.qc(B.to(self.para['device'])).reshape([B.shape[0], -1])
        # C = np.zeros(B.shape, dtype=np.complex128)
        # for i in range(B.shape[1]):
        #     C[:, i] = self._matvec(B[:, i])
        C = C.cpu().numpy()
        return C

    def _rmatvec(self, x):
        return self._matmat(np.eye(2**self.para['length'])).T * x

if __name__ == '__main__':
    para = dict()
    para['length'] = 10
    para['time_it'] = 10
    para['print_time_it'] = 10
    A = QC2Mat(para)
    A_ = A._matmat(np.eye(2**10, dtype=np.complex128))
    print(A_.shape)
    # eigvals = np.linalg.eigvals(A_)
    # print(eigvals)