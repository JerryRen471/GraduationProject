from TimeEvolution import *
import pylops
from pylops import LinearOperator

from Library import BasicFun as bf
from Library import PlotFun as pf
from Library import PhysModule as phy

class TE2Mat(LinearOperator):
    def __init__(self, para):
        para_def = dict()
        para_def['J'] = [0, 0, 1]
        para_def['h'] = [1.5, 0, 0]
        para_def['length'] = 10
        para_def['spin'] = 'half'
        para_def['BC'] = 'open'
        para_def['tau'] = 0.01
        para_def['time_it'] = 10
        para_def['time_tot'] = para_def['time_it'] * para_def['tau']
        para_def['print_time_it'] = 10
        para_def['print_dtime'] = para_def['tau'] * para_def['print_time_it']
        para_def['device'] = None
        para_def['dtype'] = tc.complex128
        if para is None:
            para = dict()
        para = dict(para_def, **para)
        para['d'] = phy.from_spin2phys_dim(para['spin'])
        # para['time_it'] = round(para['time_tot'] / para['tau'])
        # para['print_time_it'] = round(para['print_dtime'] / para['tau'])
        para['device'] = bf.choose_device(para['device'])
        self.para = para
        super().__init__(dtype=para['dtype'], shape=(2**para['length'], 2**para['length']))

    def _matvec(self, x):
        x_ = tc.from_numpy(x).to(self.para['device'])
        x_ = x_.reshape([2] * self.para['length'])
        y_ = time_evolution_Heisenberg_chain(self.para, x_, save=False).reshape(-1)
        y = y_.cpu().numpy()
        return y

    def _matmat(self, B):
        C = np.zeros(B.shape, dtype=np.complex128)
        for i in range(B.shape[1]):
            C[:, i] = self._matvec(B[:, i])
        return C

    def _rmatvec(self, x):
        return self._matmat(np.eye(2**self.length)).T * x

if __name__ == '__main__':
    para = dict()
    para['length'] = 10
    para['time_it'] = 10
    para['print_time_it'] = 10
    A = TE2Mat(para)
    A_ = A._matmat(np.eye(2**10, dtype=np.complex128))
    print(A_.shape)