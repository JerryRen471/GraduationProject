import sys
sys.path.insert(0, 'D:\\Manual\\Code\\tests\\Graduation Project\\')


import os
os.chdir('D:\\Manual\\Code\\tests\\Graduation Project')

import time
T1 = time.perf_counter()

for i in range(0, 3, 1):
    os.system('python TimeEvolution_NN_NNN.py --t1 {} --t2 {} --V1 {} --V2 {}'.format(1, i*0.1, 1, i*0.1))
    os.system('python LSTM.py')
    os.system('python draw.py --t1 {} --t2 {} --V1 {} --V2 {}'.format(1, i*0.1, 1, i*0.1))
    pass

T2 = time.perf_counter()
print('程序运行时间:%.4f秒'%((T2 - T1)))