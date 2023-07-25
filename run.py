# import sys
# sys.path.insert(0, 'D:\\Manual\\Code\\tests\\Graduation Project\\')


import os
os.chdir('GraduationProject')

import time
T1 = time.perf_counter()

for i in [5, 10, 15, 20, 25]:
    os.system('python TimeEvolution.py --pt_it {}'.format(int(i)))
    os.system('python ADQC.py --pt_it {}'.format(int(i)))
    # os.system('python draw.py --Jx {} --Jy {} --Jz {} --hx {}'.format(i, i, i, 0.5))
    pass

T2 = time.perf_counter()
print('程序运行时间:%.4f秒'%((T2 - T1)))