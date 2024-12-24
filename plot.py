import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 指数函数定义
def exp_func(x, a, b):
    return - a * np.exp(b * x) + 1


data = pd.read_csv('/data/home/scv7454/run/GraduationProject/Data/PXP_scaling.csv')

para_dict = {
    # 'delta': 0.1,
    # 'J': 1,
    # 'lambda': 1,
    'sample_num': 1,
    'entangle_dim': 1,
    'time_interval': 0.2,
    # 'evol_num': 9,
    'train_set_type': 'product',
    'loss': 'multi_mags'
}

para_seq = ['length', 
            # 'delta', 'J', 'lambda', 
            'time_interval', 'evol_num', 'sample_num', 'entangle_dim', 'train_set_type', 'loss']

for param, value in para_dict.items():
    data = data[data[param] == value]

x_axis = 'evol_num'
y_axis = 'length'

train_set_type_list = ['product', 'non_product', 'Z2']
z_axis_list = ['H_diff', 'spectrum_diff', 'gate_fidelity']

dtypes = {
    'length': int,
    # 'delta': float,
    # 'J': float,
    # 'lambda': float,
    'time_interval': float,
    'evol_num': int,
    'sample_num': int,
    'train_set_type': str,
    'entangle_dim': int,
    'loss': str,
    'gate_fidelity': float,
    'spectrum_diff': float,
    'H_diff': float,
    'train_loss': float,
    'test_loss': float,
    'train_fidelity': float,
    'test_fidelity': float
}

length_list = data['length'].unique()
evol_num_list = data['evol_num'].unique()

group_by_length = data.groupby('length')

for i, (length, group) in enumerate(group_by_length):
    fig, ax = plt.subplots()
    ax.cla()

    x = group['evol_num']
    y = group['gate_fidelity']
    # 拟合数据
    popt, pcov = curve_fit(exp_func, x, y, maxfev=10000)
    
    # 计算拟合精度
    perr = np.sqrt(np.diag(pcov))

    # 打印拟合参数和精度
    print(f"Length: {length}, Fitted parameters: a={popt[0]}, b={popt[1]}, Uncertainties: da={perr[0]}, db={perr[1]}")

    # 绘制散点图和拟合曲线
    ax.scatter(x, y, label='Data', color='red')
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = exp_func(x_fit, *popt)
    ax.plot(x_fit, y_fit, label='Fitted Curve', color='blue')
    ax.set_xlabel('evol_num')
    ax.set_ylabel('gate_fidelity')
    ax.set_title(f'Length: {length}\nLength: {length}, Fitted parameters: a={popt[0]}, b={popt[1]}, Uncertainties: da={perr[0]}, db={perr[1]}', wrap=True)
    ax.legend()
    fig.savefig(f'fit_GateFidelity_Vs_EvolNum_length{length}.svg')
    print(f"Length: {length}, Fitted parameters: a={popt[0]}, b={popt[1]}")