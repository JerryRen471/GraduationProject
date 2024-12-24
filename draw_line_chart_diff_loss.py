import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import seaborn as sns

# mpl.rc('text', usetex=True)

# 读取CSV文件
data = pd.read_csv('/data/home/scv7454/run/GraduationProject/Data/PXP_20241206.csv')
# data = data[(data['evol_num'] >= 5)]
# data['time_tot'] = data['time_interval'] * data['evol_num']
# data = data[(data['time_tot'] <= 2) & (data['time_tot'] >= 0.1)]
# data = data[(data['time_interval'] >= 1) & (data['time_interval'] <= 2)]

# loss = 'multi_mags'
# print(loss)
# Example usage:
para_dict = {
    'length': 10,
    # 'J': 1,
    # 'delta': 2,
    # 'theta': 0,
    # 'sample_num': 1,
    'evol_num': 1,
    'time_interval': 0.2,
    'data_type': 'product',
    # 'loss': loss
}

x_axis = 'sample_num'
# x_axis = 'evol_num'
# x_axis = 'time_interval'
y_axis_list = ['gate_fidelity', 'spectrum_diff', 'similarity', 'H_diff']
varing_para = 'loss_type'
varing_para_set = ['multi_mags', 'fidelity']
choose_n = 2
pic_path = '/data/home/scv7454/run/GraduationProject/pics/PXP/LineChartOf{}/'.format(x_axis.capitalize())

def choose_n_points(list_to_choose, n):
    l = len(list_to_choose)
    m = (l-n) // (n-1)
    choose = []
    for i in range(n):
        choose.append(list_to_choose[i*(m+1)])
    return choose
varing_para_set = choose_n_points(varing_para_set, choose_n)

# evol_num_list = [i for i in range(1, 2)]
train_set_type_list = ['product']

dtypes = {
    'length': int,
    # 'delta': float,
    # 'J': float,
    # 'theta': float,
    'time_interval': float,
    'evol_num': int,
    'sample_num': int,
    'data_type': str,
    'loss_type': str,
    'gate_fidelity': float,
    'spectrum_diff': float,
    'train_loss': float,
    'test_loss': float,
    'train_fide': float,
    'test_fide': float
}

for y_axis in y_axis_list:
    fig, ax = plt.subplots()
    ax.cla()
    for train_set_type in train_set_type_list:
        para_dict['data_type'] = train_set_type
    
        cmap = sns.color_palette('plasma', len(varing_para_set))
        print(len(varing_para_set))
        linestyle = '-' if train_set_type == 'product' else '--'
        for i, varing_para_value in enumerate(varing_para_set):
            para_dict[varing_para] = varing_para_value

            os.makedirs(pic_path, exist_ok=True)
            # 提取符合条件的行
            filtered_data = data.copy()
            for key, value in para_dict.items():
                filtered_data = filtered_data[filtered_data[key] == value]
            y = filtered_data.groupby(x_axis)[y_axis].agg(['mean', 'var']).reset_index()
            
            # 计算误差线
            y_err = filtered_data.groupby(x_axis)[y_axis].std().reset_index()
        
            # 绘制折线图
            ax.plot(filtered_data[x_axis].drop_duplicates(), y['mean'], label='{}={}, {}'.format(varing_para, varing_para_value, train_set_type), color=cmap[i], linestyle=linestyle)
    
    ax.set_xlabel(r'$N_s$')
    ax.set_ylabel(y_axis)
    title = ''
    para_dict.pop('data_type')
    para_dict.pop(varing_para)
    for key, value in para_dict.items():
        title += f'{key}: {value}, '
    title = title.rstrip(', ')
    ax.set_title(title, wrap=True)
        
    ax.legend()
    fig.savefig(pic_path + '{y_axis}Vs{x_axis}_Varing{varing_para}.svg'.format(y_axis=y_axis.capitalize(), x_axis=x_axis.capitalize(), varing_para=varing_para.capitalize()))
        # plt.close(fig)

# # 提取符合条件的行
# filtered_data = data[(data['entangle_dim'] == para_dict['entangle_dim']) &
#                      (data['time_interval'] == para_dict['time_interval']) &
#                      (data['train_set_type'] == para_dict['train_set_type']) &
#                      (data['loss'] == para_dict['loss']) &
#                      (data['length'] == para_dict['length'])]

# # 绘制折线图
# plt.plot(filtered_data[x_axis], filtered_data[y_axis])
# plt.xlabel(x_axis)
# plt.ylabel(y_axis)
# plt.title('Line Chart')
# plt.show()

