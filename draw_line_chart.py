import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# 读取CSV文件
data = pd.read_csv('/data/home/scv7454/run/GraduationProject/Data/Heis_sorted.csv')
# data = data[(data['evol_num'] >= 5)]
# data['time_tot'] = data['time_interval'] * data['evol_num']
# data = data[(data['time_tot'] <= 2) & (data['time_tot'] >= 0.1)]
# data = data[(data['time_interval'] >= 1) & (data['time_interval'] <= 2)]

loss = 'multi_mags'
# print(loss)
# Example usage:
para_dict = {
    # 'sample_num': 10,
    'evol_num': 1,
    'entangle_dim': 1,
    # 'time_interval': 0.2,
    'train_set_type': 'Z2',
    'loss': loss
}

x_axis = 'sample_num'
y_axis_list = ['gate_fidelity', 'spectrum_diff', 'H_diff']
varing_para = 'time_interval'
varing_para_set = [0.02*i for i in range(1, 11)]
choose_n = 4
def choose_n_points(list_to_choose, n):
    l = len(list_to_choose)
    m = (l-n) // (n-1)
    choose = []
    for i in range(n):
        choose.append(list_to_choose[i*(m+1)])
    return choose
varing_para_set = choose_n_points(varing_para_set, choose_n)

evol_num_list = [i for i in range(1, 2)]
train_set_type_list = ['Z2']

dtypes = {
    'length': int,
    'delta': float,
    'J': float,
    'lambda': float,
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

# 绘制横坐标为总时长(time_interval * evol_num)的不同y轴的折线图，标明参数 *time_interval* 、evol_num、sample_num、train_set_type、loss的组合
# 对比相同总时长，不同time_interval的影响

for y_axis in y_axis_list:
    fig, ax = plt.subplots()
    for train_set_type in train_set_type_list:
        para_dict['train_set_type'] = train_set_type
    
        ax.cla()
        cmap = sns.color_palette('plasma', len(varing_para_set))
        print(len(varing_para_set))
        for i, varing_para_value in enumerate(varing_para_set):
            para_dict[varing_para] = varing_para_value

            pic_path = '/data/home/scv7454/run/GraduationProject/pics/PXP/LineChartOf{}/{}/'.format(x_axis.capitalize(), loss)
            os.makedirs(pic_path, exist_ok=True)
            # 提取符合条件的行
            filtered_data = data.copy()
            for key, value in para_dict.items():
                filtered_data = filtered_data[filtered_data[key] == value]
        
            # 绘制折线图
            ax.plot(filtered_data[x_axis], filtered_data[y_axis], label='{}={}'.format(varing_para, varing_para_value), color=cmap[i])
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            title = 'Line Chart - '
            for key, value in para_dict.items():
                title += f'{key}: {value}, '
            title = title.rstrip(', ')
            ax.set_title(title, wrap=True)
        
        ax.legend()
        fig.savefig(pic_path + '{y_axis}Vs{x_axis}_Varing{varing_para}_{data_type}.svg'.format(y_axis=y_axis.capitalize(), x_axis=x_axis.capitalize(), varing_para=varing_para.capitalize(), data_type=train_set_type))
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

