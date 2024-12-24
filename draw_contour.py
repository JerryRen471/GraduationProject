import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.colors import LogNorm

def draw_contour(pic_path, save_fix, data, para_dict, para_seq, x_axis, y_axis, z_axis, levels, cmap):
    pic_path += ''.join([f"({key}={value})" for key, value in para_dict.items()]) + '/'

    x_axis_values = data[x_axis].unique()
    y_axis_values = data[y_axis].unique()

    # 筛选数据
    for param, value in para_dict.items():
        data = data[data[param] == value]

    # 提取需要的列
    unique_columns = [y_axis, x_axis]
    data = data.drop_duplicates(subset=unique_columns, keep='last')

    # 提取x、y和z轴
    x, y = np.meshgrid(x_axis_values, y_axis_values)
    z = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x_axis_values):
        for j, yj in enumerate(y_axis_values):
            try:
                z[j, i] = data[(data[x_axis] == xi) & (data[y_axis] == yj)][z_axis].values[0]
            except Exception as e:
                print(x_axis, y_axis, '=', xi, yj)
                para_dict_copy = para_dict.copy()
                para_dict_copy[x_axis] = xi
                para_dict_copy[y_axis] = yj
                with open(save_fix, 'a') as file:
                    # file.write(f'{xi}, {yj}\n')
                    file.write(' '.join(str(para_dict_copy[key]) for key in para_seq))
                    file.write('\n')
                print(e)
                z[j, i] = -1

    # 绘制等高线图
    if isinstance(levels, np.ndarray) == False:
        levels = np.linspace(z.min(), z.max(), 50)
    try:
        contour = plt.contourf(x, y, z, levels=levels, cmap=cmap, norm=LogNorm())
    except Exception as e:
        contour = plt.contourf(x, y, z, cmap=cmap, norm=LogNorm())
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title('Contour Plot of {}\n'.format(z_axis) + str(para_dict), wrap=True)
    plt.colorbar(contour)
    plt.grid(True)

    os.makedirs(pic_path, exist_ok=True)
    plt.savefig(pic_path + '{z}_{x}Vs{y}.svg'.format(z=z_axis, x=x_axis, y=y_axis))
    plt.close()

    # 根据 time_interval 分组数据
    grouped_data = data.groupby(y_axis)

    # 设置颜色映射
    cmap = sns.color_palette('plasma', len(grouped_data))

    # 绘制折线图
    plt.figure()
    legend = []
    for i, (y, group) in enumerate(grouped_data):
        plt.plot(group[x_axis], group[z_axis], label="{}={}".format(y_axis, y), color=cmap[i])
    plt.xlabel(x_axis)
    plt.ylabel(z_axis)
    plt.title('{} vs {} for different {}\n'.format(z_axis, x_axis, y_axis) + str(para_dict), wrap=True)
    plt.legend()
    os.makedirs(pic_path, exist_ok=True)
    plt.savefig(pic_path + '{}_{}.svg'.format(z_axis, x_axis))
    plt.close()

    # 根据 evol_num 分组数据
    grouped_data = data.groupby(x_axis)

    # 设置颜色映射
    cmap = sns.color_palette('plasma', len(grouped_data))

    # 绘制折线图
    plt.figure()
    legend = []
    for i, (x, group) in enumerate(grouped_data):
        plt.plot(group[y_axis], group[z_axis], label="{}={}".format(x_axis, x), color=cmap[i])

    plt.xlabel(y_axis)
    plt.ylabel(z_axis)
    plt.title('{} vs {} for different {}\n'.format(z_axis, y_axis, x_axis) + str(para_dict), wrap=True)
    plt.legend()
    os.makedirs(pic_path, exist_ok=True)
    plt.savefig(pic_path + '{}_{}.svg'.format(z_axis, y_axis))
    plt.close()

loss = 'multi_mags'
print(loss)
# Example usage:
para_dict = {
    # 'delta': 0.1,
    # 'J': 1,
    # 'lambda': 1,
    # 'sample_num': 1,
    'length': int(10),
    'entangle_dim': int(1),
    'time_interval': 0.12,
    # 'evol_num': 9,
    'train_set_type': 'product',
    'loss': loss
}
para_seq = ['length', 
            # 'delta', 'J', 'lambda', 
            'time_interval', 'evol_num', 'sample_num', 'entangle_dim', 'train_set_type', 'loss']


x_axis = 'sample_num'
y_axis = 'evol_num'

train_set_type_list = ['Z2']
z_axis_list = ['H_diff', 'spectrum_diff', 'gate_fidelity']

pic_path = '/data/home/scv7454/run/GraduationProject/pics/PXP/'
save_fix = '/data/home/scv7454/run/GraduationProject/fix_PXP.txt'

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

# 读取CSV文件
data = pd.read_csv('/data/home/scv7454/run/GraduationProject/Data/PXP_right_spectrum.csv').astype(dtypes)
# data.to_csv('/data/home/scv7454/run/GraduationProject/Data/PXP_tmp.csv', index=False)
data = data[(data['evol_num'] <= 10)]

# for train_set_type in train_set_type_list:
#     para_dict['train_set_type'] = train_set_type
#     for z_axis in z_axis_list:
#         draw_contour(pic_path, data, para_dict, x_axis, y_axis, z_axis, levels, cmap)
with open(save_fix, 'w') as file:
    file.write('')

for train_set_type in train_set_type_list:
    para_dict['train_set_type'] = train_set_type
    for z_axis in z_axis_list:
        print(train_set_type, z_axis)
        if z_axis == 'spectrum_diff':
            levels = None
            cmap = 'viridis_r'
        elif z_axis == 'gate_fidelity':
            levels = np.linspace(0.0, 1.0, 51)
            cmap = 'viridis'
        elif z_axis == 'H_diff':
            levels = None
            cmap = 'viridis_r'
        with open(save_fix, 'a') as file:
            file.write(f'{train_set_type} {z_axis}\n')
        draw_contour(pic_path, save_fix, data, para_dict, para_seq, x_axis, y_axis, z_axis, levels, cmap)