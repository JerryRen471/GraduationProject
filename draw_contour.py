import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def draw_contour(pic_path, data, para_dict, x_axis, y_axis, z_axis, levels, cmap):
    # PXP
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
                print(x_axis, y_axis, '=')
                print(xi, yj)
                print(e)
                z[j, i] = -1

    # 绘制等高线图
    if isinstance(levels, np.ndarray) == False:
        levels = np.linspace(z.min(), z.max(), 50)
    contour = plt.contourf(x, y, z, levels=levels, cmap=cmap)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title('Contour Plot of {}\n'.format(z_axis) + str(para_dict))
    plt.colorbar(contour)
    plt.grid(True)

    os.makedirs(pic_path, exist_ok=True)
    plt.savefig(pic_path + '{z}_{x}Vs{y}.svg'.format(z=z_axis, x=x_axis, y=y_axis))
    plt.close()

    # 根据 time_interval 分组数据
    grouped_data = data.groupby(y_axis)

    # 设置颜色映射
    cmap = sns.color_palette(cmap, len(grouped_data))

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for i, (y, group) in enumerate(grouped_data):
        plt.plot(group[x_axis], group[z_axis], label="{}={}".format(y_axis, y), color=cmap[i])

    plt.xlabel(x_axis)
    plt.ylabel(z_axis)
    plt.title('{} vs {} for different {}\n'.format(z_axis, x_axis, y_axis) + str(para_dict))
    plt.legend()
    os.makedirs(pic_path, exist_ok=True)
    plt.savefig(pic_path + '{}_{}.svg'.format(z_axis, x_axis))

    # 根据 evol_num 分组数据
    grouped_data = data.groupby(x_axis)

    # 设置颜色映射
    cmap = sns.color_palette(cmap, len(grouped_data))

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for i, (x, group) in enumerate(grouped_data):
        plt.plot(group[y_axis], group[z_axis], label="{}={}".format(x_axis, x), color=cmap[i])

    plt.xlabel(y_axis)
    plt.ylabel(z_axis)
    plt.title('{} vs {} for different {}\n'.format(z_axis, y_axis, x_axis) + str(para_dict))
    plt.legend()
    os.makedirs(pic_path, exist_ok=True)
    plt.savefig(pic_path + '{}_{}.svg'.format(z_axis, y_axis))
    plt.show()

# Example usage:
para_dict = {
    'sample_num': 1,
    'entangle_dim': 1,
    'train_set_type': 'Z2',
    'loss': 'fidelity',
    'length': 10
}
x_axis = 'evol_num'
y_axis = 'time_interval'
z_axis = 'gate_fidelity'
# levels = np.linspace(0.0, 1.0, 51)
levels = None
cmap = 'viridis'

pic_path = '/data/home/scv7454/run/GraduationProject/pics/PXP/'

# 读取CSV文件
data = pd.read_csv('/data/home/scv7454/run/GraduationProject/Data/PXP_tmp.csv')

train_set_type_list = ['Z2', 'product', 'non_product']
z_axis_list = ['gate_fidelity', 'spectrum_diff', 'H_diff']

for train_set_type in train_set_type_list:
    para_dict['train_set_type'] = train_set_type
    for z_axis in z_axis_list:
        draw_contour(pic_path, data, para_dict, x_axis, y_axis, z_axis, levels, cmap)
