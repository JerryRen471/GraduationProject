import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
data = pd.read_csv('/data/home/scv7454/run/GraduationProject/Data/PXP.csv')

# 指定列作为xyz轴
param = ['time_interval', 'sample_num', 'train_set_type', 'loss', 'length']
given_param = [0.1, 1, 'entangled', 'multi_mags', 10]

# 筛选数据
for i in range(len(param)):
    data = data[data[param[i]] == given_param[i]]

# 提取需要的列
unique_columns = ['entangle_dim', 'evol_num']

data = data.drop_duplicates(subset=unique_columns, keep='last')

# 提取x、y和z轴
x, y = np.meshgrid(data[unique_columns[0]].unique(), data[unique_columns[1]].unique())
z = np.zeros_like(x, dtype=float)
for i, xi in enumerate(data[unique_columns[0]].unique()):
    for j, yj in enumerate(data[unique_columns[1]].unique()):
        try:
            z[j, i] = data[(data[unique_columns[0]] == xi) \
                            & (data[unique_columns[1]] == yj)]['spectrum_diff'].values[0]
        except Exception as e:
            print(unique_columns, '=')
            print(xi, yj)
            print(e)
            z[j, i] = -1

# 绘制等高线图
levels = np.linspace(z.min(), z.max(), 50)
contour = plt.contourf(x, y, z, levels=levels, cmap='viridis_r')
plt.xlabel(unique_columns[0])
plt.ylabel(unique_columns[1])
plt.title('Contour Plot of spectrum_diff')
plt.colorbar(contour)
plt.grid(True)

plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/PXP/SpectrumDiff_{unique_columns[0]}Vs{unique_columns[1]}.svg')
plt.close()

subset_1 = data[data[unique_columns[0]] == 1]
subset_2 = data[data[unique_columns[1]] == 1]
