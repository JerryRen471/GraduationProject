import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载 CSV 数据
csv_file_path = '/data/home/scv7454/run/GraduationProject/Data/PXP.csv'
data = pd.read_csv(csv_file_path)
data.sort_values(['time_interval','sample_num','evol_num'], ascending=True, inplace=True)

# 获取所有唯一的 (train_set_type, loss) 组合
unique_combinations = data[['train_set_type', 'loss', 'length']].drop_duplicates()

# 创建等高线图
for _, row in unique_combinations.iterrows():
    train_set_type, length = row['train_set_type'], row['length']
    subset = data[(data['train_set_type'] == train_set_type) & (data['length'] == length)]

    # 创建网格
    X, Y = np.meshgrid(subset['time_interval'].unique(), subset['evol_num'].unique())
    Z = np.zeros_like(X)

    for i, time_interval in enumerate(subset['time_interval'].unique()):
        print(i)
        for j, evol_num in enumerate(subset['evol_num'].unique()):
            Z[j, i] = subset[(subset['time_interval'] == time_interval) & (subset['evol_num'] == evol_num)]['gate_fidelity'].values[0]

    # 绘制等高线图
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, Z, cmap='viridis', levels=np.linspace(0, 1, 101))
    plt.colorbar(contour)
    plt.title(f'Gate Fidelity Contour Plot\n(initial state={train_set_type})')
    plt.xlabel('Time Interval')
    plt.ylabel('Evol Num')
    plt.grid(True)
    plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/PXP/GateFidelityContourPlot(init_state={train_set_type}).svg')
