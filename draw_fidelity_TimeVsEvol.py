import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载 CSV 数据
csv_file_path = '/data/home/scv7454/run/GraduationProject/Data/QuantumSun.csv'
data = pd.read_csv(csv_file_path)
data.sort_values(['time_interval','sample_num','evol_num'], ascending=True, inplace=True)

# 获取所有唯一的 (train_set_type, loss) 组合
unique_combinations = data[['train_set_type', 'loss', 'length', 'sample_num', 'alpha']].drop_duplicates()

# 创建等高线图
for _, row in unique_combinations.iterrows():
    train_set_type, length, loss, sample_num, alpha = row['train_set_type'], row['length'], row['loss'], row['sample_num'], row['alpha']
    # time_interval = 2
    subset = data[(data['train_set_type'] == train_set_type) & (data['length'] == length) \
            & (data['loss'] == loss) & (data['sample_num'] == sample_num) & (data['alpha'] == alpha) \
            & (data['evol_num'] <= 100)\
            & (data['alpha'] == 0.75)\
            & (data['time_interval'] >= 0.1)
            ]

    if (len(subset['time_interval'].unique()) > 1) & (len(subset['evol_num'].unique()) > 1):
        # 创建网格
        X, Y = np.meshgrid(subset['time_interval'].unique(), subset['evol_num'].unique())
        Z = np.zeros_like(X, dtype=float)

        for i, time_interval in enumerate(subset['time_interval'].unique()):
            print(i)
            for j, evol_num in enumerate(subset['evol_num'].unique()):
                Z[j, i] = subset[(subset['time_interval'] == time_interval) \
                                & (subset['evol_num'] == evol_num) \
                                # & (subset['hl'] == 0)\
                                ]['spectrum_diff'].values[0]

        # 绘制等高线图
        plt.figure(figsize=(6, 5))
        contour = plt.contourf(X, Y, Z, cmap='viridis_r')
        plt.colorbar(contour)
        plt.title(f'Spectrum Diff Contour Plot\n(alpha={alpha})')
        plt.xlabel('Time Interval')
        # plt.xticks([a for a in range(1,11)])
        plt.ylabel('Evol Num')
        plt.grid(True)
        plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/QuantumSun/length{int(length)}/alpha{(alpha)}/loss_{loss}/SpecctrumDiffTimeVsEvol(sample_num={int(sample_num)},alpha={alpha}).svg')
