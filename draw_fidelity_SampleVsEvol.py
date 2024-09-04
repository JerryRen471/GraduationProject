import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载 CSV 数据
csv_file_path = '/data/home/scv7454/run/GraduationProject/Data/PXP.csv'
data = pd.read_csv(csv_file_path)
data.sort_values(['time_interval','sample_num','evol_num'], ascending=True, inplace=True)

# 获取所有唯一的 (train_set_type, loss) 组合
unique_combinations = data[['train_set_type', 'loss', 'length', 'time_interval']].drop_duplicates()

# 创建等高线图
for _, row in unique_combinations.iterrows():
    train_set_type, length, loss, time_interval = row['train_set_type'], row['length'], row['loss'], row['time_interval']
    # time_interval = 2
    subset = data[(data['train_set_type'] == train_set_type) & (data['length'] == length) \
            & (data['loss'] == loss) & (data['time_interval'] == time_interval) \
            & (data['evol_num'] <= 100)\
            ]

    if (len(subset['sample_num'].unique()) > 1) & (len(subset['evol_num'].unique()) > 1):
        # 创建网格
        X, Y = np.meshgrid(subset['sample_num'].unique(), subset['evol_num'].unique())
        Z = np.zeros_like(X, dtype=float)

        for i, sample_num in enumerate(subset['sample_num'].unique()):
            print(i)
            for j, evol_num in enumerate(subset['evol_num'].unique()):
                Z[j, i] = subset[(subset['sample_num'] == sample_num) \
                                & (subset['evol_num'] == evol_num) \
                                # & (subset['hl'] == 0)\
                                ]['spectrum_diff'].values[0]

        # 绘制等高线图
        plt.figure(figsize=(6, 5))
        contour = plt.contourf(X, Y, Z, cmap='viridis_r')
        plt.colorbar(contour)
        plt.title(f'Spectrum Diff Contour Plot\n(initial state={train_set_type})')
        plt.xlabel('Sample Num')
        plt.xticks([a for a in range(1,11)])
        plt.ylabel('Evol Num')
        plt.grid(True)
        plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/PXP/SpectrumDiffSampleVsEvol(time_interval={time_interval},init_state={train_set_type}).svg')
