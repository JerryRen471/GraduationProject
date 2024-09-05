import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载 CSV 数据
csv_file_path = '/data/home/scv7454/run/GraduationProject/Data/PXP.csv'
data = pd.read_csv(csv_file_path)
data.sort_values(['time_interval','sample_num','evol_num'], ascending=True, inplace=True)

time_seg

unique_combinations = data[['train_set_type', 'loss', 'length']].drop_duplicates()
Z2_para = unique_combinations[(unique_combinations['train_set_type'] == 'Z2')]
for _, row in unique_combinations.iterrows():
    train_set_type, length, loss = row['train_set_type'], row['length'], row['loss']
    print(train_set_type, length, loss)

    subset = data[(data['train_set_type'] == train_set_type) & (data['length'] == length) \
                & (data['loss'] == loss) \
                & (data['time_interval'] <= 0.3)\
                # & (data['evol_num'] <= 10)\
                    ]

    if (len(subset['time_interval'].unique()) > 1) & (len(subset['evol_num'].unique()) > 1):
        # 创建网格
        X, Y = np.meshgrid(subset['time_interval'].unique(), subset['evol_num'].unique())
        Z = np.zeros_like(X, dtype=float)

        for i, time_interval in enumerate(subset['time_interval'].unique()):
            print(i)
            for j, evol_num in enumerate(subset['evol_num'].unique()):
                try:
                    Z[j, i] = subset[(subset['time_interval'] == time_interval) \
                                & (subset['evol_num'] == evol_num)]['spectrum_diff'].values[0]
                except Exception as e:
                    print(e)
                    Z[j, i] = -1

        # 绘制等高线图
        plt.figure(figsize=(6, 5))
        contour = plt.contourf(X, Y, Z, cmap='viridis_r')
        plt.colorbar(contour)
        plt.title(f'Spectrum Difference Contour Plot\n(initial state={train_set_type})')
        plt.xlabel('Time interval')
        # plt.xticks([a for a in range(1,11)])
        plt.ylabel('Evol number')
        plt.grid(True)
        plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/PXP/(t=0-0.3)SpectrumDiffTimeVsEvol(init_state={train_set_type}).svg')


# 获取所有唯一的 (train_set_type, loss) 组合
unique_combinations = data[['train_set_type', 'loss', 'length']].drop_duplicates()

# 创建等高线图
for _, row in unique_combinations.iterrows():
    train_set_type, length, loss = row['train_set_type'], row['length'], row['loss']
    # time_interval = 2
    subset = data[(data['train_set_type'] == train_set_type) & (data['length'] == length) \
            & (data['loss'] == loss) \
            & (data['time_interval'] >= 0.3)\
            # & (data['evol_num'] <= 10)\
                ]

    if (len(subset['time_interval'].unique()) > 1) & (len(subset['evol_num'].unique()) > 1):
        # 创建网格
        X, Y = np.meshgrid(subset['time_interval'].unique(), subset['evol_num'].unique())
        Z = np.zeros_like(X, dtype=float)

        for i, time_interval in enumerate(subset['time_interval'].unique()):
            print(i)
            for j, evol_num in enumerate(subset['evol_num'].unique()):
                try:
                    Z[j, i] = subset[(subset['time_interval'] == time_interval) \
                                & (subset['evol_num'] == evol_num)]['spectrum_diff'].values[0]
                except Exception as e:
                    print(e)
                    Z[j, i] = -1

        # 绘制等高线图
        plt.figure(figsize=(6, 5))
        contour = plt.contourf(X, Y, Z, cmap='viridis_r')
        plt.colorbar(contour)
        plt.title(f'Spectrum Difference Contour Plot\n(initial state={train_set_type})')
        plt.xlabel('Time interval')
        # plt.xticks([a for a in range(1,11)])
        plt.ylabel('Evol number')
        plt.grid(True)
        plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/PXP/SpectrumDiffTimeVsEvol(init_state={train_set_type}).svg')
