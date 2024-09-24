import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.colors as colors

# 加载 CSV 数据
csv_file_path = '/data/home/scv7454/run/GraduationProject/Data/PXP.csv'
data = pd.read_csv(csv_file_path)
data.sort_values(['time_interval','sample_num','evol_num'], ascending=True, inplace=True)

subset = data[(data['time_interval']==0.1) & (data['train_set_type']=='entangled')]
X, Y = np.meshgrid(subset['entangle_dim'].unique(), subset['evol_num'].unique())
Z = np.zeros_like(X, dtype=float)

for i, entangle_dim in enumerate(subset['entangle_dim'].unique()):
    for j, evol_num in enumerate(subset['evol_num'].unique()):
        try:
            Z[j, i] = subset[(subset['entangle_dim'] == entangle_dim) \
                            & (subset['evol_num'] == evol_num)]['spectrum_diff'].values[0]
        except Exception as e:
            print(entangle_dim, evol_num)
            print(e)
            Z[j, i] = 0

fig_size = (6, 5)
# 绘制等高线图
plt.figure(figsize=fig_size)
levels = np.linspace(Z.min(), Z.max(), 50)

# 创建第一个子图用于等高线图
contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis_r')
plt.colorbar(contour)
plt.title(f'Spectrum Difference Contour\n(initial state=entangled)')
plt.xlabel('Entangle Dim')
plt.ylabel('Evol Number')
plt.xticks(subset['entangle_dim'].unique())

plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/PXP/SpctDiffEntangleVsEvol(init=entangled).svg')
plt.close()

set_evol_num = subset[subset['evol_num']==4]
plt.plot(set_evol_num['entangle_dim'], set_evol_num['spectrum_diff'])
plt.xlabel('Entangle Dim')
plt.ylabel('Spectrum Difference')
plt.title(f'Spectrum Difference vs Entangle Dim\n(initial state=entangled)')
plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/PXP/SpctDiffVsEntangle(evol_num=4).svg')

def plot_spectrum_difference(data, time_seg_1, time_seg_2, fig_size, specified_time_interval=None, specified_evol_num=None):
    unique_combinations = data[['train_set_type', 'loss', 'length']].drop_duplicates()
    
    for _, row in unique_combinations.iterrows():
        train_set_type, length, loss = row['train_set_type'], row['length'], row['loss']
        print(train_set_type, length, loss)

        subset = data[(data['train_set_type'] == train_set_type) & (data['length'] == length) \
                    & (data['loss'] == loss) \
                    & (data['time_interval'] >= time_seg_1) & (data['time_interval'] <= time_seg_2)]

        if (len(subset['time_interval'].unique()) > 1) & (len(subset['evol_num'].unique()) > 1) & (train_set_type=='Z2'):
            # 创建网格
            X, Y = np.meshgrid(subset['time_interval'].unique(), subset['evol_num'].unique())
            Z = np.zeros_like(X, dtype=float)

            for i, time_interval in enumerate(subset['time_interval'].unique()):
                print(time_interval)
                for j, evol_num in enumerate(subset['evol_num'].unique()):
                    try:
                        Z[j, i] = subset[(subset['time_interval'] == time_interval) \
                                        & (subset['evol_num'] == evol_num)]['spectrum_diff'].values[0]
                    except Exception as e:
                        print(e)
                        Z[j, i] = -1

            # 绘制等高线图
            plt.figure(figsize=fig_size)
            levels = np.linspace(Z.min(), Z.max(), 50)

            # 创建第一个子图用于等高线图
            ax1 = plt.subplot(311)
            contour = ax1.contourf(X, Y, Z, levels=levels, cmap='viridis_r')
            plt.colorbar(contour, ax=ax1)
            ax1.set_title(f'Spectrum Difference Contour\n(initial state={train_set_type})')
            ax1.set_xlabel('Time Interval')
            ax1.set_ylabel('Evol Number')

            plt.grid(True)

            # 创建第二个子图用于指定 time_interval 的折线图
            ax2 = plt.subplot(312)
            if specified_time_interval is not None:
                line_subset = data[data['time_interval'] == specified_time_interval]
                ax2.plot(line_subset['evol_num'], line_subset['spectrum_diff'], label=f'Time Interval: {specified_time_interval}', color='red')
                ax2.set_title(f'Spectrum Difference for Time Interval: {specified_time_interval}')
            ax2.set_ylabel('Spectrum Difference')
            ax2.set_xlabel('Evol Number')
            ax2.legend()

            # 创建第三个子图用于指定 evol_num 的折线图
            ax3 = plt.subplot(313)
            if specified_evol_num is not None:
                evol_subset = data[data['evol_num'] == specified_evol_num]
                ax3.plot(evol_subset['time_interval'], evol_subset['spectrum_diff'], label=f'Evol Num: {specified_evol_num}', color='blue')
                ax3.set_title(f'Spectrum Difference for Evol Num: {specified_evol_num}')
            ax3.set_ylabel('Spectrum Difference')
            ax3.set_xlabel('Time Interval')
            ax3.legend()

            plt.tight_layout()
            plt.savefig(f'/data/home/scv7454/run/GraduationProject/pics/PXP/(t={time_seg_1}-{time_seg_2})SpctDiffTimeVsEvol(init={train_set_type}).svg')
            plt.show()

# 调用函数
# plot_spectrum_difference(data, time_seg_1=0, time_seg_2=0.3, fig_size=(15,15), specified_time_interval=0.01, specified_evol_num=1)
# plot_spectrum_difference(data, time_seg_1=time_seg_2, time_seg_2=time_seg_3, fig_size=fig_size)

