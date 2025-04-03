from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# 读取CSV文件
data = pd.read_csv('/data/home/scv7454/run/GraduationProject/Data/PXP_scaling_multi_sample.csv')
# data = data[(data['evol_num'] >= 5)]
# data['time_tot'] = data['time_interval'] * data['evol_num']
# data = data[(data['time_tot'] <= 2) & (data['time_tot'] >= 0.1)]
# data = data[(data['time_interval'] >= 1) & (data['time_interval'] <= 2)]

loss = 'multi_mags'
# print(loss)
# Example usage:
para_dict = {
    'sample_num': 1,
    'evol_num': 1,
    'entangle_dim': 1,
    'time_interval': 0.2,
    'train_set_type': 'Z2',
    'loss': loss
    # 'length': 10
}

x_axis = 'length'
y_axis_list = ['gate_fidelity', 'spectrum_diff', 'H_diff']
evol_num_list = [i for i in range(1, 11)]
train_set_type_list = ['product', 'non_product', 'Z2']

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
        slope_list = []
    
        ax.cla()
        cmap = sns.color_palette('plasma', len(evol_num_list))
        for i, evol_num in enumerate(evol_num_list):
            para_dict['evol_num'] = evol_num

            pic_path = '/data/home/scv7454/run/GraduationProject/pics/PXP/LineChartOfLength/{}/'.format(loss)
            os.makedirs(pic_path, exist_ok=True)
            # 提取符合条件的行
            filtered_data = data.copy()
            for key, value in para_dict.items():
                filtered_data = filtered_data[filtered_data[key] == value]
        
            y = filtered_data.groupby('length')[y_axis].agg(['mean', 'var']).reset_index()
            
            # 计算误差线
            y_err = filtered_data.groupby('length')[y_axis].std().reset_index()

            # 绘制折线图
            ax.fill_between(filtered_data[x_axis].drop_duplicates(), y['mean']-y_err[y_axis], y['mean']+y_err[y_axis], alpha=0.2, color=cmap[i])
            ax.plot(filtered_data[x_axis].drop_duplicates(), y['mean'], label='evol_num={}'.format(evol_num), color=cmap[i])
            
            # 线性拟合
            coefficients = np.polyfit(y['length'], y['mean'], 1)
            slope_list.append(coefficients[0])
            linear_fit = np.poly1d(coefficients)
            r_squared = 1 - (np.sum((y['mean'] - linear_fit(y['length']))**2) / np.sum((y['mean'] - np.mean(y['mean']))**2))
            print(f'Fitting parameters for evol_num={evol_num}: slope={coefficients[0]}, intercept={coefficients[1]}, R²={r_squared}')
            # 绘制拟合曲线
            ax.plot(y['length'], linear_fit(y['length']), linestyle='--', color=cmap[i], label='Fit: R²={:.2f}, slope={:.2f}'.format(r_squared, coefficients[0]))

            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            title = 'Line Chart - '
            for key, value in para_dict.items():
                title += f'{key}: {value}, '
            title = title.rstrip(', ')
            ax.set_title(title, wrap=True)
            
            ax.legend()
            fig.savefig(pic_path + 'Shaded_{}_Length_sample{}_{}.svg'.format(y_axis, para_dict['sample_num'], train_set_type))
            # plt.close(fig)
            
        print(slope_list)
        fig_slope, ax_slope = plt.subplots()
        # 对evol_num_list取对数
        log_evol_num_list = np.log(evol_num_list)
        
        # 线性拟合
        coefficients_log = np.polyfit(log_evol_num_list, slope_list, 1)
        linear_fit_log = np.poly1d(coefficients_log)
        
        # 计算R²
        r_squared_log = 1 - (np.sum((slope_list - linear_fit_log(log_evol_num_list))**2) / np.sum((slope_list - np.mean(slope_list))**2))
        
        # 绘制拟合曲线
        ax_slope.plot(log_evol_num_list, linear_fit_log(log_evol_num_list), linestyle='--', color='red', 
              label='Fit: R²={:.2f}, slope={:.2f}'.format(r_squared_log, coefficients_log[0]))
        
        # 绘制原本的曲线
        ax_slope.plot(log_evol_num_list, slope_list, marker='o', label='Original Slope', color='blue')
        
        ax_slope.set_xlabel('log(evol_num)')
        ax_slope.set_ylabel('Slope')
        ax_slope.legend()
        fig_slope.savefig(pic_path + 'Slope_{}_Length_sample{}_{}.svg'.format(y_axis, para_dict['sample_num'], train_set_type))

# # 绘制折线图
# plt.plot(filtered_data[x_axis], filtered_data[y_axis])
# plt.xlabel(x_axis)
# plt.ylabel(y_axis)
# plt.title('Line Chart')
# plt.show()

