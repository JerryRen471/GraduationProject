import pandas as pd

dtypes = {
    'model_name': str,
    'length': int,
    'J': float,
    'delta': float,
    'theta': float,
    'time_interval': float,
    'evol_num': int,
    'sample_num': int,
    'data_type': str,
    'loss_type': str,
    'gate_fidelity': float,
    'spectrum_diff': float,
    'train_loss': float,
    'test_loss': float,
    'train_fide': float,
    'test_fide': float
}

# Read the CSV file
df = pd.read_csv("/data/home/scv7454/run/GraduationProject/Data/Heis.csv", dtype=dtypes)
df.pop('depth')
# df_new = pd.read_csv("/data/home/scv7454/run/GraduationProject/Data/PXP_tmp.csv", dtype=dtypes)

subset=['J', 'delta', 'theta', 'loss_type', 'data_type', 'time_interval', 'evol_num', 'sample_num', 'length']
# Append the new rows to the end of the old dataframe
# df = pd.concat(df_old, ignore_index=True)
df = df.sort_values(by=subset)

# Write the updated dataframe back to the CSV file
df.to_csv("/data/home/scv7454/run/GraduationProject/Data/Heis_sorted.csv", index=False)

# import os
# os.remove("/data/home/scv7454/run/GraduationProject/Data/PXP_tmp.csv")