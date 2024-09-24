import pandas as pd

# Read the CSV file
df_old = pd.read_csv("/data/home/scv7454/run/GraduationProject/Data/PXP.csv")
df_new = pd.read_csv("/data/home/scv7454/run/GraduationProject/Data/PXP_tmp.csv")

# Append the new rows to the end of the old dataframe
df = df_old.append(df_new, ignore_index=True)
    
# Write the updated dataframe back to the CSV file
df.to_csv("/data/home/scv7454/run/GraduationProject/Data/PXP.csv", index=False)

import os
os.remove("/data/home/scv7454/run/GraduationProject/Data/PXP_tmp.csv")