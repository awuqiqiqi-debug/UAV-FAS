"""
更新init_location.xlsx文件，设置新的初始坐标
"""
import pandas as pd
import numpy as np

file_path = './data/init_location.xlsx'

# UAV初始坐标: (0, 25, 50) m
uav_data = {
    'x': [0],
    'y': [25],
    'z': [50]
}
df_uav = pd.DataFrame(uav_data)

# 合法用户坐标: 用户1(4,47,0), 用户2(25,25,0)
user_data = {
    'x': [4, 25],
    'y': [47, 25],
    'z': [0, 0]
}
df_user = pd.DataFrame(user_data)

# 窃听者固定坐标: (47, -4, 0) m
attacker_data = {
    'x': [47],
    'y': [-4],
    'z': [0]
}
df_attacker = pd.DataFrame(attacker_data)

# RIS固定坐标: (0, 50, 12.5) m
ris_data = {
    'x': [0],
    'y': [50],
    'z': [12.5]
}
df_ris = pd.DataFrame(ris_data)

# RIS法线方向: [0, 1, 0] (指向y轴)
ris_norm_data = {
    'x': [0],
    'y': [1],
    'z': [0]
}
df_ris_norm = pd.DataFrame(ris_norm_data)

# 写入Excel文件
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    df_uav.to_excel(writer, sheet_name='UAV', index=False)
    df_user.to_excel(writer, sheet_name='user', index=False)
    df_attacker.to_excel(writer, sheet_name='attacker', index=False)
    df_ris.to_excel(writer, sheet_name='RIS', index=False)
    df_ris_norm.to_excel(writer, sheet_name='RIS_norm_vec', index=False)

print("init_location.xlsx 更新完成！")
print(f"UAV初始位置: {uav_data}")
print(f"用户位置: {user_data}")
print(f"窃听者位置: {attacker_data}")
print(f"RIS位置: {ris_data}")
