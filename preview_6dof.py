import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from stewart_controller import Stewart_Platform

csv_path = (
    r"D:\study\postgraduate\UAV\3D-VRO"
    r"\data\reastistic_ship\realistic_ship_6dof.csv"
)

data = pd.read_csv(csv_path)
platform = Stewart_Platform(
    66, 87.5, 50.8, 130, 0.2269, 0.82, 5 * np.pi / 6
)

angles = []

# 只检查开头 10 秒；先关闭平移，并把姿态缩小到原数据的 20%
for i in range(0, 100):
    rotation = data.loc[i, ["Roll(rad)", "Pitch(rad)", "Yaw(rad)"]].to_numpy(float)
    servo_angles = platform.calculate(
        np.zeros(3),
        rotation * 1.0,
    ) * 180 / np.pi
    angles.append(servo_angles)

angles = np.asarray(angles)
print("是否全部为有限数值：", np.isfinite(angles).all())
print("六路输出角度范围（度）：")
print("最小值：", np.round(angles.min(axis=0), 2))
print("最大值：", np.round(angles.max(axis=0), 2))