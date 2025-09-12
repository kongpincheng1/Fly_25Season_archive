import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("TkAgg")   # 改成 Tkinter 后端，而不是 Qt
import matplotlib.pyplot as plt
# import plotly.io as pio
# pio.renderers.default = "browser"


# ==============================================================================
# 🚀 STEP 0: 在这里配置你的物理参数和模拟场景
# ==============================================================================

# --- 1. 无人机当前的姿态和位置 (模拟值) ---
# 你可以任意修改这些值，看看结果如何变化！
DRONE_POSITION_NED = np.array([0.0, 0.0, -0.8])  # (x, y, z) in meters. Z是负数表示在空中
DRONE_ROLL_DEG = 0.0      # 滚转角 (度)
DRONE_PITCH_DEG = -1.0     # 俯仰角 (度) - 模拟无人机为了向前飞而前倾
DRONE_YAW_DEG = 0.0      # 偏航角 (度)

# --- 2. 相机相对于无人机机体的安装参数 (你的测量值) ---
# 平移 (x:前, y:右, z:下)
CAM_POS_IN_BODY = np.array([0, 0, 0])
# 旋转 (相机向下倾斜20度)
# --- 3. 投放器相对于无人机机体的位置 (你的测量值) ---
DROPPER_POS_IN_BODY = np.array([0.0, 0.0, 0]) # (x, y, z)

# --- 4. 视觉系统的观测结果 (模拟值) ---
# 假设视觉算法检测到目标在相机的这个坐标位置
TARGET_POS_IN_CAM = np.array([0.1, -0.05, 0.8]) # (X, Y, Z) in camera frame

# ==============================================================================
# 🤖 STEP 1: 构建基础变换矩阵
# ==============================================================================
print("="*50)
print("🤖 STEP 1: 构建基础变换矩阵")
print("="*50)

# --- 1.1 无人机机体 -> 相机 (T_body_cam) ---
# 旋转部分: 相机安装的俯仰角
R_body_cam = np.array([
    [ 0.,  1.,  0.],
    [-1.,  0.,  0.],
    [ 0.,  0.,  1.]
])
# 构建4x4齐次变换矩阵
T_body_cam = np.eye(4)
T_body_cam[:3, :3] = R_body_cam
T_body_cam[:3, 3] = CAM_POS_IN_BODY
print("✔️ 从 无人机机体(Body) -> 相机(Camera) 的变换矩阵 T_body_cam:\n", np.round(T_body_cam, 3))

# --- 1.2 世界(NED) -> 无人机机体 (T_world_body) ---
# 旋转部分: 无人机当前的姿态
R_world_body = R.from_euler('xyz', [DRONE_ROLL_DEG, DRONE_PITCH_DEG, DRONE_YAW_DEG], degrees=True).as_matrix()
# 构建4x4齐次变换矩阵
T_world_body = np.eye(4)
T_world_body[:3, :3] = R_world_body
T_world_body[:3, 3] = DRONE_POSITION_NED
print("\n✔️ 从 世界(World/NED) -> 无人机机体(Body) 的变换矩阵 T_world_body:\n", np.round(T_world_body, 3))


# ==============================================================================
# 🧠 STEP 2: 执行坐标变换之旅
# ==============================================================================
print("\n" + "="*50)
print("🧠 STEP 2: 坐标变换之旅：从相机观测到世界坐标")
print("="*50)

# --- 2.1 将目标位置从 相机坐标系 -> 机体坐标系 ---
# 将目标点转换为4x1齐次坐标
P_cam_h = np.append(TARGET_POS_IN_CAM, 1)
print(f"\n--- 阶段 A: 相机 -> 机体 ---")
print(f"观测到的目标点 (在相机坐标系中 P_cam):\n {np.round(P_cam_h[:3], 3)}")

# 应用变换: P_body = T_body_cam * P_cam
P_target_in_body_h = T_body_cam @ P_cam_h
print(f"应用 T_body_cam 矩阵后...")
print(f"计算出的目标点 (在机体坐标系中 P_body):\n {np.round(P_target_in_body_h[:3], 3)}")


# --- 2.2 将目标位置从 机体坐标系 -> 世界坐标系 ---
print(f"\n--- 阶段 B: 机体 -> 世界 ---")
print(f"当前目标点 (在机体坐标系中 P_body):\n {np.round(P_target_in_body_h[:3], 3)},\n{P_target_in_body_h}")

# 应用变换: P_world = T_world_body * P_body
P_target_in_world_h = T_world_body @ P_target_in_body_h
print(f"应用 T_world_body 矩阵后...")
print(f"计算出的目标点 (在世界坐标系中 P_world):\n {np.round(P_target_in_world_h[:3], 3)}")


# ==============================================================================
# 🎯 STEP 3: 计算无人机最终的飞行目标点
# ==============================================================================
print("\n" + "="*50)
print("🎯 STEP 3: 计算无人机最终的飞行目标点")
print("="*50)

# --- 3.1 计算投放器在世界坐标系中的位置 ---
# 投放器在机体中的位置（齐次坐标）
p_dropper_in_body_h = np.append(DROPPER_POS_IN_BODY, 1)
p_dropper_in_world_h = T_world_body @ p_dropper_in_body_h
print(f"投放器在世界坐标系中的位置 P_dropper_world:\n {np.round(p_dropper_in_world_h[:3], 3)}")

# --- 3.2 计算无人机中心和投放器之间的世界坐标偏移量 ---
# 这个偏移量是一个随姿态变化的向量
offset_dropper_in_world = p_dropper_in_world_h[:3] - DRONE_POSITION_NED
print(f"\n由于无人机倾斜, 投放器产生的世界坐标偏移量 Offset_world:\n {np.round(offset_dropper_in_world, 3)}")
print("(注意: 这个值不再是你测量的 [0, 0, 0.15] 了!)")

# --- 3.3 计算无人机应该飞到的目标点 ---
# Drone_Goal = Target_in_World - Offset_in_World
drone_goal_pos_ned = P_target_in_world_h[:3] - offset_dropper_in_world
print("\n------------------ 最终结果 ------------------")
print(f"为了让投放器对准目标, 无人机中心需要飞到这个世界坐标:")
print(f"🎯 Drone_Goal_Pos_NED = {np.round(drone_goal_pos_ned, 3)}")
print("----------------------------------------------")

