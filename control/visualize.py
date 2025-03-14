import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

class Visualize:
    def __init__(self, save_dir="plots"):
        self.positions = []  # 存储三维坐标
        self.save_dir = save_dir  # 设置保存目录
        os.makedirs(self.save_dir, exist_ok=True)  # 自动创建目录

        # # 注册退出时保存图像的函数
        # atexit.register(self.plot_and_save)
        # signal.signal(signal.SIGINT, self.handle_exit)  # 监听 Ctrl+C 事件

    def save_position(self, position):
        """ 添加位置点 """
        self.positions.append(position)

    def plot_and_save(self):
        """ 画出三维轨迹图并保存 """
        if not self.positions:
            print("No positions to plot.")
            return
        
        x, y, z = zip(*self.positions)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, marker='o', linestyle='-')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Position Visualization")

        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        ax.set_zlim(min(z), max(z))

        # **让x、y、z轴刻度间隔为1**
        ax.set_xticks(np.arange(int(min(x)), int(max(x)) + 1, step=1))
        ax.set_yticks(np.arange(int(min(y)), int(max(y)) + 1, step=1))
        ax.set_zticks(np.arange(int(min(z)), int(max(z)) + 1, step=1))

        # **调整三维比例，防止拉伸**
        ax.set_box_aspect([1, 1, 1])  # 让 x, y, z 轴的比例一致
        ax.set_aspect('auto')  # 防止坐标轴形状失真

        plot_count = len([f for f in os.listdir(self.save_dir) if f.endswith(".png")])
        save_path = os.path.join(self.save_dir, f"plot_{plot_count + 1}.png")

        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved plot to {save_path}")

    # def handle_exit(self, signum, frame):
    #     """ 处理 Ctrl+C 退出 """
    #     print("\nDetected Ctrl+C! Saving plot before exiting...")
    #     self.plot_and_save()
    #     exit(0)  # 退出程序