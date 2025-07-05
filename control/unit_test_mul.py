#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from collections import deque
import time
import re
from enum import Enum
import subprocess
import cv2

from control.visual_servoing import VisualServoingController, VisualState # 从你的包中导入视觉控制器

# from control.visualize import Visualize
# import RPi.GPIO as GPIO
CAMERA_NAME_HINT = "USB"

class MissionState(Enum):
    START = 0
    TAKING_OFF = 1
    GLOBAL_SEARCH = 2       # 飞到高处，执行一次性的全局搜索
    TARGETING_CYCLE = 3     # 进入针对每个目标的循环
    LANDING = 4
    MISSION_COMPLETE = 5


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # === 新增：初始化视觉部分 ===
        self.vision_controller = VisualServoingController(
            model_path='/home/weights/best.engine', # 你的模型路径
            target_class_name='circle'
        )
        device_path = self.find_video_device_by_name(CAMERA_NAME_HINT)
        self.cap = cv2.VideoCapture(device_path if device_path else 0)

        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头！")
            rclpy.shutdown()

        self.is_vision_ready = False

        # === 新增：任务流程管理变量 ===
        self.mission_state = MissionState.GLOBAL_SEARCH
        self.target_priority = ["Middle", "Left", "Right"]
        self.current_target_index = 0
        self.visited_targets_count = 0
        #=========================================================

        self.timer = self.create_timer(0.1, self.timer_callback)  # 每0.1秒调用一次

    def find_video_device_by_name(self,name_hint="USB Camera"):
    # (This function remains unchanged)
        try:
            result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError): return None
        lines = result.stdout.splitlines()
        matched_device_name = False
        for line in lines:
            if name_hint in line: matched_device_name = True
            elif matched_device_name and "/dev/video" in line:
                match = re.search(r"(/dev/video\d+)", line)
                if match: return match.group(1)
        return None

    def execute_visual_command(self, command):
        """根据视觉指令来控制无人机"""
        if command is None:
            return

        # 简单的比例控制，将指令转换为小的位置增量
        step_size_xy = 0.1  # 水平移动步长
        step_size_z = 0.0   # 这里我们只做水平调整
        
        delta_x, delta_y = 0.0, 0.0
        if "向右平移" in command: delta_y = step_size_xy  #??????
        if "向左平移" in command: delta_y = -step_size_xy
        if "向前平移" in command: delta_x = step_size_xy
        if "向后平移" in command: delta_x = -step_size_xy
        
        if delta_y > 0: self.get_logger().info("执行: 向右平移")
        elif delta_y < 0: self.get_logger().info("执行: 向左平移")

        if delta_x > 0: self.get_logger().info("执行: 向前平移")
        elif delta_x < 0: self.get_logger().info("执行: 向后平移")
    


    #定时器
    def timer_callback(self) -> None:
        """Callback function for the timer."""        
        # 更新日志计数器
        if not self.is_vision_ready:
            # 只有在第一次进入timer_callback时执行
            if self.vision_controller.load_model():
                self.is_vision_ready = True
                self.get_logger().info("视觉系统准备就绪，开始执行任务逻辑。")
            else:
                self.get_logger().error("视觉系统初始化失败，节点将不执行任务。")
                return # 如果模型加载失败，直接返回，不执行后续逻辑        
        
        # --- 视觉处理部分 ---
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("无法捕获图像")
            return
        
        # 调用视觉控制器处理图像
        visual_state, visual_command, annotated_frame = self.vision_controller.process_frame(frame)
        cv2.imshow("Drone View", annotated_frame)
        cv2.waitKey(1)

        if self.mission_state == MissionState.GLOBAL_SEARCH:
            #上升到global——search高度
            self.get_logger().info("飞行器上升")

            if self.vision_controller.initial_target_map:
                self.get_logger().info("全局搜索完成，进入目标打击循环。")
                self.mission_state = MissionState.TARGETING_CYCLE

        # GLOBAL_SEARCH执行一次之后，mission_state状态都为TARGETING_CYCLE
        elif self.mission_state == MissionState.TARGETING_CYCLE:
            if self.visited_targets_count >= len(self.target_priority):
                self.mission_state = MissionState.LANDING
                return
            
            current_target_name = self.target_priority[self.current_target_index]

            if self.vision_controller.visual_state not in [VisualState.CENTERING, VisualState.TARGET_LOCKED]:
                self.get_logger().info(f"设置新目标: [{current_target_name}]")
                self.vision_controller.set_target(current_target_name)

            if visual_state == VisualState.CENTERING:
                self.execute_visual_command(visual_command)

            elif visual_state == VisualState.TARGET_LOCKED:
                self.get_logger().info(f"目标 [{current_target_name}] 已锁定，准备下降。")
                self.get_logger().info("模拟飞行器已经下降完成")
                self.get_logger().info(f"模拟对 [{current_target_name}] 完成投放！")
                    # 3. 更新统一的、可靠的进度计数器
                self.visited_targets_count += 1
                self.current_target_index += 1
                self.get_logger().info(f"进度更新：已完成 {self.visited_targets_count} 个目标。")
                
                if self.visited_targets_count >= len(self.target_priority):
                    self.mission_state = MissionState.LANDING
                else:
                    # 5. 准备下一个目标
                    self.get_logger().info("正在模拟爬升回巡航高度...")
                    # ... 这里可以添加爬升的逻辑 ...
                    
                    # 直接设置下一个目标，让视觉控制器开始寻找
                    next_target_name = self.target_priority[self.current_target_index]
                    self.vision_controller.set_target(next_target_name)


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
