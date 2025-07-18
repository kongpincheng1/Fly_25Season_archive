#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry
from geometry_msgs.msg import Point
from collections import deque
import time
from control.DronePositionChecker import DronePositionChecker
from control.AlignmentChecker import AlignmentChecker
from control.ServoControl import ServoControl
from control.visual_servoing import VisualServoingController, VisualState # 从你的包中导入视觉控制器
import cv2
from enum import Enum
import subprocess
import re
import os
import csv
import argparse # <<< 新增
import sys      # <<< 新增


class MissionState(Enum):
    IDLE = 0
    STARTING_MISSION = 1
    IN_MISSION = 2
    PREPARING_OFFBOARD = 2.5
    SWITCHING_TO_OFFBOARD1 = 3
    IN_OFFBOARD = 3.5

    GLOBAL_SEARCH = 4
    TARGETING_CYCLE = 5
    RECONFIRMING_TARGETS = 6 # <<< 新增的状态
    PROACTIVE_SEARCH = 7  
    TIMEOUT = 8
    DROP_COMPLETE = 9

    SWITCHING_TO_MISSION = 10
    MISSION_RESUMED = 11
    DONE = 12

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self,args) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.target_position_subscriber = self.create_subscription(Point, '/target_position',
                                                                   self.target_position_callback, 10)

        base_photo_path = args.photo_path
        base_video_path = args.video_path
        
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_photo_path = os.path.join(base_photo_path, f"run_{run_timestamp}")
        self.get_logger().info(f"This run's photos will be saved to: {unique_photo_path}")
        unique_video_filename = f"mission_{run_timestamp}.avi" # AVI格式与MJPG编码器配合良好        
        
        # === 初始化视觉部分 (带视频录制功能) ===
        self.vision_controller = VisualServoingController(
            model_path=args.model_path,
            # 拍照功能
            enable_photo_capture=False,
            photo_save_path=unique_photo_path, 
            photo_capture_interval=30,
            # <<< 新增：启用并配置视频录制 >>>
            enable_video_recording=True,           # 设置为 True 来开启录制
            video_save_path=base_video_path,       # 视频保存的目录
            video_filename=unique_video_filename,  # 带有时间戳的唯一文件名
            video_fps=30.0                         # 视频帧率 (与你的timer频率匹配)
        )
        
        device_path = self.find_video_device_by_name(args.camera_hint)
        self.cap = cv2.VideoCapture(device_path if device_path else 0)        
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头！")
            rclpy.shutdown()
        
        self.is_vision_ready = False

        # === 新增：任务流程管理变量 ===
        self.Drop_mission_state = MissionState.GLOBAL_SEARCH
        self.state = MissionState.IDLE
        self.target_priority = ["Middle", "Left", "Right"]
        self.current_target_index = 0
        self.visited_targets_count = 0
        #=========================================================

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        

        self.target_position = None
        self.last_found_x_NED = None
        self.last_found_y_NED = None
        self.last_found_z_NED = None


        # <<< 修改：从命令行参数初始化任务参数 >>>
        self.first_alignment_height = args.first_alignment_height
        self.align_maxstep = args.align_maxstep
        self.afterAlign_descentHeight = args.descent_height
        self.global_search_height = args.search_height
        self.proactive_search_distance = args.proactive_search_dist
        
        # <<< 新增：从命令行参数获取超时和延迟设置 >>>
        self.drop_phase_timeout = args.drop_phase_timeout
        self.search_timeout = args.search_timeout
        self.second_align_maxtime = args.second_align_maxtime

        self.depthcam_xoffset = args.depthcam_xoffset
        self.depthcam_yoffset = args.depthcam_yoffset

        self.trigger_distance = args.trigger_distance
        self.position_threshold = args.position_threshold

        self.global_search_target_z = None

        self.initial_position = None
        self.initial_z = None  # 初始高度
        self.initial_x = None  #
        self.initial_y = None
        self.init_yaw = None

        self.DropArea_x = None
        self.DropArea_y = None

        # <<< 新增：用于计时超时的状态变量 >>>
        self.drop_phase_start_time = None
        self.second_align_start_timestamp = None
        self.search1_phase_start_time = None
        self.search2_phase_start_time = None
        self.timeout_drop_start_time = None

        self.first_alingment_tartget_height = None
        self.is_ReadyToTakeoff = False
        self.is_AtTakeoffHeight = False
        self.is_AtDropArea = False
        self.is_FinishDrop = False
        self.Is_Finish_1st_Drop = False
        self.Is_Finish_2nd_Drop = False

        self.Is_Descending_to_depth_camera_height = False

        self.droping_x = None
        self.droping_y = None
        self.droping_z = None

        # 新增日志计数器，用于减少日志输出频率
        self.log_counter = 0
        self.timeout_drop_count = 0
        self.timeout_drop_delay = 1.0

        self.first_alignment_complete = False
        self.second_alignment_complete = False

        

        # 为主动搜索阶段设置的状态变量
        self.proactive_target_x = None
        self.proactive_target_y = None
        self.is_proactive_target_set = False # <<< 新增：用于确保目标点只计算一次

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.03, self.timer_callback)
        
        #初始化位置判断器
        self.initPositionChecker = DronePositionChecker(
            logger_func=self.get_logger().info,
            tolerance=0.17, 
            duration=5.0
        )

          # 初始化 AlignmentChecker
        # <<< 修改：使用命令行参数来初始化 AlignmentChecker >>>
        self.first_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,
            threshold=args.first_align_threshold,
            time_window=args.first_align_time_window,
            check_frequency=args.first_align_check_freq
        )
        self.second_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,
            threshold=args.second_align_threshold,
            time_window=args.second_align_time_window,
            check_frequency=args.second_align_check_freq
        )
        # 初始化舵机控制器
        self.servo_control = ServoControl()
        
        # ========== PID控制参数设置区域 ==========
        # 📌 饱和P控制参数（大误差阶段）
        
        # 📌 细调阶段PID参数（小误差阶段）
        self.epsilon = self.align_maxstep  # 切换阈值 (0.2m) - 可调参数
        self.Kp_fine = 0.9911  # P增益 - 可调参数 (建议范围: 1.0-2.5)
        self.Ki = 0.1021       # I增益 - 可调参数 (建议范围: 0.1-0.8)
        self.Kd = 0.0009
        
        # 📌 PID状态变量
        self.integral_x = 0.0      # X方向积分项
        self.integral_y = 0.0      # Y方向积分项
        self.last_error_x = 0.0    # 上次X误差 (用于微分计算)
        self.last_error_y = 0.0    # 上次Y误差 (用于微分计算)
        self.dt = 0.03             # 控制周期 (秒) - 与timer频率一致
        
        # 📌 积分限幅参数
        self.max_integral = self.epsilon  # 积分限幅值 - 可调参数
        # =========================================

        # ========== 目标像素坐标日志自动生成带时间戳的文件 ==========
        # 生成带时间戳的日志目录和文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = '/Users/jihaobi/cqufly/fly3/mylog'  # 日志目录
        log_filename = f'bucket_pixel_log_{timestamp}.csv'  # 带时间戳的文件名
        os.makedirs(log_dir, exist_ok=True)
        self.pixel_log_path = os.path.join(log_dir, log_filename)
        
        # 创建CSV文件并写入表头
        with open(self.pixel_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'target_x', 'target_y', 'bucket_type', 'alignment_stage'])
        
        self.get_logger().info(f"日志文件已创建: {self.pixel_log_path}")
        # =========================================


    def target_position_callback(self, msg: Point):
        """Callback function for receiving target position."""
        self.target_position = msg  

    def fly_to_position(self, x, y, z):
        """Fly to the specified position."""
        self.publish_position_setpoint(x, y, z)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = self.init_yaw  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

        # <<< 新增：重写 destroy_node 方法以进行清理 >>>
    def destroy_node(self):
        """在节点关闭前，执行必要的清理工作。"""
        self.get_logger().info("节点正在关闭，执行清理程序...")
        # 清理视觉控制器（保存视频）
        if self.vision_controller:
            self.vision_controller.cleanup()
        # 清理摄像头
        if self.cap and self.cap.isOpened():
            self.cap.release()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        # 调用父类的方法完成ROS节点的销毁
        super().destroy_node()
        self.get_logger().info("清理完成，节点已关闭。")
    
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
        step_size_xy = 0.3  # 水平移动步长
        step_size_z = 0.0   # 这里我们只做水平调整

        current_x, current_y = self.coordinate_NED2FRD(self.vehicle_local_position.x, self.vehicle_local_position.y)
        
        delta_x, delta_y = 0.0, 0.0
        if "向右平移" in command: delta_y = step_size_xy  
        if "向左平移" in command: delta_y = -step_size_xy
        if "向前平移" in command: delta_x = step_size_xy
        if "向后平移" in command: delta_x = -step_size_xy

        # 计算新的FRD目标点
        target_x_frd = current_x + delta_x
        target_y_frd = current_y + delta_y
        
        # 转换回NED并发布
        target_x_ned, target_y_ned = self.coordinate_FRD2NED(target_x_frd, target_y_frd)
        self.publish_position_setpoint(target_x_ned, target_y_ned, self.global_search_target_z)    
    
    def drop_payload(self,servo_1,servo_2):
        self.servo_control.open_servo(servo_1,servo_2)

        self.get_logger().info("---------------Payload dropped.-------------------")


    def fly_forward_check(self, threshold=0.2):
        """Check if the drone has reached the drop area."""
        current_x = self.vehicle_local_position.x
        current_y = self.vehicle_local_position.y
        error = math.sqrt((current_x - self.DropArea_x)**2 + (current_y - self.DropArea_y)**2)
        if self.log_counter % 10 == 0:
            self.get_logger().info(f"--当前x：{current_x:.2f},y:{current_y:.2f}米，--目标x：{self.DropArea_x:.2f} 米，y:{self.DropArea_y:.2f},--误差：{error:.2f} 米")
        if error < threshold:
            self.is_AtDropArea = True
    
    def first_alignment_check(self, target_x, target_y):
        """Check first alignment with the target."""
        current_x = self.vehicle_local_position.x
        current_y = self.vehicle_local_position.y
        is_align_now = self.first_alignment_checker.check(
            current_x,
            current_y,
            target_x=target_x,
            target_y=target_y
)       
        if is_align_now:
            self.first_alignment_complete = True
            self.second_alignment_checker.reset()
            self.get_logger().info("------------------------first对准完成！------------------------")

    def second_alignment_check(self, target_x, target_y):
        """Check second alignment with the target."""
        is_align_now = self.second_alignment_checker.check(
    current_x=self.vehicle_local_position.x,
    current_y=self.vehicle_local_position.y,
            target_x=target_x,
            target_y=target_y
        )
        if is_align_now:
            self.second_alignment_complete = True
            self.get_logger().info("-------------------------second对准完成！------------------------")

    def calculate_drop_area_position(self,x,y):
        
        '''
        计算投放区域位置
        '''
        x_target = x*math.cos(self.init_yaw)-y*math.sin(self.init_yaw) + self.initial_x
        y_target = x*math.sin(self.init_yaw)+y*math.cos(self.init_yaw) + self.initial_y
        if self.log_counter % 10 == 0:
            self.get_logger().info(f"Flying to FRDposition: x={x:.3f}, y={y:.3f}")
        return x_target, y_target

    def coordinate_NED2FRD(self,x_NED,y_NED):
        '''
        将NED坐标转换为FRD坐标。
        '''
        x_FRD = (x_NED-self.initial_x)*math.cos(self.init_yaw)+(y_NED-self.initial_y)*math.sin(self.init_yaw)
        y_FRD = -(x_NED-self.initial_x)*math.sin(self.init_yaw)+(y_NED-self.initial_y)*math.cos(self.init_yaw)
        return x_FRD, y_FRD

    def coordinate_FRD2NED(self,x,y):
        '''
        将FRD坐标转换为NED坐标。
        '''
        x_target = x*math.cos(self.init_yaw)-y*math.sin(self.init_yaw) + self.initial_x
        y_target = x*math.sin(self.init_yaw)+y*math.cos(self.init_yaw) + self.initial_y

        return x_target, y_target

    def adjust_to_target(self):
        """Adjust drone position towards the current target."""   
        is_in_second_alignment = self.first_alignment_complete and not self.second_alignment_complete
        
        # 如果正处于第二次对准阶段，无条件检查超时
        if is_in_second_alignment:
            # 启动计时器 (如果尚未启动)
            if self.second_align_start_timestamp is None:
                self.second_align_start_timestamp = self.get_clock().now()
                self.get_logger().info(f"第二次对准开始，启动 {self.second_align_maxtime} 秒超时计时器。")

            # 计算已过时间
            elapsed_drop_time = (self.get_clock().now() - self.second_align_start_timestamp).nanoseconds / 1e9

            # 检查是否超时
            if elapsed_drop_time > self.second_align_maxtime:
                self.get_logger().warn(f"第二次对准超时 ({elapsed_drop_time:.1f}s > {self.second_align_maxtime}s)，强制执行投放！")
                
                # <<< 开始投放逻辑 (从原代码中移动至此) >>>
                if not self.Is_Finish_1st_Drop:
                    self.drop_payload(-1.0, 1.0)
                    self.get_logger().info("——————————————————————DROP (TIMEOUT)————————————————————————")
                    self.Is_Finish_1st_Drop = True
                    self.second_align_start_timestamp = None # 重置计时器
                elif not self.Is_Finish_2nd_Drop:
                    self.drop_payload(1.0, -1.0)
                    self.get_logger().info("——————————————————————DROP (TIMEOUT)————————————————————————")
                    self.Is_Finish_2nd_Drop = True
                    self.second_align_start_timestamp = None # 重置计时器
                
                self.droping_x = self.vehicle_local_position.x
                self.droping_y = self.vehicle_local_position.y
                self.droping_z = self.vehicle_local_position.z
                # <<< 投放逻辑结束 >>>

                return # 既然已经超时投放，直接结束本次函数调用
            
      
        if self.target_position:
            # ========== pid控制实现，记录目标像素坐标 ========== 
            from datetime import datetime
            
            # 确定当前对准阶段
            alignment_stage = "unknown"
            if not self.first_alignment_complete:
                alignment_stage = "first_alignment"
            elif not self.second_alignment_complete:
                alignment_stage = "second_alignment"
            else:
                alignment_stage = "completed"
            
            # 确定桶类型（这里可以根据实际情况调整）
            bucket_type = "unknown"
            if hasattr(self, 'current_target_name'):
                bucket_type = self.current_target_name
            
            # 记录日志
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(self.pixel_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    self.target_position.x,
                    self.target_position.y,
                    bucket_type,
                    alignment_stage
                ])
            # ========== 原有控制逻辑 ==========
            # 获取当前位置
            current_xned, current_yned = self.vehicle_local_position.x, self.vehicle_local_position.y
            current_x, current_y = self.coordinate_NED2FRD(current_xned, current_yned)
            
            # 📌 计算相机坐标系误差（考虑相机中心偏移）
            dx_cam = self.target_position.y + self.depthcam_xoffset  # 相机中心相对投放中心的Y偏差
            dy_cam = -self.target_position.x + self.depthcam_yoffset  # 相机中心相对投放中心的X偏差
            distance = math.hypot(dx_cam, dy_cam)
            
            # 📌 根据误差大小选择控制策略
            if distance < self.epsilon:
                # ——————— 细调阶段：PID控制 ———————
                if self.log_counter % 10 == 0:
                    self.get_logger().info(f"PID细调阶段 - 误差:{distance:.3f}m < 阈值:{self.epsilon:.3f}m")
                
                # 计算误差项
                error_x = dx_cam
                error_y = dy_cam
                
                # 📌 积分项计算（带限幅防饱和）
                self.integral_x += error_x * self.dt
                self.integral_y += error_y * self.dt
                # 积分限幅
                self.integral_x = max(min(self.integral_x, self.max_integral), -self.max_integral)
                self.integral_y = max(min(self.integral_y, self.max_integral), -self.max_integral)
                
                # 📌 微分项计算
                derivative_x = (error_x - self.last_error_x) / self.dt
                derivative_y = (error_y - self.last_error_y) / self.dt
                
                # 📌 PID控制量计算
                control_x = (self.Kp_fine * error_x + 
                           self.Ki * self.integral_x + 
                           self.Kd * derivative_x)
                control_y = (self.Kp_fine * error_y + 
                           self.Ki * self.integral_y + 
                           self.Kd * derivative_y)
                
                # 保存本次误差用于下次微分计算
                self.last_error_x = error_x
                self.last_error_y = error_y
                
                if self.log_counter % 10 == 0:
                    self.get_logger().info(f"PID输出: P={self.Kp_fine * error_x:.3f}, I={self.Ki * self.integral_x:.3f}, D={self.Kd * derivative_x:.3f}")
                
            else:
                # ——————— 大误差阶段：饱和P控制 ———————
                if self.log_counter % 10 == 0:
                    self.get_logger().info(f"饱和P控制阶段 - 误差:{distance:.3f}m >= 阈值:{self.epsilon:.3f}m")
                
                # 📌 饱和比例控制
                scale = self.align_maxstep / distance
                control_x = dx_cam * scale
                control_y = dy_cam * scale
                
                # 清零PID状态，避免积累
                self.integral_x = 0.0
                self.integral_y = 0.0
                self.last_error_x = 0.0
                self.last_error_y = 0.0
                
                if self.log_counter % 10 == 0:
                    self.get_logger().info(f"饱和P输出: scale={scale:.3f}, 最大步长={self.align_maxstep:.3f}m")

            # ============== 目标位置计算 ==============
            # 计算FRD目标位置
            target_x_FRD = current_x + control_x
            target_y_FRD = current_y + control_y
            
            # 转换为NED坐标
            target_x_NED, target_y_NED = self.coordinate_FRD2NED(target_x_FRD, target_y_FRD)
            
            # 精确目标位置（用于对准检查）
            precise_target_x_FRD = current_x + dx_cam
            precise_target_y_FRD = current_y + dy_cam
            precise_target_x_NED, precise_target_y_NED = self.coordinate_FRD2NED(precise_target_x_FRD, precise_target_y_FRD)
            
            # ============== 两次对准逻辑 ==============
            # First alignment
            if not self.first_alignment_complete:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行第一次对准")
                self.fly_to_position(target_x_NED, target_y_NED, self.first_alingment_tartget_height)
                self.first_alignment_check(precise_target_x_NED, precise_target_y_NED)
                self.last_found_x_NED = target_x_NED
                self.last_found_y_NED = target_y_NED
                self.last_found_z_NED = self.first_alingment_tartget_height

            elif self.first_alignment_complete and not self.second_alignment_complete:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行第二次精确对准")
                self.fly_to_position(target_x_NED, target_y_NED, self.first_alingment_tartget_height + self.afterAlign_descentHeight)
                self.second_alignment_check(precise_target_x_NED, precise_target_y_NED)
                self.last_found_x_NED = target_x_NED
                self.last_found_y_NED = target_y_NED
                self.last_found_z_NED = self.first_alingment_tartget_height + self.afterAlign_descentHeight

            self.target_position = None
            
            if self.first_alignment_complete and self.second_alignment_complete :
                if not self.Is_Finish_1st_Drop:
                    self.drop_payload(-1.0,1.0)
                    self.get_logger().info("——————————————————————DROP————————————————————————")
                    self.Is_Finish_1st_Drop = True
                    self.second_align_start_timestamp = None
                elif not self.Is_Finish_2nd_Drop:
                    self.drop_payload(1.0,-1.0)
                    self.get_logger().info("——————————————————————DROP————————————————————————")
                    self.Is_Finish_2nd_Drop = True
                    self.second_align_start_timestamp = None
                
                self.droping_x = self.vehicle_local_position.x
                self.droping_y = self.vehicle_local_position.y
                self.droping_z = self.vehicle_local_position.z

                
        else:
            if self.last_found_x_NED and self.last_found_y_NED and self.last_found_z_NED:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("使用上次记录")
                self.fly_to_position(self.last_found_x_NED, self.last_found_y_NED, self.last_found_z_NED)
            else:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("上次记录不存在")
                self.fly_to_position(self.DropArea_x, self.DropArea_y, self.first_alingment_tartget_height)
    
    def publish_trajectory_setpoint(self):
        """发布轨迹设定点"""
        msg = TrajectorySetpoint()
        
        if self.vehicle_local_position is not None:
            msg.position = [
                float(self.vehicle_local_position.x), 
                float(self.vehicle_local_position.y),  
                float(self.vehicle_local_position.z)
            ]
            
            # 使用初始航向角（如果有记录）
            if self.initial_position is not None:
                msg.yaw = float(self.initial_position.heading)
            else:
                msg.yaw = float(self.vehicle_local_position.heading)
        else:
            msg.position = [float('nan'), float('nan'), float('nan')]
            msg.yaw = float('nan')
        
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
    
    def is_at_trigger_position(self):
        """检查是否到达触发位置（前方30米）"""
        if self.initial_position is None or self.vehicle_local_position is None:
            return False
        
        forward_distance = self.calculate_forward_distance()
        if self.log_counter % 10 ==0:
            self.get_logger().info(
                f"Position check - Forward: {forward_distance:.1f}m, "
                f"Target: {self.trigger_distance}m"
            )
        
        # 使用前进距离作为主要判断条件
        return forward_distance >= (self.trigger_distance - self.position_threshold)
    
    def start_mission(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=4.0, param2=3.0)
        self.get_logger().info("Switching to Mission mode")
    
    def calculate_forward_distance(self):
        """计算无人机在前进方向上的距离"""
        if self.initial_position is None or self.vehicle_local_position is None:
            return 0.0
        
        x_frd, y_frd = self.coordinate_NED2FRD(
        self.vehicle_local_position.x,
        self.vehicle_local_position.y
    )
    
    # FRD坐标系的X值就是我们需要的“前进距离”
        return x_frd    
    
    #定时器
    def timer_callback(self) -> None:
        """Callback function for the timer."""
        
        if not self.is_vision_ready:
            # 只有在第一次进入timer_callback时执行
            if self.vision_controller.load_model():
                self.is_vision_ready = True
                self.get_logger().info("视觉系统准备就绪，开始执行任务逻辑。")
            else:
                self.get_logger().error("视觉系统初始化失败，节点将不执行任务。")
                return # 如果模型加载失败，直接返回，不执行后续逻辑          
        
        # 更新日志计数器
        self.log_counter += 1
        
        # --- 视觉处理部分 ---
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("无法捕获图像")
            return
        
        # 调用视觉控制器处理图像
        visual_state, visual_command, annotated_frame = self.vision_controller.process_frame(frame)
        cv2.imshow("Drone View", annotated_frame)
        cv2.waitKey(1)        
        
        
        if self.state == MissionState.IDLE:
            # 等待飞控连接并准备就绪
            if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self.get_logger().info("Vehicle is armed. Starting mission.")
                
                # 记录初始位置
                if self.vehicle_local_position is not None:
                    self.initial_position = self.vehicle_local_position
                    self.initial_x = self.initial_position.x
                    self.initial_y = self.initial_position.y
                    self.initial_z = self.initial_position.z
                    self.init_yaw = self.initial_position.heading
                    self.get_logger().info(
                        f"Initial position recorded: "
                        f"x={self.initial_position.x:.2f}, "
                        f"y={self.initial_position.y:.2f}, "
                        f"z={self.initial_position.z:.2f}, "
                        f"heading={self.initial_position.heading:.2f}"
                    )
                else:
                    self.get_logger().warn("No position data available, using default")
                
                self.state = MissionState.STARTING_MISSION
                self.start_mission()
            else:
                self.get_logger().info("arm the vehicle")
                self.arm()

        elif self.state == MissionState.STARTING_MISSION:
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_MISSION:
                self.get_logger().info("Successfully switched to Mission mode.")
                self.state = MissionState.IN_MISSION

        elif self.state == MissionState.IN_MISSION:
            # 新的触发条件：检查是否飞到前方特定距离
            if self.is_at_trigger_position():
                self.get_logger().info(f"✓ Reached {self.trigger_distance}m forward position. Switching to Offboard.")
                self.DropArea_x = self.vehicle_local_position.x
                self.DropArea_y = self.vehicle_local_position.y
                # 记录当前航向角
                self.init_yaw = self.vehicle_local_position.heading
                # 设定第一次对准的目标高度
                self.first_alingment_tartget_height = self.initial_z + self.first_alignment_height

                # 重置计数器并转换到准备状态
                self.offboard_setpoint_counter = 0
                self.state = MissionState.PREPARING_OFFBOARD
                
            else:
                # 显示当前位置信息（降低频率避免日志过多）
                if hasattr(self, '_last_log_time'):
                    if time.time() - self._last_log_time > 1.0:  # 每秒显示一次
                        forward_dist = self.calculate_forward_distance()
                        self.get_logger().info(f"In mission, forward distance: {forward_dist:.1f}m / {self.trigger_distance}m")
                        self._last_log_time = time.time()
                else:
                    self._last_log_time = time.time()
        
        elif self.state == MissionState.PREPARING_OFFBOARD:
            # 持续发送心跳和设定点，目标为保持当前位置
            self.publish_offboard_control_heartbeat_signal()
            # 发布一个设定点让无人机稳定在当前位置
            self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.vehicle_local_position.z)

            # 等待大约1秒 (33次调用 * 0.03秒/次)，以建立稳定的指令流
            if self.offboard_setpoint_counter >= 33:
                self.get_logger().info("设定点指令流已建立，尝试切换到Offboard模式。")
                self.state = MissionState.SWITCHING_TO_OFFBOARD1
            
            self.offboard_setpoint_counter += 1
        
        elif self.state == MissionState.SWITCHING_TO_OFFBOARD1:
            # **重要**: 进入Offboard模式前必须持续发送设定点
            self.publish_offboard_control_heartbeat_signal()
            self.publish_trajectory_setpoint() # 先发送一个保持当前位置的指令
            self.DropArea_x = self.vehicle_local_position.x
            self.DropArea_y = self.vehicle_local_position.y
            self.init_yaw = self.vehicle_local_position.heading
            self.engage_offboard_mode()
            
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.get_logger().info("Successfully switched to Offboard mode.")
                self.state = MissionState.IN_OFFBOARD


        elif self.state == MissionState.IN_OFFBOARD:
            self.publish_offboard_control_heartbeat_signal()
            if not self.is_AtDropArea:
                self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.first_alingment_tartget_height)            
                self.fly_forward_check()
            
            
            if self.is_AtDropArea and not self.is_FinishDrop:
                #======增加限时模块========
                if self.drop_phase_start_time is None:
                    self.get_logger().info(f"已到达投水区域，启动 {self.drop_phase_timeout} 秒投放任务倒计时。")
                    self.drop_phase_start_time = self.get_clock().now()
                
                elapsed_drop_time = (self.get_clock().now() - self.drop_phase_start_time).nanoseconds / 1e9
                if elapsed_drop_time > self.drop_phase_timeout:
                    if self.timeout_drop_start_time is None:
                        self.timeout_drop_start_time = self.get_clock().now()
                    self.get_logger().warn(f"投放阶段超时（超过 {self.drop_phase_timeout} 秒），任务中止，进入侦察。")
                    if self.timeout_drop_count==0:
                        self.drop_payload(-1,1)
                        self.timeout_drop_count+=1
                    elapsed_time = ( self.get_clock().now()-self.timeout_drop_start_time).nanoseconds / 1e9

                    if elapsed_time > self.timeout_drop_delay:
                        self.drop_payload(1,-1)
                        self.timeout_drop_count+=1
                    if self.timeout_drop_count == 2:
                        self.get_logger().warn(f"投放阶段超时（超过 {self.drop_phase_timeout} 秒），任务中止，已全部投放，进入侦察。")
                        self.Drop_mission_state = MissionState.TIMEOUT
        
                        return
                #======增加限时模块========
                
                if self.Drop_mission_state == MissionState.GLOBAL_SEARCH:
                    
                    #全局搜索限时10s
                    if self.search1_phase_start_time is None:
                        self.get_logger().info(f"开始全局搜索，限时 {self.search_timeout} 秒。")
                        self.search1_phase_start_time = self.get_clock().now()
                                        
                    #上升到global——search高度
                    self.global_search_target_z = float(self.initial_z+self.global_search_height)
                    self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.global_search_target_z)
                    if self.log_counter % 10 == 0:
                        self.get_logger().info("进行全局搜索")

                    if self.vision_controller.initial_target_map:
                        self.get_logger().info("全局搜索完成，进入目标打击循环。")
                        self.Drop_mission_state = MissionState.TARGETING_CYCLE
                    
                    if self.search1_phase_start_time and (self.get_clock().now() - self.search1_phase_start_time).nanoseconds / 1e9 > self.search_timeout:
                        if self.timeout_drop_start_time is None:
                            self.timeout_drop_start_time = self.get_clock().now()
                        self.get_logger().warn(f"第一次全局搜索阶段超时（超过 {self.search_timeout} 秒），任务中止")
                        if self.timeout_drop_count==0:
                            self.drop_payload(-1,1)
                            self.timeout_drop_count+=1
                        elapsed_time = ( self.get_clock().now()-self.timeout_drop_start_time).nanoseconds / 1e9
                        if elapsed_time > self.timeout_drop_delay:
                            self.drop_payload(1,-1)
                            self.timeout_drop_count+=1
                        if self.timeout_drop_count==2:    
                            self.get_logger().warn(f"全局搜索超时（超过 {self.search_timeout} 秒），未找到目标，全部投放。")
                            self.Drop_mission_state = MissionState.TIMEOUT
                            return
                    
                elif self.Drop_mission_state == MissionState.RECONFIRMING_TARGETS: # <<< 新增的处理块
                    if self.search2_phase_start_time is None:
                        self.get_logger().info(f"开始第二次全局搜索，限时 {self.search_timeout} 秒。")
                        self.search2_phase_start_time = self.get_clock().now()
                    if self.search2_phase_start_time and (self.get_clock().now() - self.search2_phase_start_time).nanoseconds / 1e9 > self.search_timeout:
                        if self.timeout_drop_start_time is None:
                            self.timeout_drop_start_time = self.get_clock().now()
                        self.get_logger().warn(f"第二次全局搜索超时（超过 {self.search_timeout} 秒），任务中止.")
                        if self.timeout_drop_count==0:
                            self.drop_payload(-1,1)
                            self.timeout_drop_count+=1
                        elapsed_time = ( self.get_clock().now()-self.timeout_drop_start_time).nanoseconds / 1e9
                        if elapsed_time > self.timeout_drop_delay:
                            self.drop_payload(1,-1)
                            self.timeout_drop_count+=1
                        if self.timeout_drop_count==2:    
                            self.get_logger().error(f"第二次全局搜索超时（超过 {self.search_timeout} 秒），未找到目标，已全部投放。")
                            self.Drop_mission_state = MissionState.TIMEOUT
                            return
                    
                    self.get_logger().info("正在爬升并重新确认目标位置...")
                    self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.global_search_target_z)
                    
                    num_targets_seen = self.vision_controller.get_current_detection_count()
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"重新确认中... 当前看到 {num_targets_seen} / 3 个目标")

                    # 当再次看到大于2个目标时，才真正进入下一个目标的打击流程
                    if num_targets_seen > 1:
                        self.get_logger().info("重新确认成功！已找到至少两个目标。准备攻击下一个目标。")

                        self.first_alignment_complete = False
                        self.second_alignment_complete = False
                        self.Is_Descending_to_depth_camera_height = False
                        self.first_alignment_checker.reset()
                        self.second_alignment_checker.reset()
                        
                        # 转换回目标打击循环状态
                        self.Drop_mission_state = MissionState.TARGETING_CYCLE
                
                elif self.Drop_mission_state == MissionState.PROACTIVE_SEARCH:
                    # --- 在这个状态下，无人机爬升并向下一个目标的大致方向移动 ---
                    # 1. 计算主动搜索的目标点 (只在第一次进入时计算)
                    if not self.is_proactive_target_set:
                        self.get_logger().info("计算主动搜索的目标点...")
                        
                        # 确定下一个目标是左还是右
                        next_target_name = self.target_priority[self.current_target_index]
                        
                        y_offset_frd = 0.0
                        if "Left" in next_target_name:
                            y_offset_frd = -self.proactive_search_distance # FRD坐标系中，Y轴负方向是左
                            self.get_logger().info(f"下一个目标在左侧，向左移动 {self.proactive_search_distance} 米。")
                        elif "Right" in next_target_name:
                            y_offset_frd = self.proactive_search_distance # FRD坐标系中，Y轴正方向是右
                            self.get_logger().info(f"下一个目标在右侧，向右移动 {self.proactive_search_distance} 米。")
                        
                        # 基于投水区的中心点，计算偏移后的NED坐标
                        # 注意：这里我们使用 self.DropArea_x 和 self.DropArea_y 作为基准点
                        # 这样可以避免从有微小误差的投放点开始计算
                        base_x, base_y = self.DropArea_x, self.DropArea_y
                        
                        # 将FRD的偏移量转换为NED的偏移量
                        delta_x_ned = 0 * math.cos(self.init_yaw) - y_offset_frd * math.sin(self.init_yaw)
                        delta_y_ned = 0 * math.sin(self.init_yaw) + y_offset_frd * math.cos(self.init_yaw)

                        # 计算最终的NED目标点
                        self.proactive_target_x = base_x + delta_x_ned
                        self.proactive_target_y = base_y + delta_y_ned
                        
                        self.is_proactive_target_set = True
                        self.get_logger().info(f"主动搜索目标点(NED): x={self.proactive_target_x:.2f}, y={self.proactive_target_y:.2f}")

                    # 2. 命令无人机飞向目标点，并爬升到全局搜索高度
                    self.publish_position_setpoint(self.proactive_target_x, self.proactive_target_y, self.global_search_target_z)

                    # 3. 检查是否已到达目标点
                    current_x = self.vehicle_local_position.x
                    current_y = self.vehicle_local_position.y
                    height_error = abs(self.vehicle_local_position.z - self.global_search_target_z)
                    distance_error = math.sqrt((current_x - self.proactive_target_x)**2 + (current_y - self.proactive_target_y)**2)
                    
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"主动搜索中... 距离目标点: {distance_error:.2f}米, 高度误差: {height_error:.2f}米")

                    # 如果水平和垂直都接近目标位置，则认为此阶段完成
                    if distance_error < 0.3 and height_error < 0.3:
                        self.get_logger().info("主动搜索阶段完成，已到达预定搜索区域。")
                        self.get_logger().info("现在切换到悬停确认阶段(RECONFIRMING_TARGETS)。")
                        
                        # 状态切换到确认阶段
                        self.Drop_mission_state = MissionState.RECONFIRMING_TARGETS
                        
                        # 重置标志位，以便下次（如果还有第三个目标）可以再次使用
                        self.is_proactive_target_set = False

                # GLOBAL_SEARCH执行一次之后，mission_state状态都为TARGETING_CYCLE
                elif self.Drop_mission_state == MissionState.TARGETING_CYCLE:
                    if self.visited_targets_count >= len(self.target_priority):
                        self.Drop_mission_state = MissionState.TIMEOUT
                        return
                    current_target_name = self.target_priority[self.current_target_index]

                    if self.vision_controller.visual_state not in [VisualState.CENTERING, VisualState.TARGET_LOCKED]:
                        self.get_logger().info(f"设置新目标: [{current_target_name}]")
                        self.vision_controller.set_target(current_target_name)

                    if visual_state == VisualState.CENTERING:
                        self.execute_visual_command(visual_command)

                    elif visual_state == VisualState.TARGET_LOCKED:
            
                        # 在这里执行下降和投放逻辑
                        if not self.Is_Descending_to_depth_camera_height:
                            self.get_logger().info(f"目标 [{current_target_name}] 已锁定，准备下降。")
                            self.publish_position_setpoint(self.vehicle_local_position.x, self.vehicle_local_position.y, self.first_alingment_tartget_height)                        
                            if abs(self.vehicle_local_position.z - self.first_alingment_tartget_height) < 0.2:
                                self.Is_Descending_to_depth_camera_height = True
                                self.get_logger().info(f"目标 [{current_target_name}] 已锁定，下降完成。")
                        
                        if self.Is_Descending_to_depth_camera_height == True :
                            self.adjust_to_target() 
                            if self.Is_Finish_1st_Drop and self.visited_targets_count == 0:                        
                                # 更新任务进度
                                self.visited_targets_count += 1
                                self.current_target_index += 1
                                
                                if self.visited_targets_count < 2:
                                    # 投放完成，不要直接设置下一个目标！
                                    # 而是进入“重新确认”状态
                                    self.get_logger().info("第一次投放完成。进入主动搜索阶段。")
                                    self.Drop_mission_state = MissionState.PROACTIVE_SEARCH                                    
                                    # 重置视觉控制器到通用搜索模式
                                    self.vision_controller.reset_to_search_mode()
                                else:
                                    pass
                            if self.Is_Finish_1st_Drop and self.Is_Finish_2nd_Drop:
                                self.is_FinishDrop = True
                
                elif self.Drop_mission_state == MissionState.TIMEOUT:
                    self.is_FinishDrop = True
                    self.get_logger().info("超时，已进入下一模式")

            if self.is_FinishDrop: 
                self.state = MissionState.DROP_COMPLETE
                self.get_logger().info("任务完成")
        
        elif self.state == MissionState.DROP_COMPLETE:
            self.get_logger().info("投放区域任务结束，重新返回mission模式")
            self.start_mission()
            self.state = MissionState.SWITCHING_TO_MISSION
        
        elif self.state == MissionState.SWITCHING_TO_MISSION:
            # 不再发送Offboard指令
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_MISSION:
                self.get_logger().info("成功切换回mission模式")
                self.state = MissionState.MISSION_RESUMED

        elif self.state == MissionState.MISSION_RESUMED:
            # 在这个状态下，节点可以什么都不做，只打印日志，或者准备关闭
            if self.log_counter % 100 == 0: # 降低日志频率
                self.get_logger().info("Drone is now in Mission Mode. This node is idle.")
            pass # 什么都不做

        if self.offboard_setpoint_counter < 30:
            self.offboard_setpoint_counter += 1

def main(args=None) -> None:
    # 1. 初始化rclpy，它会处理ROS特有的参数
    rclpy.init(args=args)

    # 2. 设置我们自己的命令行参数解析器
    parser = argparse.ArgumentParser(description="Offboard control script for PX4 drone mission.")
    
    # 添加你想通过命令行配置的参数
    parser.add_argument('--model-path', type=str, default='/home/weights/0711.engine',
                        help='Path to the object detection model file.')
    parser.add_argument('--photo-path', type=str, default='/home/image_recodes',
                        help='Base directory to save captured photos.')
    parser.add_argument('--video-path', type=str, default='/home/video_recodes',
                        help='Base directory to save recorded mission videos.')
    parser.add_argument('--camera-hint', type=str, default='USB',
                        help='Hint to find the camera device name (e.g., "USB", "C920").')
    
    parser.add_argument('--first-alignment-height', type=float, default=-2.0,
                        help='Takeoff height in meters (negative value for altitude).')
    parser.add_argument('--descent-height', type=float, default=1.0,
                        help='Descent height after first alignment in meters (positive value).')
    
    parser.add_argument('--forward-x', type=float, default=2.3,
                        help='Forward distance to fly to the drop area in meters.')
    parser.add_argument('--search-height', type=float, default=-5.0,
                        help='Global search height in meters (negative value for altitude).')
   
    parser.add_argument('--align-maxstep', type=float, default=0.2,
                        help='Maximum step size for each alignment adjustment.')
    parser.add_argument('--proactive-search-dist', type=float, default=0.6,
                        help='Distance to move sideways for proactive search.')
    
    # <<< 新增：在这里为 AlignmentChecker 添加参数 >>>
    parser.add_argument('--first-align-threshold', type=float, default=0.15,
                        help='Threshold (distance in meters) for the first alignment.')
    parser.add_argument('--first-align-time-window', type=float, default=2.0,
                        help='Time window (seconds) to maintain stability for the first alignment.')
    parser.add_argument('--first-align-check-freq', type=int, default=5,
                        help='Check frequency (how many timer calls per check) for the first alignment.')
    parser.add_argument('--second-align-threshold', type=float, default=0.10,
                        help='Threshold (distance in meters) for the second alignment.')
    parser.add_argument('--second-align-time-window', type=float, default=3.0,
                        help='Time window (seconds) to maintain stability for the second alignment.')
    parser.add_argument('--second-align-check-freq', type=int, default=5,
                        help='Check frequency (how many timer calls per check) for the second alignment.')    
    
    parser.add_argument('--drop-phase-timeout', type=float, default=90.0,
                        help='Maximum time in seconds for the entire dropping phase.')
    parser.add_argument('--search-timeout', type=float, default=10.0,
                        help='Maximum time in seconds for each search attempt.')
    parser.add_argument('--second-align-maxtime', type=float, default=8.0,
                        help='Maximum time in seconds for each search attempt.')
    
    
    parser.add_argument('--depthcam_xoffset', type=float, default=-0.065,
                        help='Maximum time in seconds for each search attempt.')
    parser.add_argument('--depthcam_yoffset', type=float, default=0.033,
                        help='Maximum time in seconds for each search attempt.')
    parser.add_argument('--trigger-distance', type=float, default=32.5,
                    help='Forward distance in meters to trigger offboard mode.')
    parser.add_argument('--position-threshold', type=float, default=0.5,
                    help='Position tolerance in meters for reaching the trigger distance.')
    
    # 3. 解析参数
    # 使用 rclpy.utilities.remove_ros_args 来确保我们只解析自己的参数，
    # 这样可以安全地与 ROS2 的参数（如 --ros-args）一起使用。
    custom_args = parser.parse_args(args=rclpy.utilities.remove_ros_args(args=sys.argv)[1:])

    print('Starting offboard control node with custom parameters...')
    
    # 4. 将解析后的参数传入节点
    offboard_control = OffboardControl(args=custom_args)

    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        print("程序被用户中断 (Ctrl+C)")
    finally:
        print("Shutting down the node...")
        offboard_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)