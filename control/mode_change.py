#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from enum import Enum
import time
import math

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import MissionResult
from px4_msgs.msg import VehicleLocalPosition

class MissionState(Enum):
    """简单的状态机"""
    IDLE = 0
    STARTING_MISSION = 1
    IN_MISSION = 2
    SWITCHING_TO_OFFBOARD1 = 3
    IN_OFFBOARD = 4
    SWITCHING_TO_MISSION = 5
    MISSION_RESUMED = 6
    DONE = 7

class OffboardMissionNode(Node):

    def __init__(self):
        super().__init__('offboard_mission_node')

        # 配置QoS，确保与PX4的通信稳定
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 创建发布者和订阅者
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        # 添加位置订阅来监控任务进度
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', 
            self.local_position_callback, qos_profile)

        self.mission_result_sub = self.create_subscription(
            MissionResult, '/fmu/out/mission_result', self.mission_result_callback, qos_profile)
        
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)

        # 状态机和任务参数
        self.state = MissionState.IDLE
        self.vehicle_status = None
        self.mission_result = None
        self.vehicle_local_position = None
        
        # 新增：位置触发参数
        self.initial_position = None  # 解锁时的初始位置
        self.trigger_distance = 28.0  # 触发距离：30米
        self.position_threshold = 1.0  # 位置判断阈值：1米        board_task_start_time = None
        self.offboard_task_duration = 10.0  # Offboard 任务持续10秒

        # 创建一个定时器来驱动主逻辑
        self.timer = self.create_timer(0.1, self.timer_callback) # 10Hz

    def vehicle_status_callback(self, msg):
        """获取飞控状态"""
        self.vehicle_status = msg

    def local_position_callback(self, msg):
        """获取无人机位置"""
        self.vehicle_local_position = msg

    def mission_result_callback(self, msg):
        """获取航线执行结果"""
        self.mission_result = msg
        self.get_logger().info(f"Mission result updated: sequence reached: {msg.seq_reached}")

    def calculate_forward_distance(self):
        """计算无人机在前进方向上的距离"""
        if self.initial_position is None or self.vehicle_local_position is None:
            return 0.0
        
        # 计算从初始位置到当前位置的向量
        dx = self.vehicle_local_position.x - self.initial_position.x
        dy = self.vehicle_local_position.y - self.initial_position.y
        
        # 获取初始航向角
        initial_yaw = self.initial_position.heading
        
        # 计算在初始航向方向上的前进距离
        # 前进方向：X轴正方向在初始航向角方向
        forward_distance = dx * math.cos(initial_yaw) + dy * math.sin(initial_yaw)
        
        return forward_distance

    def get_horizontal_distance_from_start(self):
        """计算水平距离（不考虑方向）"""
        if self.initial_position is None or self.vehicle_local_position is None:
            return 0.0
        
        dx = self.vehicle_local_position.x - self.initial_position.x
        dy = self.vehicle_local_position.y - self.initial_position.y
        
        return math.sqrt(dx*dx + dy*dy)

    def is_at_trigger_position(self):
        """检查是否到达触发位置（前方30米）"""
        if self.initial_position is None or self.vehicle_local_position is None:
            return False
        
        # 方法1：使用前进方向距离
        forward_distance = self.calculate_forward_distance()
        
        # 方法2：使用水平距离（备用）
        horizontal_distance = self.get_horizontal_distance_from_start()
        
        self.get_logger().info(
            f"Position check - Forward: {forward_distance:.1f}m, "
            f"Horizontal: {horizontal_distance:.1f}m, "
            f"Target: {self.trigger_distance}m"
        )
        
        # 使用前进距离作为主要判断条件
        return forward_distance >= (self.trigger_distance - self.position_threshold)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Arm command sent")

    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info("Disarm command sent")
        
    def start_mission(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=4.0, param2=3.0)
        self.get_logger().info("Switching to Mission mode")

    def switch_to_offboard(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to Offboard mode")
    
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

    def publish_offboard_control_mode(self):
        """发布Offboard控制模式"""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.attitude = False
        msg.acceleration = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self):
        """发布轨迹设定点"""
        msg = TrajectorySetpoint()
        
        if self.vehicle_local_position is not None:
            msg.position = [
                float(self.vehicle_local_position.x), 
                float(5.0 + self.vehicle_local_position.y),  # 偏移Y轴5米（需要悬停则去掉5米）
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

    def timer_callback(self):
        """主逻辑循环，由定时器驱动"""
        if self.vehicle_status is None:
            self.get_logger().warn("Waiting for vehicle status...")
            return

        # 状态机逻辑
        if self.state == MissionState.IDLE:
            # 等待飞控连接并准备就绪
            if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self.get_logger().info("Vehicle is armed. Starting mission.")
                
                # 记录初始位置
                if self.vehicle_local_position is not None:
                    self.initial_position = self.vehicle_local_position
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
                # 在实际应用中，你可能需要先发送起飞指令，然后开始mission
                # 这里为简化，假设飞机已在空中或通过QGC起飞
                # 如果要代码自动arm，可以在这里调用 self.arm()
                self.get_logger().info("Please arm the vehicle to start.")
                self.arm()

        elif self.state == MissionState.STARTING_MISSION:
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_MISSION:
                self.get_logger().info("Successfully switched to Mission mode.")
                self.state = MissionState.IN_MISSION

        elif self.state == MissionState.IN_MISSION:
            # 新的触发条件：检查是否飞到前方30米
            if self.is_at_trigger_position():
                self.get_logger().info("✓ Reached 30m forward position. Switching to Offboard.")
                self.state = MissionState.SWITCHING_TO_OFFBOARD1

            else:
                # 显示当前位置信息（降低频率避免日志过多）
                if hasattr(self, '_last_log_time'):
                    if time.time() - self._last_log_time > 1.0:  # 每秒显示一次
                        forward_dist = self.calculate_forward_distance()
                        self.get_logger().info(f"In mission, forward distance: {forward_dist:.1f}m / {self.trigger_distance}m")
                        self._last_log_time = time.time()
                else:
                    self._last_log_time = time.time()
        
        elif self.state == MissionState.SWITCHING_TO_OFFBOARD1:
            # **重要**: 进入Offboard模式前必须持续发送设定点
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint() # 先发送一个保持当前位置的指令
            self.switch_to_offboard()
            
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.get_logger().info("Successfully switched to Offboard mode.")
                self.state = MissionState.IN_OFFBOARD
                self.offboard_task_start_time = time.time()

        elif self.state == MissionState.IN_OFFBOARD:
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint() # 持续发送指令执行Offboard任务
            
            # 检查Offboard任务是否完成 (例如，持续10秒)
            if time.time() - self.offboard_task_start_time > self.offboard_task_duration:
                self.get_logger().info("Offboard task finished. Switching back to Mission mode.")
                self.state = MissionState.SWITCHING_TO_MISSION
                
        elif self.state == MissionState.SWITCHING_TO_MISSION:
            self.start_mission() # 发送指令切回Mission模式
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_MISSION:
                self.get_logger().info("Successfully switched back to Mission mode.")
                self.state = MissionState.MISSION_RESUMED

        elif self.state == MissionState.MISSION_RESUMED:
            # 监控任务是否继续或完成
            if self.mission_result and hasattr(self.mission_result, 'finished') and self.mission_result.finished:
                self.get_logger().info("Mission complete!")
                self.state = MissionState.DONE
                self.destroy_node()
                rclpy.shutdown()
            else:
                # 如果没有 mission_result.finished 属性，可以用其他方式判断任务完成
                self.get_logger().debug("Mission resumed, waiting for completion...")

        elif self.state == MissionState.DONE:
            pass # 什么都不做

def main(args=None):
    rclpy.init(args=args)
    node = OffboardMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)


