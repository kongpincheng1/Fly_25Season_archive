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
import RPi.GPIO as GPIO



class DronePositionChecker:
    def __init__(self, logger_func,tolerance=0.25, duration=3.0):
        """
        初始化无人机位置检查器。
        :param tolerance: 位置变化容差（米）。
        :param duration: 判断稳定所需的时间（秒）。
        """
        self.logger_func = logger_func
        self.tolerance = tolerance
        self.duration = duration
        self.positions = deque()  # 用于存储最近位置和时间戳

    def update_position(self, position):
        """
        更新无人机的当前位置。
        :param position: 位置元组 (x, y, z)。
        """
        current_time = time.time()
        self.positions.append((position, current_time))

        # 移除超出时间窗口的旧数据
        while self.positions and current_time - self.positions[0][1] > self.duration:
            self.positions.popleft()

    def is_stable(self):
        """
        判断无人机当前位置是否稳定。
        :return: 如果稳定返回 True，否则返回 False。
        """
        if len(self.positions) < 100:
            return False  # 数据不足时无法判断

        # 计算所有位置的最大距离
        max_distance = 0
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                dist = self._distance(self.positions[i][0], self.positions[j][0])
                max_distance = max(max_distance, dist)

        self.logger_func(f"最大误差为{max_distance:.4f},m.")
        self.logger_func(f"{self.tolerance},return:{max_distance<=self.tolerance}")
        return max_distance <= self.tolerance

    @staticmethod
    def _distance(pos1, pos2):
        """
        计算两点之间的欧几里得距离。
        :param pos1: 点1 (x, y, z)。
        :param pos2: 点2 (x, y, z)。
        :return: 两点之间的距离。
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

class AlignmentChecker:
    def __init__(self, logger_func, threshold=0.15, time_window=2.0, check_frequency=10):
        """
        :param logger_func: 日志记录函数（例如 Node.get_logger().info）
        :param threshold: 误差阈值 (米)
        :param time_window: 时间窗口 (秒)
        :param check_frequency: 误差检查频率 (每秒次数)
        """
        self.logger_func = logger_func
        self.threshold = threshold
        self.time_window = time_window
        self.check_frequency = check_frequency
        self.error_deque = deque(maxlen=int(time_window * check_frequency))  # 固定大小的队列
        self.first_aligned = False

    def alignment_check(self, current_x, current_y, target_x, target_y):
        # 计算当前位置与目标点的误差
        result = math.sqrt(
            (current_x - target_x)**2 + (current_y - target_y)**2
        )

        # 将误差记录到队列中
        self.error_deque.append(result)

        # 检查队列是否已满
        if len(self.error_deque) == self.error_deque.maxlen:
            # 判断队列内所有误差是否小于阈值
            if all(error < self.threshold for error in self.error_deque):
                self.first_aligned = True
                self.logger_func(f"目标对准成功：误差连续 {len(self.error_deque)} 次小于阈值 {self.threshold}")
                # return True

        # 打印当前误差和队列状态
        self.logger_func(f"当前误差： {result},")
        # return False


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
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
        self.center_height_subscriber = self.create_subscription(Float32, '/current_height',
                                                                    self.current_height_callback, 10)


        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        
        self.target_position = None
        self.target_reached = False
        self.CurrentHeightFromCamera = 0
        
        self.already_reached = False
        self.first_aligned = False
        self.second_aligned = False
        self.last_found_x_FRD = None
        self.last_found_y_FRD = None
        self.last_found_z_FRD = None

        self.takeoff_height = -0.7
        self.forward_x = 2

        self.initial_z = None  # 初始高度
        self.initial_x = None  #
        self.initial_y = None
        self.init_yaw = None

        self.DropArea_x = None
        self.DropArea_y = None

        self.takeoff_target_height = None
        self.is_ReadyToTakeoff = False   #判断飞机是否具备起飞条件
        self.is_AtTakeoffHeight = False #判断飞机是否到达起飞高度
        self.is_AtDropArea = False      #判断飞机是否到达投水区
        self.is_FinishDrop = False      #判断飞机是否完成投水

        self.droping_x = None
        self.droping_y = None
        self.droping_z = None


        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        #初始化位置判断器
        self.initPositionChecker = DronePositionChecker(
            logger_func=self.get_logger().info,
            tolerance=0.17, 
            duration=3.0
        )

          # 初始化 AlignmentChecker
        self.first_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,  # 传递日志记录函数
            threshold=0.15,
            time_window=2.0,
            check_frequency=10
        )
        self.second_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,  # 传递日志记录函数
            threshold=0.05,
            time_window=3.0,
            check_frequency=10
        )


    def target_position_callback(self, msg: Point):
        """Callback function for receiving target position."""
        self.target_position = msg
        self.get_logger().info(f"Received target position: {self.target_position}")

    def fly_to_position(self, x, y, z):
        """Fly to the specified position."""
        self.publish_position_setpoint(x, y, z)
        self.get_logger().info(f"Flying to position: x={x}, y={y}, z={z}")


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
        # self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

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

    
    def current_height_callback(self, msg):
        self.CurrentHeightFromCamera = msg.data
        self.get_logger().info(f"recevied:{self.CurrentHeightFromCamera}")

    
    def drop_payload(self):
        """Drop the payload (example logic)."""
                # 设置GPIO模式为BCM
        GPIO.setmode(GPIO.BCM)

        # 设置GPIO17为输出模式
        servo_pin = 17
        GPIO.setup(servo_pin, GPIO.OUT)

        # 设置PWM频率为50Hz
        pwm = GPIO.PWM(servo_pin, 50)
        pwm.start(0)

        def set_angle(angle):
            try:
            # 将角度转换为占空比
                duty = 2.5 + (angle / 18.0)
                pwm.ChangeDutyCycle(duty)
                time.sleep(1)
                pwm.ChangeDutyCycle(0)
            except Exception as e:
                print("e")

        try:
            set_angle(90)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            # pwm.stop()  # 确保停止PWM
            # GPIO.cleanup()  # 清理GPIO设置
            pass
        self.get_logger().info("Payload dropped.")

    def takeoff_relative(self, relative_height):
        """
        起飞到相对当前高度的指定高度
        :param relative_height: 相对高度 (比如上升 2 米)
        """
        # 初始化高度
        if self.initial_z is None:
            self.initial_z = self.vehicle_local_position.z
            self.initial_x = self.vehicle_local_position.x
            self.initial_y = self.vehicle_local_position.y
            self.init_yaw = self.vehicle_local_position.heading
            self.takeoff_target_height = float(self.initial_z + relative_height)
            self.get_logger().info(f"初始稳定位置记录为：x:{self.initial_x:.2f},y:{self.initial_y:.2f},z:{self.initial_z:.2f} 米, init_yaw:{self.init_yaw:.2f}")
            self.get_logger().info(f"target_height:{self.takeoff_target_height}")
        # 计算目标高度
        
        self.fly_to_position_FRD2NED(0.0, 0.0, self.takeoff_target_height)
        

    def takeoff_height_check(self, threshold=0.1):
        """
        检查是否到达相对目标高度
        :param threshold: 高度误差阈值
        :return: True 如果到达目标高度，否则 False
        """
        if self.takeoff_target_height is None:
            self.get_logger().warn("目标高度尚未设置！")
            return False

        current_height = self.vehicle_local_position.z
        height_error = abs(current_height - self.takeoff_target_height)

        self.get_logger().info(f"当前高度：{current_height:.2f} 米，目标高度：{self.takeoff_target_height:.2f} 米，高度误差：{height_error:.2f} 米")
        if height_error < threshold :
            self.is_AtTakeoffHeight=True



    def fly_forward(self,x):
        
        self.DropArea_x, self.DropArea_y = self.fly_to_position_FRD2NED(x, 0, self.takeoff_target_height)
        # self.get_logger().info("向前飞行")
        

    def fly_forward_check(self,threshold = 0.1):
        current_x = self.vehicle_local_position.x
        current_y = self.vehicle_local_position.y
        error = math.sqrt((current_x - self.DropArea_x)**2+(current_y - self.DropArea_y)**2)
        self.get_logger().info(f"--当前x：{current_x:.2f},y:{current_y:.2f}米，--目标x：{self.DropArea_x:.2f} 米，y:{self.DropArea_y:.2f},--误差：{error:.2f} 米")
        if error < threshold :
            self.is_AtDropArea=True
    
    def first_alignment_check(self, target_x1 , target_y1):
        aligned = self.first_alignment_checker.alignment_check(
    current_x=self.vehicle_local_position.x,
    current_y=self.vehicle_local_position.y,
    target_x=target_x1,
    target_y=target_y1
)       
        if aligned:
            self.first_alignment_checker.first_aligned = True
            self.get_logger().info("first对准完成！")


    def second_alignment_check(self, target_x1 , target_y1):
        aligned = self.second_alignment_checker.alignment_check(
    current_x=self.vehicle_local_position.x,
    current_y=self.vehicle_local_position.y,
    target_x=target_x1,
    target_y=target_y1)
       
        if aligned:
            self.second_aligned = True
            self.get_logger().info("second对准完成！")

    def fly_to_position_FRD2NED(self,x,y,z):
        '''
        通过旋转矩阵, 将FRD坐标系转换为NED坐标系。再根据初始误差增加平移矩阵。

        '''
        x_target = x*math.cos(self.init_yaw)-y*math.sin(self.init_yaw) + self.initial_x
        y_target = x*math.sin(self.init_yaw)+y*math.cos(self.init_yaw) + self.initial_y
        z_target = z
        # self.publish_position_setpoint(x_target, y_target, z_target)
        self.get_logger().info(f"Flying to FRDposition: x={x}, y={y}, z={z}")
        
        return x_target, y_target

    def coordinate_NED2FRD(self,x_NED,y_NED):
        '''
        将NED坐标转换为FRD坐标。
        '''
        x_FRD = (x_NED-self.initial_x)*math.cos(self.init_yaw)+(y_NED-self.initial_y)*math.sin(self.init_yaw)
        y_FRD = -(x_NED-self.initial_x)*math.sin(self.init_yaw)+(y_NED-self.initial_y)*math.cos(self.init_yaw)
        return x_FRD, y_FRD


    def adjust_to_target(self):
        # current_x, current_y =self.coordinate_NED2FRD(self.vehicle_local_position.x, self.vehicle_local_position.y)
        # self.get_logger().info(f"转换的机体坐标：x:{current_x:.2f}, y:{current_y:.2f}，北东地下的坐标，x:{self.vehicle_local_position.x:.2f},y:{self.vehicle_local_position.y:.2f}")

        """Adjust drone position towards the target."""
        if self.target_position:
            # Example logic: Adjust position incrementally based on target position

            current_x, current_y =self.coordinate_NED2FRD(self.vehicle_local_position.x, self.vehicle_local_position.y)



            target_x_FRD =  + self.target_position.y
            target_y_FRD =  - self.target_position.x
            # self.get_logger().info(f"相机坐标系下目标点：x:{self.target_position.x},y:{self.target_position.y}").
            self.get_logger().info(f"本地机身坐标系下目标点：x:{target_x_FRD:.2f},y:{target_y_FRD:.2f}")
            # self.get_logger().info(f"北东地坐标系下飞机坐标：x:{self.vehicle_local_position.x},y:{self.vehicle_local_position.y}").

            # First alignment
            # if not self.first_alignment_checker.first_aligned:
            #     self.get_logger().info("第一次对准操作")
            #     target_x_NED, target_y_NED = self.fly_to_position_FRD2NED(target_x_FRD, target_y_FRD, self.takeoff_target_height)  # Raise the altitude
            #     self.first_alignment_check(target_x_NED, target_y_NED)
            #     self.last_found_x_FRD = target_x_FRD
            #     self.last_found_y_FRD = target_y_FRD
            #     self.last_found_z_FRD = self.takeoff_target_height

            # # Second alignment, bringing it closer to the target
            # if self.first_alignment_checker.first_aligned and not self.second_alignment_checker.first_aligned:
            #     self.get_logger().info("第二次对准操作")
            #     target_x_NED2, target_y_NED2 = self.fly_to_position_FRD2NED(target_x_FRD, target_y_FRD, self.takeoff_target_height + 0.2)
            #     self.second_alignment_check(target_x_NED2, target_y_NED2)
            #     self.last_found_x_FRD = target_x_FRD
            #     self.last_found_y_FRD = target_y_FRD
            #     self.last_found_z_FRD = self.takeoff_target_height + 0.2

            self.target_position = None
            # drop
            if self.first_alignment_checker.first_aligned and self.second_alignment_checker.first_aligned:
                self.drop_payload()
                self.droping_x = self.vehicle_local_position.x
                self.droping_y = self.vehicle_local_position.y
                self.droping_x = self.vehicle_local_position.z
                self.is_FinishDrop = True
        else:
            if self.last_found_x_FRD and self.last_found_y_FRD and self.last_found_z_FRD:
                self.get_logger().info("使用上次记录")
                self.fly_to_position_FRD2NED(self.last_found_x_FRD, self.last_found_y_FRD, self.last_found_z_FRD)
            else:
                # self.get_logger().info("上次记录不存在")
                pass

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()
        # 发布 Offboard 模式切换指令,解锁飞机
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()  
            self.get_logger().info("try offboard")

        # if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            
            # if self.initial_z is None:
            #     self.initial_z = self.vehicle_local_position.z
            #     self.initial_x = self.vehicle_local_position.x
            #     self.initial_y = self.vehicle_local_position.y
            #     self.init_yaw = self.vehicle_local_position.heading
            #     # self.takeoff_target_height = float(self.initial_z + relative_height)
            #     self.get_logger().info(f"初始稳定位置记录为：x:{self.initial_x:.2f},y:{self.initial_y:.2f},z:{self.initial_z:.2f} 米, init_yaw:{self.init_yaw:.2f}")
        
        if not self.is_FinishDrop:
            self.drop_payload()
            
        
        if self.is_FinishDrop:

            self.fly_to_position(self.droping_x, self.droping_y, self.droping_z)



        # Continue flying after task completion
        if self.offboard_setpoint_counter < 30:
            self.offboard_setpoint_counter += 1


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


 
