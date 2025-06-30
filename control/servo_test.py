from rclpy.node import Node
import rclpy
from control.ServoControl import ServoControl
class ServoTest(Node):
    def __init__(self):
        super().__init__('servo_test_node')
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.servo_control = ServoControl()
        self.count=0


    def timer_callback(self):
        self.count += 1
        if self.count % 2 == 0:
            self.servo_control.open_servo(1.0,1.0)
        else:
            self.servo_control.open_servo(-1.0,-1.0)    

def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    node = ServoTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)