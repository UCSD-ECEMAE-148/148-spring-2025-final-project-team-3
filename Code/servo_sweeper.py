import rclpy
from rclpy.node import Node
import time
from adafruit_pca9685 import PCA9685
import board
import busio

class ServoSweeper(Node):
    def __init__(self):
        super().__init__('servo_sweeper')
        self.get_logger().info('Starting Servo Sweeper Node')

        # Initialize I2C and PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50  # Standard for servos

        self.channel = 0  # Channel on PCA9685
        self.min_us = 500
        self.max_us = 2500
        self.pwm_period = 1000000 / self.pca.frequency  # us

        self.move_servo()

    def set_servo_angle(self, angle):
        pulse = self.min_us + (angle / 180.0) * (self.max_us - self.min_us)
        duty_cycle = int((pulse / self.pwm_period) * 0xFFFF)
        self.pca.channels[self.channel].duty_cycle = duty_cycle
        self.get_logger().info(f'Set angle to {angle}°')

    def disable_servo(self):
        # Completely stop PWM output on the channel
        self.pca.channels[self.channel].duty_cycle = 0
        self.get_logger().info('Servo disabled')

    def move_servo(self):
        # Move to 0°
        self.set_servo_angle(0)
        time.sleep(1)

        # Move to 100°
        self.set_servo_angle(100)
        time.sleep(1)

        # Disable servo and shutdown node
        self.disable_servo()
        self.get_logger().info('Motion complete. Shutting down node.')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ServoSweeper()
    rclpy.spin(node)  # This will exit as soon as rclpy.shutdown() is called

if __name__ == '__main__':
    main()
