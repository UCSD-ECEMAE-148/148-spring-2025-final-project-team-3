import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, Int32MultiArray, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import os
import numpy as np
import subprocess
import signal

NODE_NAME = 'lane_guidance_node'
CENTROID_TOPIC_NAME = '/centroid'
ACTUATOR_TOPIC_NAME = '/cmd_vel'
OBJ_CENTROID_TOPIC_NAME = '/object_detections/centroid'
DEPTH_TOPIC_NAME = '/object_detection/depth'  # Depth topic from object detection node

class PathPlanner(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.twist_publisher = self.create_publisher(Twist, ACTUATOR_TOPIC_NAME, 10)
        self.twist_cmd = Twist()
        self.bridge = CvBridge()
        
        # Check if OBJ_CENTROID_TOPIC_NAME is publishing data
        self.topic_to_use = self.check_topic_availability()
        
        # Subscribe to the chosen topic
        self.centroid_subscriber = self.create_subscription(
            Float32, 
            self.topic_to_use, 
            self.controller, 
            10
        )
        
        # Subscribe to depth topic for obstacle detection
        self.depth_subscriber = self.create_subscription(
            Image,
            DEPTH_TOPIC_NAME,
            self.depth_callback,
            10
        )
        
        self.get_logger().info(f'Subscribed to topic: {self.topic_to_use}')
        self.get_logger().info(f'Subscribed to depth topic: {DEPTH_TOPIC_NAME}')

        # Default actuator values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('Kp_steering', 0.2),
                ('Ki_steering', 0.0),
                ('Kd_steering', 0.1),
                ('error_threshold', 0.15),
                ('zero_throttle',0.0),
                ('max_throttle', 0.2),
                ('min_throttle', 0.1),
                ('max_right_steering', 1.0),
                ('max_left_steering', -1.0),
                ('depth_threshold', 0.2),  # meters - stop if obstacle closer than this
                ('roi_width_ratio', 0.3),  # width of center ROI as ratio of image width
                ('roi_height_ratio', 0.5),  # height of center ROI as ratio of image height
                ('sweep_duration', 5.0),   # seconds to run servo sweeper
                ('resume_delay', 1.2),     # seconds to wait after sweep before resuming
                ('turn_speed', 0.5),       # angular velocity for turns (rad/s)
                ('forward_speed', 0.15),   # linear velocity for forward movement
                ('reverse_speed', -0.15),  # linear velocity for reverse movement
                ('turn_angle_deg', 80.0),  # turn angle in degrees
                ('forward_distance', 0.4), # forward distance in meters
                ('reverse_distance', 0.4), # reverse distance in meters  
                ('final_forward_distance', 0.34), # final forward distance in meters
            ])
        self.Kp = self.get_parameter('Kp_steering').value # between [0,1]
        self.Ki = self.get_parameter('Ki_steering').value # between [0,1]
        self.Kd = self.get_parameter('Kd_steering').value # between [0,1]
        self.error_threshold = self.get_parameter('error_threshold').value # between [0,1]
        self.zero_throttle = self.get_parameter('zero_throttle').value # between [-1,1] but should be around 0
        self.max_throttle = self.get_parameter('max_throttle').value # between [-1,1]
        self.min_throttle = self.get_parameter('min_throttle').value # between [-1,1]
        self.max_right_steering = self.get_parameter('max_right_steering').value # between [-1,1]
        self.max_left_steering = self.get_parameter('max_left_steering').value # between [-1,1]
        
        # Obstacle detection parameters
        self.depth_threshold = self.get_parameter('depth_threshold').value
        self.roi_width_ratio = self.get_parameter('roi_width_ratio').value
        self.roi_height_ratio = self.get_parameter('roi_height_ratio').value
        self.sweep_duration = self.get_parameter('sweep_duration').value
        self.resume_delay = self.get_parameter('resume_delay').value
        
        # Avoidance maneuver parameters
        self.turn_speed = self.get_parameter('turn_speed').value
        self.forward_speed = self.get_parameter('forward_speed').value
        self.reverse_speed = self.get_parameter('reverse_speed').value
        self.turn_angle_deg = self.get_parameter('turn_angle_deg').value
        self.forward_distance = self.get_parameter('forward_distance').value
        self.reverse_distance = self.get_parameter('reverse_distance').value
        self.final_forward_distance = self.get_parameter('final_forward_distance').value
        
        # Calculate time durations for each maneuver step
        self.turn_duration = abs(np.radians(self.turn_angle_deg) / self.turn_speed)
        self.forward_duration = abs(self.forward_distance / self.forward_speed)
        self.reverse_duration = abs(self.reverse_distance / self.reverse_speed)
        self.final_forward_duration = abs(self.final_forward_distance / self.forward_speed)

        # initializing PID control
        self.Ts = float(1/20)
        self.ek = 0 # current error
        self.ek_1 = 0 # previous error
        self.proportional_error = 0 # proportional error term for steering
        self.derivative_error = 0 # derivative error term for steering
        self.integral_error = 0 # integral error term for steering
        self.integral_max = 1E-8
        
        # Obstacle detection variables
        self.obstacle_too_close = False  # State: is there currently an obstacle within danger threshold
        self.is_sweeping = False
        self.is_avoiding = False  # State: currently executing avoidance maneuver
        self.avoidance_step = 0   # Current step in avoidance sequence
        self.avoidance_start_time = None
        self.sweep_process = None
        self.last_depth_frame = None
        self.sweep_start_time = None
        
        # Timer for checking sweep completion and avoidance maneuvers
        self.sweep_timer = self.create_timer(0.1, self.check_sweep_and_avoidance_status)
        
        self.get_logger().info(
            f'\nKp_steering: {self.Kp}'
            f'\nKi_steering: {self.Ki}'
            f'\nKd_steering: {self.Kd}'
            f'\nerror_threshold: {self.error_threshold}'
            f'\nzero_throttle: {self.zero_throttle}'
            f'\nmax_throttle: {self.max_throttle}'
            f'\nmin_throttle: {self.min_throttle}'
            f'\nmax_right_steering: {self.max_right_steering}'
            f'\nmax_left_steering: {self.max_left_steering}'
            f'\ndepth_threshold: {self.depth_threshold}'
            f'\nroi_width_ratio: {self.roi_width_ratio}'
            f'\nroi_height_ratio: {self.roi_height_ratio}'
            f'\nsweep_duration: {self.sweep_duration}'
            f'\nfinal_forward_distance: {self.final_forward_distance}'
            f'\nturn_duration: {self.turn_duration:.2f}s'
            f'\nforward_duration: {self.forward_duration:.2f}s'
            f'\nreverse_duration: {self.reverse_duration:.2f}s'
            f'\nfinal_forward_duration: {self.final_forward_duration:.2f}s'
        )

    def check_topic_availability(self):
        """Check if OBJ_CENTROID_TOPIC_NAME is available and publishing data"""
        self.get_logger().info(f'Checking topic availability...')
        
        # Get list of available topics
        topic_names_and_types = dict(self.get_topic_names_and_types())
        
        # Check if the object centroid topic exists
        if OBJ_CENTROID_TOPIC_NAME not in topic_names_and_types:
            self.get_logger().info(f'Topic {OBJ_CENTROID_TOPIC_NAME} not found. Using {CENTROID_TOPIC_NAME}')
            return CENTROID_TOPIC_NAME
        
        # Check if the topic has the correct message type
        if topic_names_and_types[OBJ_CENTROID_TOPIC_NAME][0] != 'std_msgs/msg/Float32':
            self.get_logger().warn(f'Topic {OBJ_CENTROID_TOPIC_NAME} has wrong message type. Using {CENTROID_TOPIC_NAME}')
            return CENTROID_TOPIC_NAME
        
        # Try to receive a message within a timeout period
        self.get_logger().info(f'Topic {OBJ_CENTROID_TOPIC_NAME} found. Testing for active data...')
        
        # Create a temporary subscription to test for data
        self.test_data_received = False
        test_subscription = self.create_subscription(
            Float32,
            OBJ_CENTROID_TOPIC_NAME,
            self._test_callback,
            10
        )
        
        # Wait for up to 3 seconds for a message
        start_time = time.time()
        timeout = 3.0
        
        while not self.test_data_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Destroy the test subscription
        self.destroy_subscription(test_subscription)
        
        if self.test_data_received:
            self.get_logger().info(f'Data detected on {OBJ_CENTROID_TOPIC_NAME}. Using this topic.')
            return OBJ_CENTROID_TOPIC_NAME
        else:
            self.get_logger().info(f'No data detected on {OBJ_CENTROID_TOPIC_NAME} within {timeout}s. Using {CENTROID_TOPIC_NAME}')
            return CENTROID_TOPIC_NAME

    def _test_callback(self, msg):
        """Temporary callback to test if data is being received"""
        self.test_data_received = True

    def depth_callback(self, msg):
        """Process depth image for obstacle detection"""
        try:
            # Convert ROS image message to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.last_depth_frame = depth_image
            
            # Check for obstacles in the forward path
            self.check_for_obstacles(depth_image)
            
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def check_for_obstacles(self, depth_image):
        """Check if there's an obstacle too close in the forward path"""
        if depth_image is None:
            return
            
        height, width = depth_image.shape
        
        # Define region of interest (center of the image)
        roi_width = int(width * self.roi_width_ratio)
        roi_height = int(height * self.roi_height_ratio)
        
        # Calculate ROI boundaries (center region)
        x_start = (width - roi_width) // 2
        x_end = x_start + roi_width
        y_start = (height - roi_height) // 2
        y_end = y_start + roi_height
        
        # Extract ROI
        roi = depth_image[y_start:y_end, x_start:x_end]
        
        # Filter out invalid depth values (typically 0 or nan)
        valid_depths = roi[(roi > 0) & (~np.isnan(roi)) & (~np.isinf(roi))]
        
        if len(valid_depths) == 0:
            return
            
        # Find minimum depth in ROI
        min_depth = np.min(valid_depths)
        
        # Only care about obstacles that are TOO CLOSE (under threshold)
        obstacle_too_close = min_depth < self.depth_threshold
        
        # Log depth information periodically
        if hasattr(self, '_last_depth_log_time'):
            if time.time() - self._last_depth_log_time > 2.0:  # Log every 2 seconds
                self.get_logger().info(f'Min depth in ROI: {min_depth:.2f}m, Threshold: {self.depth_threshold}m')
                self._last_depth_log_time = time.time()
        else:
            self._last_depth_log_time = time.time()
        
        # Handle obstacle too close state changes
        if obstacle_too_close and not self.obstacle_too_close and not self.is_sweeping:
            self.get_logger().warn(f'Obstacle too close at {min_depth:.2f}m! Stopping and initiating sweep.')
            self.obstacle_too_close = True
            self.start_servo_sweep()
        elif not obstacle_too_close and self.obstacle_too_close and not self.is_sweeping:
            self.get_logger().info('Obstacle moved away or path clear, resuming normal operation.')
            self.obstacle_too_close = False

    def start_servo_sweep(self):
        """Start the servo sweeper node"""
        try:
            self.get_logger().info('Starting servo sweeper...')
            self.is_sweeping = True
            self.sweep_start_time = time.time()
            
            # Launch servo sweeper node (adjust command as needed for your setup)
            # This assumes you have a servo_sweeper ROS2 node or launch file
            self.sweep_process = subprocess.Popen(
                ['ros2', 'run', 'final_pkg', 'servo_sweeper'],  
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            self.get_logger().info(f'Servo sweeper started with PID: {self.sweep_process.pid}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to start servo sweeper: {str(e)}')
            self.is_sweeping = False

    def stop_servo_sweep(self):
        """Stop the servo sweeper node"""
        if self.sweep_process is not None:
            try:
                self.get_logger().info('Stopping servo sweeper...')
                self.sweep_process.terminate()
                
                # Wait for process to terminate gracefully
                try:
                    self.sweep_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.get_logger().warn('Servo sweeper did not terminate gracefully, killing...')
                    self.sweep_process.kill()
                    self.sweep_process.wait()
                
                self.sweep_process = None
                self.get_logger().info('Servo sweeper stopped.')
                
            except Exception as e:
                self.get_logger().error(f'Error stopping servo sweeper: {str(e)}')

    def check_sweep_and_avoidance_status(self):
        """Timer callback to check if sweep is complete and handle avoidance maneuvers"""
        
        # Check if sweep is complete
        if self.is_sweeping and self.sweep_start_time is not None:
            elapsed_time = time.time() - self.sweep_start_time
            
            if elapsed_time >= self.sweep_duration:
                self.get_logger().info('Sweep duration completed, stopping sweep...')
                self.stop_servo_sweep()
                
                # Start avoidance maneuver after sweep
                self.get_logger().info('Starting avoidance maneuver sequence...')
                self.start_avoidance_maneuver()
                
                self.is_sweeping = False
                self.sweep_start_time = None
        
        # Handle avoidance maneuver steps
        elif self.is_avoiding and self.avoidance_start_time is not None:
            self.execute_avoidance_step()

    def start_avoidance_maneuver(self):
        """Start the avoidance maneuver sequence"""
        self.is_avoiding = True
        self.avoidance_step = 1  # Start with step 1: turn right
        self.avoidance_start_time = time.time()
        self.get_logger().info('Step 1: Turning right 80 degrees...')

    def execute_avoidance_step(self):
        """Execute the current step of the avoidance maneuver"""
        elapsed_time = time.time() - self.avoidance_start_time
        
        if self.avoidance_step == 1:  # Turn right 80 degrees
            if elapsed_time < self.turn_duration:
                self.twist_cmd.linear.x = 0.0
                self.twist_cmd.angular.z = -self.turn_speed  # Negative for right turn
                self.twist_publisher.publish(self.twist_cmd)
            else:
                self.next_avoidance_step("Step 2: Moving forward 0.4m...")
                
        elif self.avoidance_step == 2:  # Move forward 0.4m
            if elapsed_time < self.forward_duration:
                self.twist_cmd.linear.x = self.forward_speed
                self.twist_cmd.angular.z = 0.0
                self.twist_publisher.publish(self.twist_cmd)
            else:
                self.next_avoidance_step("Step 3: Moving backward 0.4m...")
                
        elif self.avoidance_step == 3:  # Move backward 0.4m
            if elapsed_time < self.reverse_duration:
                self.twist_cmd.linear.x = self.reverse_speed
                self.twist_cmd.angular.z = 0.0
                self.twist_publisher.publish(self.twist_cmd)
            else:
                self.next_avoidance_step("Step 4: Turning left 80 degrees...")
                
        elif self.avoidance_step == 4:  # Turn left 80 degrees
            if elapsed_time < self.turn_duration:
                self.twist_cmd.linear.x = 0.0
                self.twist_cmd.angular.z = self.turn_speed  # Positive for left turn
                self.twist_publisher.publish(self.twist_cmd)
            else:
                self.next_avoidance_step("Step 5: Moving forward 0.34m...")
                
        elif self.avoidance_step == 5:  # Move forward 0.34m
            if elapsed_time < self.final_forward_duration:
                self.twist_cmd.linear.x = self.forward_speed
                self.twist_cmd.angular.z = 0.0
                self.twist_publisher.publish(self.twist_cmd)
            else:
                self.finish_avoidance_maneuver()

    def next_avoidance_step(self, log_message):
        """Move to the next step in avoidance maneuver"""
        self.avoidance_step += 1
        self.avoidance_start_time = time.time()
        self.get_logger().info(log_message)
        
        # Stop motors briefly between steps
        self.twist_cmd.linear.x = 0.0
        self.twist_cmd.angular.z = 0.0
        self.twist_publisher.publish(self.twist_cmd)

    def finish_avoidance_maneuver(self):
        """Complete the avoidance maneuver and resume normal operation"""
        self.get_logger().info('Avoidance maneuver completed. Resuming normal guidance...')
        
        # Stop the robot
        self.twist_cmd.linear.x = 0.0
        self.twist_cmd.angular.z = 0.0
        self.twist_publisher.publish(self.twist_cmd)
        
        # Reset avoidance state
        self.is_avoiding = False
        self.avoidance_step = 0
        self.avoidance_start_time = None
        self.obstacle_too_close = False  # Clear obstacle state
        
        # Wait briefly before resuming normal operation
        time.sleep(self.resume_delay)

    def controller(self, data):
        # Only stop if obstacle is TOO CLOSE, currently sweeping, or executing avoidance maneuver
        # Continue normal driving even when objects are detected at safe distances
        if self.obstacle_too_close or self.is_sweeping or self.is_avoiding:
            # If we're avoiding, the avoidance logic handles motor control
            if not self.is_avoiding:
                self.twist_cmd.linear.x = self.zero_throttle
                self.twist_cmd.angular.z = 0.0
                self.twist_publisher.publish(self.twist_cmd)
            return
        
        # Normal PID control - drives normally regardless of distant object detection
        # setting up PID control
        self.ek = data.data

        # Throttle gain scheduling (function of error)
        self.inf_throttle = self.min_throttle - (self.min_throttle - self.max_throttle) / (1 - self.error_threshold)
        throttle_float_raw = ((self.min_throttle - self.max_throttle)  / (1 - self.error_threshold)) * abs(self.ek) + self.inf_throttle
        throttle_float = self.clamp(throttle_float_raw, self.max_throttle, self.min_throttle)

        # Steering PID terms
        self.proportional_error = self.Kp * self.ek
        self.derivative_error = self.Kd * (self.ek - self.ek_1) / self.Ts
        self.integral_error += self.Ki * self.ek * self.Ts
        self.integral_error = self.clamp(self.integral_error, self.integral_max)
        steering_float_raw = self.proportional_error + self.derivative_error + self.integral_error
        steering_float = self.clamp(steering_float_raw, self.max_right_steering, self.max_left_steering)

        # Publish values
        try:
            # publish control signals
            self.twist_cmd.angular.z = steering_float
            self.twist_cmd.linear.x = throttle_float
            self.twist_publisher.publish(self.twist_cmd)

            # shift current time and error values to previous values
            self.ek_1 = self.ek

        except KeyboardInterrupt:
            self.twist_cmd.linear.x = self.zero_throttle
            self.twist_publisher.publish(self.twist_cmd)

    def clamp(self, value, upper_bound, lower_bound=None):
        if lower_bound==None:
            lower_bound = -upper_bound # making lower bound symmetric about zero
        if value < lower_bound:
            value_c = lower_bound
        elif value > upper_bound:
            value_c = upper_bound
        else:
            value_c = value
        return value_c 

    def cleanup(self):
        """Clean up resources"""
        if self.is_sweeping:
            self.stop_servo_sweep()
        
        # Stop the car
        self.twist_cmd.linear.x = self.zero_throttle
        self.twist_cmd.angular.z = 0.0
        self.twist_publisher.publish(self.twist_cmd)
        
        # Reset all states
        self.is_avoiding = False
        self.obstacle_too_close = False


def main(args=None):
    rclpy.init(args=args)
    path_planner_publisher = PathPlanner()
    try:
        rclpy.spin(path_planner_publisher)
        path_planner_publisher.cleanup()
        path_planner_publisher.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        path_planner_publisher.get_logger().info(f'Shutting down {NODE_NAME}...')
        path_planner_publisher.cleanup()
        time.sleep(1)
        path_planner_publisher.destroy_node()
        rclpy.shutdown()
        path_planner_publisher.get_logger().info(f'{NODE_NAME} shut down successfully.')


if __name__ == '__main__':
    main()