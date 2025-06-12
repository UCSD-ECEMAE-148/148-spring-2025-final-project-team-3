#!/usr/bin/env python
# ROS 2 imports
import rclpy  # Core ROS 2 Python library
from rclpy.node import Node  # Node class for creating ROS 2 nodes

# Message types we'll publish
from sensor_msgs.msg import Image  # Standard ROS Image message
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32  # Added Float32 for centroid error

# OpenCV bridge for converting images between ROS and OpenCV
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# Roboflow OAK-D Lite interface
from roboflowoak import RoboflowOak


class ObjDetectionNode(Node):
    def __init__(self):
        # Initialize the ROS 2 node with the name 'object_detection_node'
        super().__init__('object_detection_node')

        # === ROS 2 PUBLISHERS ===
        # Publisher for coordinates of detected garbage objects
        #self.detection_pub = self.create_publisher(
        #    PointStamped,
        #    '/object_detections/centroid',
        #    10
        #)

        # NEW: Publisher for centroid error (steering value) - matches lane detection format
        self.centroid_error_publisher = self.create_publisher(
            Float32,
            '/object_detections/centroid',
            10
        )

        # Publisher for the annotated image (detections drawn on it)
        self.image_pub = self.create_publisher(
            Image,
            '/object_detections/image',
            10
        )

        # Publisher for the width information (replaces depth)
        self.width_pub = self.create_publisher(
            Float32,
            '/object_detections/depth',  # Keep same topic name for compatibility
            10
        )

        # Publisher for detection flag (true if any garbage detected)
        self.detection_flag_pub = self.create_publisher(
            Bool,
            '/object_detections/flag',
            10
        )

        # === CV BRIDGE ===
        # Used to convert OpenCV images to ROS Image messages
        self.bridge = CvBridge()

        # === NEW VARIABLES for centroid calculation (from lane detection) ===
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_centerline', 0.5),  # Camera center reference (0.5 = middle)
                ('error_threshold', 0.15),   # Error threshold for steering decisions
                ('target_selection_method', 'closest'),  # 'closest', 'average', 'largest'
                ('width_threshold', 200.0),  # Width threshold for obstacle detection (pixels)
                ('min_width_for_detection', 0.0)  # Minimum width to consider valid detection
            ])
        
        self.camera_centerline = self.get_parameter('camera_centerline').value
        self.error_threshold = self.get_parameter('error_threshold').value
        self.target_selection_method = self.get_parameter('target_selection_method').value
        self.width_threshold = self.get_parameter('width_threshold').value
        self.min_width_for_detection = self.get_parameter('min_width_for_detection').value
        
        # Image dimensions (will be set when first frame arrives)
        self.image_width = 0
        self.image_height = 0
        self.camera_init = False
        
        # Centroid error message
        self.centroid_error = Float32()
        
        # Width message (replaces depth)
        self.width_msg = Float32()

        # === ROBOFLOWOAK INITIALIZATION ===
        self.rf = RoboflowOak(
            model="garbage-dxrv3",    # Your garbage detection model
            confidence=0.60,                # Confidence threshold
            overlap=0.01,                    # NMS overlap threshold
            version="3", 
            api_key="Tv55RvxSLtK3OR0qU9Hb", # Your Roboflow API key
            rgb=True,                       # Whether to use RGB stream
            depth=False,                    # Disable depth stream since we're not using it
            device=None,                    # Auto device selection
            blocking=True                   # Blocking detection call
        )

        # === DETECTION PARAMETERS ===
        self.target_class_name = "garbage"  # The class name we're looking for
        self.target_class = 0            # Assuming garbage has class_id 0
        
        # === CREATE A TIMER CALLBACK ===
        # Calls self.run_model every 0.1 seconds (10Hz)
        self.create_timer(0.1, self.run_model)

        self.get_logger().info(f"Object Detection Node initialized - looking for garbage")
        self.get_logger().info(f"Width threshold: {self.width_threshold} pixels")
        self.get_logger().info(f"Minimum width for detection: {self.min_width_for_detection} pixels")

    def run_model(self):
        """
        Main detection loop - runs inference and publishes results
        """
        t0 = time.time()
        
        try:
            # Run detection on OAK-D (depth=None since we disabled it)
            result, frame, raw_frame, depth = self.rf.detect()
            
            if result is None or frame is None:
                self.get_logger().warn("No result or frame received from RoboflowOAK")
                return
                
            predictions = result["predictions"]
            
            # Publish image first so that image topic gets frame regardless of detections
            self.publish_image(frame)
            
            # Initialize camera dimensions on first frame
            if not self.camera_init and frame is not None:
                self.image_height, self.image_width = frame.shape[:2]
                self.camera_init = True
                self.get_logger().info(f'Camera initialized: {self.image_width}x{self.image_height}')

            # Filter predictions by minimum width
            valid_predictions = []
            for pred in predictions:
                if pred.width >= self.min_width_for_detection:
                    valid_predictions.append(pred)
                    
            # Log detection info
            self.get_logger().info(f'Total detections: {len(predictions)}, Valid detections: {len(valid_predictions)}')
            
            # Log width information for all valid predictions
            if valid_predictions:
                widths = [pred.width for pred in valid_predictions]
                self.get_logger().info(f'Detection widths: {widths}')
            
            # Since we only have one class, we can just use all valid predictions
            garbage_predictions = valid_predictions
             
            # Publish detection flag
            detection_flag_msg = Bool()
            detection_flag_msg.data = len(garbage_predictions) > 0
            self.detection_flag_pub.publish(detection_flag_msg)

            # Calculate and publish centroid error (steering command)
            steering_error = 0.0
            if len(garbage_predictions) > 0:
                steering_error = self.calculate_centroid_error(garbage_predictions)
                self.centroid_error.data = steering_error
                self.centroid_error_publisher.publish(self.centroid_error)

            # Calculate and publish width information (replaces depth)
            if len(garbage_predictions) > 0:
                width_value = self.calculate_representative_width(garbage_predictions)
                self.width_msg.data = width_value
                self.width_pub.publish(self.width_msg)

            # Process each garbage detection for visualization
            for pred in garbage_predictions:
                # Draw bounding box on frame
                self.draw_detection(frame, pred)

            # Draw steering visualization
            if len(garbage_predictions) > 0:
                self.draw_steering_visualization(frame, garbage_predictions)

            # Display frame for debugging
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            
            # Log performance
            t = time.time() - t0
            fps = 1/t if t > 0 else 0
            self.get_logger().info(f'FPS: {fps:.2f}, Valid garbage detections: {len(garbage_predictions)}, Steering error: {steering_error:.3f}')

        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')

    def calculate_representative_width(self, predictions):
        """
        Calculate representative width for obstacle detection
        Uses the largest width among all detections as the representative value
        """
        if not predictions:
            return 0.0
            
        # Get the largest width (closest/most significant object)
        max_width = max(pred.width for pred in predictions)
        
        # Log width information
        self.get_logger().info(f'Representative width: {max_width:.1f} pixels (threshold: {self.width_threshold:.1f})')
        
        return float(max_width)

    def calculate_centroid_error(self, predictions):
        """
        Calculate steering error using lane detection method
        Adapted from locate_centroid function in lane detection script
        """
        if not self.camera_init or len(predictions) == 0:
            return 0.0

        # Camera center line (reference point)
        cam_center_line_x = int(self.image_width * self.camera_centerline)
        
        # Extract centroid positions from predictions
        cx_list = []
        distances = []  # for distance-based selection
        
        for pred in predictions:
            cx = int(pred.x)  # Center x coordinate
            cx_list.append(cx)
            
            # Calculate distance from camera center for sorting
            distance_from_center = abs(cx - cam_center_line_x)
            distances.append(distance_from_center)

        # Calculate steering error based on selection method
        if len(cx_list) == 1:
            # Single detection case
            error_x = float((cx_list[0] - cam_center_line_x) / cam_center_line_x)
            self.get_logger().info(f"Single target detected: steering error = {error_x:.3f}")
            
        elif len(cx_list) > 1:
            # Multiple detections - apply lane detection logic
            error_list = []
            
            # Calculate errors for all detected objects
            for cx_pos in cx_list:
                error = float((cx_pos - cam_center_line_x) / cam_center_line_x)
                error_list.append(error)
            
            if self.target_selection_method == 'average':
                # Average error method (for straight path)
                avg_error = sum(error_list) / len(error_list)
                error_x = avg_error
                self.get_logger().info(f"Multiple targets - average method: steering error = {error_x:.3f}")
                
            elif self.target_selection_method == 'closest':
                # Closest to center method
                min_error_index = distances.index(min(distances))
                error_x = error_list[min_error_index]
                self.get_logger().info(f"Multiple targets - closest method: steering error = {error_x:.3f}")
                
            else:  # 'minimum_error'
                # Minimum absolute error method (adapted from lane detection curve logic)
                filtered_errors = []
                for error in error_list:
                    if abs(error) >= self.error_threshold:  # Only consider significant errors
                        filtered_errors.append(error)
                
                if filtered_errors:
                    error_x = min(filtered_errors, key=abs)
                else:
                    error_x = min(error_list, key=abs)  # Fallback to minimum error
                    
                self.get_logger().info(f"Multiple targets - minimum error method: steering error = {error_x:.3f}")
        else:
            error_x = 0.0
            self.get_logger().info("No targets detected")
            
        return error_x

    def draw_steering_visualization(self, frame, predictions):
        """
        Draw steering visualization similar to lane detection
        """
        if not self.camera_init:
            return
            
        # Camera center line
        cam_center_line_x = int(self.image_width * self.camera_centerline)
        start_point = (cam_center_line_x, 0)
        end_point = (cam_center_line_x, self.image_height)
        
        # Error threshold lines
        thresh_offset = int(self.error_threshold * self.image_width / 2)
        start_point_thresh_pos = (cam_center_line_x - thresh_offset, 0)
        end_point_thresh_pos = (cam_center_line_x - thresh_offset, self.image_height)
        start_point_thresh_neg = (cam_center_line_x + thresh_offset, 0)
        end_point_thresh_neg = (cam_center_line_x + thresh_offset, self.image_height)
        
        # Draw reference lines
        cv2.line(frame, start_point, end_point, (0, 255, 0), 4)  # Center line (green)
        cv2.line(frame, start_point_thresh_pos, end_point_thresh_pos, (0, 0, 255), 2)  # Threshold lines (red)
        cv2.line(frame, start_point_thresh_neg, end_point_thresh_neg, (0, 0, 255), 2)
        
        # Draw steering error line to target
        if len(predictions) > 0:
            # Find the target we're tracking based on selection method
            if self.target_selection_method == 'closest':
                distances = [abs(int(pred.x) - cam_center_line_x) for pred in predictions]
                target_index = distances.index(min(distances))
            else:
                target_index = 0  # Default to first detection
                
            target_pred = predictions[target_index]
            target_x = int(target_pred.x)
            target_y = int(target_pred.y)
            
            # Draw error line
            start_point_error = (cam_center_line_x, target_y)
            cv2.line(frame, start_point_error, (target_x, target_y), (0, 0, 255), 4)
            
            # Highlight the target being tracked
            cv2.circle(frame, (target_x, target_y), 10, (255, 0, 0), -1)

    def draw_detection(self, frame, prediction):
        """
        Draw bounding box and labels on the frame
        """
        # Calculate bounding box corners
        x1 = int(prediction.x - prediction.width // 2)
        y1 = int(prediction.y - prediction.height // 2)
        x2 = int(prediction.x + prediction.width // 2)
        y2 = int(prediction.y + prediction.height // 2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text with width information
        label = f"{self.target_class_name} {prediction.confidence:.2f} W:{prediction.width:.0f}px"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def publish_image(self, frame):
        """
        Publish annotated image
        """
        try:
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = 'oakd_camera_frame'
            self.image_pub.publish(image_msg)
        except Exception as e:
            self.get_logger().error(f'Image publishing error: {str(e)}')


def main(args=None):
    """
    Main entry point for the ROS 2 node.
    """
    # Initialize ROS 2 Python
    rclpy.init(args=args)

    try:
        # Create an instance of the node
        node = ObjDetectionNode()

        # Keep the node alive and spinning
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Cleanup when shutting down
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()