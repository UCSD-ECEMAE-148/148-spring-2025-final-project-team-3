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

        # Publisher for the depth map
        self.depth_pub = self.create_publisher(
            Float32,
            '/object_detections/depth',
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
                ('target_selection_method', 'closest')  # 'closest', 'average', 'largest'
            ])
        
        self.camera_centerline = self.get_parameter('camera_centerline').value
        self.error_threshold = self.get_parameter('error_threshold').value
        self.target_selection_method = self.get_parameter('target_selection_method').value
        
        # Image dimensions (will be set when first frame arrives)
        self.image_width = 0
        self.image_height = 0
        self.camera_init = False
        
        # Centroid error message
        self.centroid_error = Float32()

        # === ROBOFLOWOAK INITIALIZATION ===
        self.rf = RoboflowOak(
            model="ece-148-final-project",    # Your garbage detection model
            confidence=0.79,                # Confidence threshold
            overlap=0.01,                    # NMS overlap threshold
            version="1", 
            api_key="Tv55RvxSLtK3OR0qU9Hb", # Your Roboflow API key
            rgb=True,                       # Whether to use RGB stream
            depth=True,                     # Whether to use depth stream
            device=None,                    # Auto device selection
            blocking=True                   # Blocking detection call
        )

        # === DETECTION PARAMETERS ===
        self.target_class_name = "garbage"  # The class name we're looking for
        self.target_class = 0            # Assuming garbage has class_id 0
        
        # === CREATE A TIMER CALLBACK ===
        # Calls self.run_model every 0.1 seconds (10Hz)
        self.create_timer(0.1, self.run_model)

        self.get_logger().info("Object Detection Node initialized - looking for garbage")

    def run_model(self):
        """
        Main detection loop - runs inference and publishes results
        """
        t0 = time.time()
        # Check if the RoboflowOAK instance is ready
        
        try:
            

             # Run detection on OAK-D
            result, frame, raw_frame, depth = self.rf.detect()
            self.publish_image(frame) #Image should be published first so that image topic gets unannoted image regardless of whether there are detections or not
            if result is None or frame is None:
                self.get_logger().warn("No result or frame received from RoboflowOAK")
                return
            predictions = result["predictions"]
            
            self.get_logger().info(f'Detections: {len(predictions)}')
            x = [pred.x for pred in predictions]
            y = [pred.y for pred in predictions]
            self.get_logger().info(f'Predictions: {x}, {y}')
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error during detection: {str(e)}')
            return

        try:
            # Run detection on OAK-D
            result, frame, raw_frame, depth = self.rf.detect()
            predictions = result["predictions"]

            # Initialize camera dimensions on first frame
            if not self.camera_init and frame is not None:
                self.image_height, self.image_width = frame.shape[:2]
                self.camera_init = True
                self.get_logger().info(f'Camera initialized: {self.image_width}x{self.image_height}')

            #instead of using class_id, we can check if any predictions were made since there is only one class
            if not predictions:
                self.get_logger().info("No predictions made")
                return
             
            #Since we only have one class, we can just use all predictions (this line might be outdated but I don't want to change the naming later)
            garbage_predictions = [pred for pred in predictions]       
            # Publish detection flag  (keeping this just in case we want to use it later)
            detection_flag_msg = Bool()
            detection_flag_msg.data = len(garbage_predictions) > 0
            self.detection_flag_pub.publish(detection_flag_msg)

            # Calculate and publish centroid error (steering command)
            if len(garbage_predictions) > 0:
                steering_error = self.calculate_centroid_error(garbage_predictions)
                self.centroid_error.data = steering_error
                self.centroid_error_publisher.publish(self.centroid_error)

            # Process each garbage detection
            for pred in garbage_predictions:
                # Publish coordinates
                #self.publish_coordinates(pred, depth)
                
                # Draw bounding box on frame
                self.draw_detection(frame, pred, depth)

            # Draw steering visualization
            if len(garbage_predictions) > 0:
                self.draw_steering_visualization(frame, garbage_predictions)

            # Publish annotated image
            
            
            # Publish depth image
            if depth is not None:
               for pred in predictions:
                    x = int(pred.x)
                    y = int(pred.y)

                    if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                        object_depth = depth[y, x]  # Access depth at center of bounding box
                        print(f"{pred.class_name} detected at depth: {object_depth}")
                    else:
                        print("Object center is out of bounds")

            # Log performance
            t = time.time() - t0
            fps = 1/t if t > 0 else 0
            self.get_logger().info(f'FPS: {fps:.2f}, Garbage detections: {len(garbage_predictions)}, Steering error: {steering_error if len(garbage_predictions) > 0 else 0.0:.3f}')

        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')

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
        distances = []  # for depth-based selection
        
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

    def publish_coordinates(self, prediction, depth_frame):
        """
        Publish the coordinates of a detected object
        """
        coord_msg = PointStamped()
        coord_msg.header.stamp = self.get_clock().now().to_msg()
        coord_msg.header.frame_id = 'oakd_camera_frame'
        
        # Center coordinates of bounding box
        coord_msg.point.x = float(prediction.x)
        coord_msg.point.y = float(prediction.y)
        
        # Get depth value at detection center
        if depth_frame is not None:
            try:
                # Ensure coordinates are within depth frame bounds
                y_coord = max(0, min(int(prediction.y), depth_frame.shape[0] - 1))
                x_coord = max(0, min(int(prediction.x), depth_frame.shape[1] - 1))
                
                depth_val = float(depth_frame[y_coord, x_coord])
                coord_msg.point.z = depth_val
                
            except (IndexError, ValueError) as e:
                self.get_logger().warn(f'Depth extraction error: {e}')
                coord_msg.point.z = 0.0
        else:
            coord_msg.point.z = 0.0

        self.detection_pub.publish(coord_msg)

    def draw_detection(self, frame, prediction, depth_frame):
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
        
        # Prepare label text
        label = f"{self.target_class_name} {prediction.confidence:.2f}"
        
        # Add depth information if available
        if depth_frame is not None:
            try:
                y_coord = max(0, min(int(prediction.y), depth_frame.shape[0] - 1))
                x_coord = max(0, min(int(prediction.x), depth_frame.shape[1] - 1))
                depth_val = depth_frame[y_coord, x_coord]
                
                # Adjust units as needed (mm vs m)
                if depth_val > 1000:  # Likely in mm
                    label += f" {depth_val/1000.0:.2f}m"
                else:  # Likely in m
                    label += f" {depth_val:.2f}m"
            except (IndexError, ValueError):
                pass
        
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

    def publish_depth(self, obj_depth):    
        """
        Publish depth image
        """
        if obj_depth is None:
            self.get_logger().warn("No depth frame available")
            return 
        try:

            if obj_depth.dtype != np.float32:
                obj_depth = obj_depth / 1000.0
                #cast to float32 if needed
                obj_depth = obj_depth.astype(np.float32)

            depth_point_msg = Float32()
            depth_point_msg.data = obj_depth

            self.depth_pub.publish(depth_point_msg)
        except Exception as e:
            self.get_logger().error(f'Depth publishing error: {str(e)}')


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