#!/usr/bin/env python
# ROS 2 imports
import rclpy  # Core ROS 2 Python library
from rclpy.node import Node  # Node class for creating ROS 2 nodes

# Message types we'll publish
from sensor_msgs.msg import Image  # Standard ROS Image message
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool

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
        self.detection_pub = self.create_publisher(
            PointStamped,
            '/object_detections/coordinates',
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
            Image,
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

        # === ROBOFLOWOAK INITIALIZATION ===
        self.rf = RoboflowOak(
            model="garbage-detection-2",    # Your garbage detection model
            confidence=0.05,                # Confidence threshold
            overlap=0.5,                    # NMS overlap threshold
            version="1", 
            api_key="R4jbOhEOxwSSDOBryrhH", # Your Roboflow API key
            rgb=True,                       # Whether to use RGB stream
            depth=True,                     # Whether to use depth stream
            device=None,                    # Auto device selection
            blocking=True                   # Blocking detection call
        )

        # === DETECTION PARAMETERS ===
        self.target_class_name = "garbage"  # The class name we're looking for
        self.target_class_id = 0            # Assuming garbage has class_id 0
        
        # === CREATE A TIMER CALLBACK ===
        # Calls self.run_model every 0.1 seconds (10Hz)
        self.create_timer(0.1, self.run_model)

        self.get_logger().info("Object Detection Node initialized - looking for garbage")

    def run_model(self):
        """
        Main detection loop - runs inference and publishes results
        """
        t0 = time.time()

        try:
            # Run detection on OAK-D
            result, frame, raw_frame, depth = self.rf.detect()
            predictions = result["predictions"]

            # Filter predictions for garbage only
            garbage_predictions = [pred for pred in predictions if pred.class_id == self.target_class_id]

            # Publish detection flag
            detection_flag_msg = Bool()
            detection_flag_msg.data = len(garbage_predictions) > 0
            self.detection_flag_pub.publish(detection_flag_msg)

            # Process each garbage detection
            for pred in garbage_predictions:
                # Publish coordinates
                self.publish_coordinates(pred, depth)
                
                # Draw bounding box on frame
                self.draw_detection(frame, pred, depth)

            # Publish annotated image
            self.publish_image(frame)
            
            # Publish depth image
            if depth is not None:
                self.publish_depth(depth)

            # Log performance
            t = time.time() - t0
            fps = 1/t if t > 0 else 0
            self.get_logger().info(f'FPS: {fps:.2f}, Garbage detections: {len(garbage_predictions)}')

        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')

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
                
                # Convert depth from mm to meters if needed (depends on your OAK-D setup)
                # Uncomment the next line if depth is in mm
                # depth_val = depth_val / 1000.0
                
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
        label = f"{prediction.class_name} {prediction.confidence:.2f}"
        
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

    def publish_depth(self, depth_frame):
        """
        Publish depth image
        """
        try:
            # Normalize depth for visualization if needed
            if depth_frame.dtype != np.float32:
                depth_normalized = depth_frame.astype(np.float32)
            else:
                depth_normalized = depth_frame
                
            depth_msg = self.bridge.cv2_to_imgmsg(depth_normalized, encoding='32FC1')
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            depth_msg.header.frame_id = 'oakd_camera_frame'
            self.depth_pub.publish(depth_msg)
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