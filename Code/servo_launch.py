#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('final_pkg'),
            'config',
            'lane_detection_params.yaml'
        ]),
        description='Path to the configuration file'
    )
    
    # Get the config file path
    config_file = LaunchConfiguration('config_file')
    
    # Camera driver node from final_pkg
    camera_driver_node = Node(
        package='final_pkg',
        executable='camera_driver.py',
        name='camera_driver',
        output='screen',
        parameters=[config_file],
        remappings=[
            # Add any topic remappings if needed
            # ('/camera/image_raw', '/image_raw'),
        ]
    )
    
    # Lane detection node from ucsd_robocar_lane_detection2_pkg
    lane_detection_node = Node(
        package='ucsd_robocar_lane_detection2_pkg',
        executable='lane_detection_node',  # Adjust if different
        name='lane_detection_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            # Add topic remappings to connect camera to lane detection
            # ('/camera/image_raw', '/image_raw'),
        ]
    )
    
    # Lane guidance/path planner node from final_pkg
    lane_guidance_node = Node(
        package='final_pkg',
        executable='lane_guidance_node',  # Your main controller
        name='lane_guidance_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            # Connect centroid topic from lane detection to guidance
            # ('/centroid', '/lane_detection/centroid'),
        ]
    )
    
    # Servo sweeper node (will be launched by lane_guidance_node when needed)
    # But we can also make it available as a standalone node
    servo_sweeper_node = Node(
        package='final_pkg',
        executable='servo_sweeper.py',
        name='servo_sweeper',
        output='screen',
        parameters=[config_file],
        # This node will be launched programmatically by lane_guidance_node
        # So we might not include it here, or use a condition
    )
    
    # Optional: Object detection node if you have one
    # object_detection_node = Node(
    #     package='final_pkg',
    #     executable='object_detection_node',
    #     name='object_detection_node',
    #     output='screen',
    #     parameters=[config_file],
    # )
    
    # Optional: Motor control node (VESC or Adafruit)
    # motor_control_node = Node(
    #     package='final_pkg',
    #     executable='vesc_twist_node',  # or 'adafruit_twist_node'
    #     name='vesc_twist_node',
    #     output='screen',
    #     parameters=[config_file],
    # )

    return LaunchDescription([
        config_file_arg,
        camera_driver_node,
        lane_detection_node,
        lane_guidance_node,
        # servo_sweeper_node,  # Comment out if launched programmatically
        # object_detection_node,  # Uncomment if needed
        # motor_control_node,  # Uncomment if needed
    ])