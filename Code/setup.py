import os
from glob import glob
from setuptools import setup

package_name = 'final_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include any other data files
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Final package for lane detection and guidance system',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_driver.py = final_pkg.camera_driver:main',
            'lane_guidance_node = final_pkg.lane_guidance_node:main',
            'servo_sweeper.py = final_pkg.servo_sweeper:main',
            # Add other executables as needed
            # 'object_detection_node = final_pkg.object_detection_node:main',
            # 'vesc_twist_node = final_pkg.vesc_twist_node:main',
            # 'adafruit_twist_node = final_pkg.adafruit_twist_node:main',
        ],
    },
)