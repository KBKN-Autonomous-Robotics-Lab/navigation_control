from setuptools import find_packages, setup
import os
from glob import glob
from setuptools import find_packages

package_name = 'navigation_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # アクションファイルのインストール
        (os.path.join('share', package_name), glob('action/*.action')),
        # launch
        ('share/' + package_name + '/launch', ['launch/navigation_control.launch.xml','launch/human_lanechange.launch.xml','launch/human_stop.launch.xml','launch/pothole_detection.launch.xml','launch/stop_sign_detection.launch.xml','launch/tire_detection.launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'stop_flag = navigation_control.stop_flag:main',
        "button = navigation_control.button:main",
        "trafficlight_waypoint_monitor = navigation_control.trafficlight_waypoint_monitor:main",
        "judge_trafficlight = navigation_control.judge_trafficlight:main",
        'camera_publisher = navigation_control.camera_publisher:main',
        'gps_waypoint = navigation_control.gps_waypoint:main',
        'rtk_gps_waypoint = navigation_control.rtk_gps_waypoint:main',
        'gps_waypoint_yaml = navigation_control.gps_waypoint_yaml:main',
        'stop_sign_detection = navigation_control.stop_sign_detection:main',
        'human_realsense = navigation_control.human_realsense:main',
        'stop_sign = navigation_control.stop_sign:main',
        'pedestrian = navigation_control.pedestrian:main',
        'stop_sign_control = navigation_control.stop_sign_control:main',
        'human_detection = navigation_control.human_detection:main',
        'human_control = navigation_control.human_control:main',
        'pothole_detection = navigation_control.pothole_detection:main',
        'white_line_detection = navigation_control.white_line_detection:main',
        'tire = navigation_control.tire:main',
        'tire_detection = navigation_control.tire_detection:main',
        'tire_control = navigation_control.tire_control:main',
        'IGVC_detection = navigation_control.IGVC_detection:main',
        ],
    },
)
