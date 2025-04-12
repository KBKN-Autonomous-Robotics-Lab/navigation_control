import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Odometry
import subprocess
import yaml
from my_msgs.action import StopFlag  # Actionメッセージのインポート
import std_msgs.msg as std_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import numpy as np
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import time
import geometry_msgs.msg as geometry_msgs
import glob
import cv2
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import transforms3d
from geometry_msgs.msg import Quaternion

class TrafficLightWaypointMonitor(Node):
    def __init__(self):
        super().__init__('trafficlight_waypoint_monitor')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom_ref_slam',
            self.odom_callback,
            10)
        
        # アクションをトリガーする絶対座標範囲 [minx, maxx, miny, maxy]
        self.action_zones = [
            [-240.0, -220.0, 56.5, 61.5],  # Singou 1
            [-235.0, -233.0, 69.0, 89.0],  # Singou 2
            [-249.0, -244.0, 69.0, 89.0],  # Singou 3
            [-240.0, -220.0, 68.4, 70.4],  # Singou 4
        ]

        self.action_client = ActionClient(self, StopFlag, 'traffic_flag')  # ActionClientの設定
        self.action_sent = {}  # 各ゾーンでアクションが送信されたかを追跡
        for i in range(len(self.action_zones)):
            self.action_sent[i] = False

    def odom_callback(self, msg):
        current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        for i, zone in enumerate(self.action_zones):
            if self.is_in_zone(current_position, zone) and not self.action_sent[i]:
                self.get_logger().info(f"Entered action zone {i}: {zone}")
                self.traffic_action = True
                self.send_action_request()
                self.action_sent[i] = True  # アクションが送信されたことをマーク

    def is_in_zone(self, position, zone):
        """現在の位置が指定ゾーン内にあるかを判定"""
        x, y = position
        minx, maxx, miny, maxy = zone
        return minx <= x <= maxx and miny <= y <= maxy

    # サーバーにアクションを送信する関数
    def send_action_request(self):
        goal_msg = StopFlag.Goal()
        
        if self.traffic_action:
            goal_msg.b = 1  # start judge
        else:
            goal_msg.b = 0
            
        goal_msg.a = 2  # 任意の値を設定

        self.action_client.wait_for_server()
        self.future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.future.add_done_callback(self.response_callback)

    def feedback_callback(self, feedback):
        self.get_logger().info(f"Received feedback: {feedback.feedback.rate}")

    def response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.sum}")

def main(args=None):
    rclpy.init(args=args)
    trafficlight_waypoint_monitor = TrafficLightWaypointMonitor()
    rclpy.spin(trafficlight_waypoint_monitor)


if __name__ == '__main__':
    main()

