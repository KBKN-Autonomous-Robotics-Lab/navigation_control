import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

class HumanStopDetector(Node):
    def __init__(self):
        super().__init__('human_stop_detector')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # YOLOモデルの読み込み
        self.model = YOLO('yolov8n.pt')  # person 検出

        # RealSense初期化
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        results = self.model(color_image)[0]

        stop = False

        for result in results.boxes:
            cls_id = int(result.cls)
            if self.model.names[cls_id] != 'person':
                continue

            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            distance = depth_frame.get_distance(cx, cy)

            if 0 < distance < 1.5:
                stop = True
                self.get_logger().info(f"人を検出！距離: {distance:.2f}m → 停止")
                break  # 一人でも近ければ止める

        msg = Twist()
        if stop:
            msg.linear.x = 0.0
        else:
            msg.linear.x = 0.2  # 任意の速度
        self.cmd_vel_pub.publish(msg)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HumanStopDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

