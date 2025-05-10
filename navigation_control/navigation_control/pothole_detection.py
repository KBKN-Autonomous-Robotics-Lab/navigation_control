import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
from collections import deque


class PotholeDetector(Node):

    def __init__(self):
        super().__init__('pothole_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            1)
        self.pc_publisher = self.create_publisher(PointCloud2, '/pothole_points', 1)
        self.bridge = CvBridge()
        self.detected_points = []
        self.points_buffer = deque()
        self.publish_timer = self.create_timer(0.1, self.publish_accumulated_pointcloud)
        self.lifetime_sec = 20.0  # 20秒保持

    def listener_callback(self, msg):
        # Convert ROS image to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocessing
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # White color mask
        lower_white = np.array([200, 200, 200])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(frame, lower_white, upper_white)

        # Morphology to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        now = self.get_clock().now()
        all_points = []  # 一時的な points 保存用

        for cnt in contours:
            if len(cnt) < 5 or cv2.contourArea(cnt) < 200:
                continue
            
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (major, minor), angle = ellipse

            aspect_ratio = max(major, minor) / min(major, minor)

            # カメラの角度・高さに合わせて範囲調整（例: 1.5〜3.0）
            if 1.5 < aspect_ratio < 5.0 and cy > frame.shape[0] * 0.8:
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                cv2.putText(frame, f"{aspect_ratio:.2f}", (int(cx), int(cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
                # ピクセル座標を出力
                self.get_logger().info(f"Pothole detected at pixel coordinates: ({int(cx)}, {int(cy)})")
                
                # ピクセル → 座標変換
                img_h, img_w = frame.shape[:2]
                points = []
                
                points_list = cv2.ellipse2Poly((int(cx), int(cy)), (int(major/2), int(minor/2)), int(angle), 0, 360, 10)
                for px, py in points_list:
                    x = 1.9 + (img_h - py) * 0.001
                    y = - (px - img_w / 2) * 0.001
                    z = 0.0
                    points.append((x, y, z))
                
                # 一時的に points を保存
                all_points.extend(points)

        if all_points:
            self.points_buffer.append((now, all_points))  # タイムスタンプ付きで保存

        # Show result (for debugging)
        cv2.imshow("Detected Potholes", frame)
        cv2.waitKey(1)

    def publish_accumulated_pointcloud(self):
        now = self.get_clock().now()

        # 20秒より古いものを削除
        while self.points_buffer and (now - self.points_buffer[0][0]).nanoseconds / 1e9 > self.lifetime_sec:
            self.points_buffer.popleft()

        # 残っているすべてのポイントを結合
        all_points = []
        for _, points in self.points_buffer:
            all_points.extend(points)

        if not all_points:
            # 空の PointCloud2 を publish
            header = Header()
            header.stamp = now.to_msg()
            header.frame_id = 'odom'

            pc_msg = PointCloud2()
            pc_msg.header = header
            pc_msg.height = 1
            pc_msg.width = 0
            pc_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            pc_msg.is_bigendian = False
            pc_msg.point_step = 12
            pc_msg.row_step = 0
            pc_msg.is_dense = True
            pc_msg.data = b''

            self.pc_publisher.publish(pc_msg)
            return

        # PointCloud2 フィールド定義
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = 'odom'

        pc_data = b''.join([struct.pack('fff', *pt) for pt in all_points])

        pc_msg = PointCloud2()
        pc_msg.header = header
        pc_msg.height = 1
        pc_msg.width = len(all_points)
        pc_msg.fields = fields
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12
        pc_msg.row_step = pc_msg.point_step * pc_msg.width
        pc_msg.is_dense = True
        pc_msg.data = pc_data

        self.pc_publisher.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PotholeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

