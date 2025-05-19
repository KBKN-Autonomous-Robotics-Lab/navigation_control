import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from std_msgs.msg import Header
import cv2
import numpy as np
import struct
from collections import deque
import math

def quaternion_to_euler(x, y, z, w):
    rot_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),       1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),       2 * (y*z + x*w),     1 - 2 * (x**2 + y**2)]
    ])
    roll = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
    pitch = np.arctan2(-rot_matrix[2, 0], np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2))
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return roll, pitch, yaw

def transform_point_to_global(x, y, robot_pose):
    rx, ry, yaw = robot_pose
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    gx = rx + x * cos_yaw - y * sin_yaw
    gy = ry + x * sin_yaw + y * cos_yaw
    return gx, gy

class PotholeDetector(Node):

    def __init__(self):
        super().__init__('pothole_detector')
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.listener_callback, 1)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 1)
        self.pc_publisher = self.create_publisher(PointCloud2, '/pothole_points', 1)
        self.bridge = CvBridge()
        self.robot_pose = (0.0, 0.0, 0.0)  # (x, y, yaw)
        self.points_buffer = deque()
        self.publish_timer = self.create_timer(0.1, self.publish_accumulated_pointcloud)
        self.lifetime_sec = 20.0

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = quaternion_to_euler(ori.x, ori.y, ori.z, ori.w)
        self.robot_pose = (pos.x, pos.y, yaw)

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        lower_white = np.array([200, 200, 200])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(frame, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        now = self.get_clock().now()
        all_points = []

        for cnt in contours:
            if len(cnt) < 5 or cv2.contourArea(cnt) < 200:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (major, minor), angle = ellipse
            aspect_ratio = max(major, minor) / min(major, minor)
            is_horizontal = major < minor and (80 < angle < 110)

            if is_horizontal and 3.0 < aspect_ratio < 5.0 and cy > frame.shape[0] * 0.8:
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                cv2.putText(frame, f"{aspect_ratio:.2f}", (int(cx), int(cy)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"{angle:.1f} deg", (int(cx), int(cy) + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                self.get_logger().info(f"Pothole at: ({int(cx)}, {int(cy)})")
                img_h, img_w = frame.shape[:2]

                local_points = []
                points_list = cv2.ellipse2Poly((int(cx), int(cy)), (int(major/2), int(minor/2)), int(angle), 0, 360, 10)
                for px, py in points_list:
                    x = 1.9 + (img_h - py) * 0.001
                    y = - (px - img_w / 2) * 0.001
                    z = 0.0
                    gx, gy = transform_point_to_global(x, y, self.robot_pose)
                    local_points.append((gx, gy, z))

                all_points.extend(local_points)

        if all_points:
            self.points_buffer.append((now, all_points))

        cv2.imshow("Detected Potholes", frame)
        cv2.waitKey(1)

    def publish_accumulated_pointcloud(self):
        now = self.get_clock().now()
        while self.points_buffer and (now - self.points_buffer[0][0]).nanoseconds / 1e9 > self.lifetime_sec:
            self.points_buffer.popleft()

        all_points = []
        for _, points in self.points_buffer:
            all_points.extend(points)

        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = 'odom'  # グローバル座標系

        pc_msg = PointCloud2()
        pc_msg.header = header
        pc_msg.height = 1
        pc_msg.width = len(all_points)
        pc_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12
        pc_msg.row_step = pc_msg.point_step * pc_msg.width
        pc_msg.is_dense = True
        pc_msg.data = b''.join([struct.pack('fff', *pt) for pt in all_points])

        self.pc_publisher.publish(pc_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PotholeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

