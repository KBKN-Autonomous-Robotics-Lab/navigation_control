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
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

class PotholeDetector(Node):

    def __init__(self):
        super().__init__('white_line_detector')
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw', self.listener_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom/wheel_imu', self.odom_callback, 1)
        #self.pc_publisher = self.create_publisher(PointCloud2, '/pothole_points', 1)
        self.bridge = CvBridge()
        self.robot_pose = (0.0, 0.0, 0.0)  # (x, y, yaw)
        self.points_buffer = deque()
        #self.publish_timer = self.create_timer(0.2, self.publish_accumulated_pointcloud)
        self.lifetime_sec = 20.0

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = quaternion_to_euler(ori.x, ori.y, ori.z, ori.w)
        self.robot_pose = (pos.x, pos.y, yaw)

    def listener_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # 白色の抽出
        lower_white = np.array([100, 100, 100])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(frame, lower_white, upper_white)
        
        # yellowの抽出
        lower_yellow = np.array([0, 100, 100])
        upper_yellow = np.array([80, 255, 255])
        yellow_mask = cv2.inRange(frame, lower_yellow, upper_yellow)
        

        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)


        # 輪郭抽出
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        img_h, img_w = frame.shape[:2]

        for cnt in white_contours:
            #area = cv2.contourArea(cnt)
            #if area < 500:
            #    continue

            # 回転を含む外接矩形
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            if cy < img_h / 2:
                continue

            # アスペクト比計算
            #if w < h:
            #    aspect_ratio = h / w if w != 0 else 0
            #else:
            #    aspect_ratio = w / h if h != 0 else 0

            # 縦長判定 & 十分な大きさ
            #if aspect_ratio > 2.0: # and max(w, h) > img_h * 0.5:
                # 矩形の4頂点を取得
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            # draw green point white line 
            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), cv2.FILLED)
        
        for cnt in yellow_contours:
            #area = cv2.contourArea(cnt)
            #if area < 500:
            #    continue

            # 回転を含む外接矩形
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            if cy < img_h / 2:
                continue

            # アスペクト比計算
            #if w < h:
            #    aspect_ratio = h / w if w != 0 else 0
            #else:
            #    aspect_ratio = w / h if h != 0 else 0

            # 縦長判定 & 十分な大きさ
            #if aspect_ratio > 2.0: # and max(w, h) > img_h * 0.5:
                # 矩形の4頂点を取得
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            # draw green point white line 
            cv2.drawContours(frame, [cnt], 0, (0, 255, 0), cv2.FILLED)


        cv2.imshow("Detected Lane Lines", frame)
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

def main(args=None):
    rclpy.init(args=args)
    node = PotholeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

