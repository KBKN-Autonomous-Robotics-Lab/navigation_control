import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO
from nav_msgs.msg import Odometry
from collections import deque
import math
import struct
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

class TireDetection(Node):
    def __init__(self):
        super().__init__('tire_detector')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw',  # サブスクライブする画像トピック
            self.image_callback,
            10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom/wheel_imu', self.odom_callback, 1)
        
        self.publisher = self.create_publisher(String, 'tire_status', 10)
        self.pc_publisher = self.create_publisher(PointCloud2, '/tire_points', 1)
        self.bridge = CvBridge()
        self.robot_pose = (0.0, 0.0, 0.0)  # (x, y, yaw)
        self.points_buffer = deque()
        self.publish_timer = self.create_timer(0.2, self.publish_accumulated_pointcloud)
        self.lifetime_sec = 10.0
        
        # YOLOv8モデルのロード
        self.model = YOLO('yolov8x-oiv7.pt')
        self.model.classes = [536]  # クラス0（tire）のみ検出対象

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = quaternion_to_euler(ori.x, ori.y, ori.z, ori.w)
        self.robot_pose = (pos.x, pos.y, yaw)

    def image_callback(self, msg: CompressedImage):
        self.get_logger().info('Received image')
        np_arr = np.frombuffer(msg.data, np.uint8)        
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # ROS Imageメッセージ -> OpenCV画像
        #cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # タイヤ検出
        status, tire_img, tire_info, img_shape = self.detect_tire(cv_image)
        self.publisher.publish(String(data=status))
        self.get_logger().info(f'Published tire status: {status}')

        now = self.get_clock().now()
        all_points = []

        if tire_info:
            cx, cy, w, h = tire_info
            img_h, img_w = img_shape[:2]
    
            # タイヤの輪郭ポイントを生成（楕円として仮定）
            points_list = cv2.ellipse2Poly((int(cx), int(cy)), (int(w/2), int(h/2)), 0, 0, 360, 10)
            local_points = []

            for px, py in points_list:
                # 深度情報がないため、仮の奥行き距離（例えば1.9m）を設定
                x = 1.9 + (img_h - py) * 0.001
                y = - (px - img_w / 2) * 0.001
                z = 0.0

                gx, gy = transform_point_to_global(x, y, self.robot_pose)
                local_points.append((gx, gy, z))

            all_points.extend(local_points)
            self.points_buffer.append((now, all_points))
        
        if tire_img is not None:
            cv2.imshow('Tire Detection', tire_img)
            cv2.waitKey(1)  # イメージウィンドウ更新

    def detect_tire(self, image):
        # 推論実行（classes指定不要、self.model.classesで制御されている）
        results = self.model(image)

        tire_img, tire_info, img_shape = self.extract_tire(results)
        
        if tire_img is not None:
            return "Stop", tire_img,  tire_info, img_shape
        else:
            return "Go", None, None, None
        
    def extract_tire(self, results):
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 536 and conf >=0.5:  # tire class ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    cropped_img = result.orig_img[y1:y2, x1:x2]
                    return cropped_img, (cx, cy, w, h), result.orig_img.shape
        return None, None, None
        
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
    tire_detector = TireDetection()
    try:
        rclpy.spin(tire_detector)
    except KeyboardInterrupt:
        pass
    finally:
        tire_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()  # ウィンドウを確実に閉じる

if __name__ == '__main__':
    main()

