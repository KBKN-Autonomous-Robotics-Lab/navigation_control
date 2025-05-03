import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct


class PotholeDetector(Node):

    def __init__(self):
        super().__init__('pothole_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            1)
        self.pc_publisher = self.create_publisher(PointCloud2, '/pothole_points', 10)
        self.bridge = CvBridge()
        self.detected_points = []

    def listener_callback(self, msg):
        # Convert ROS image to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocessing
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        #hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

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

        for cnt in contours:
            #area = cv2.contourArea(cnt)
            #if area < 100:
            #    continue

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
                x = 1.9 + (img_h - cy) * 0.001  # 下から上へ
                y = (cx + img_w / 2) * 0.001  # 中心から左右へ
                z = 0.0

                # PointCloud2 用データ生成
                points = [(x, y, z)] 

                # PointCloud2 メッセージ作成
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                ]

                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = 'odom'

                pc_data = b''.join([struct.pack('fff', *pt) for pt in points])

                pc_msg = PointCloud2()
                pc_msg.header = header
                pc_msg.height = 1
                pc_msg.width = len(points)
                pc_msg.fields = fields
                pc_msg.is_bigendian = False
                pc_msg.point_step = 12
                pc_msg.row_step = pc_msg.point_step * pc_msg.width
                pc_msg.is_dense = True
                pc_msg.data = pc_data

                self.pc_publisher.publish(pc_msg)
        """
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity > 0.7:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        """

        # Show result (for debugging)
        cv2.imshow("Detected Circles", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = PotholeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

