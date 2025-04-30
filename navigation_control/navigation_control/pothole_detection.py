import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class PotholeDetector(Node):

    def __init__(self):
        super().__init__('pothole_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            1)
        self.bridge = CvBridge()

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

