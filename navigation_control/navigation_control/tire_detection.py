import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

class TireDetection(Node):
    def __init__(self):
        super().__init__('tire_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # サブスクライブする画像トピック
            self.image_callback,
            1)
        self.publisher = self.create_publisher(String, 'tire_status', 10)
        self.bridge = CvBridge()
        
        # YOLOv8モデルのロード
        self.model = YOLO('yolov8x-oiv7.pt')
        self.model.classes = [536]  # クラス0（tire）のみ検出対象

    def image_callback(self, msg):
        self.get_logger().info('Received image')
        
        # ROS Imageメッセージ -> OpenCV画像
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # タイヤ検出
        status, tire_img = self.detect_tire(cv_image)
        
        self.publisher.publish(String(data=status))
        self.get_logger().info(f'Published tire status: {status}')
        
        if tire_img is not None:
            cv2.imshow('Tire Detection', tire_img)
            cv2.waitKey(1)  # イメージウィンドウ更新

    def detect_tire(self, image):
        # 推論実行（classes指定不要、self.model.classesで制御されている）
        results = self.model(image)

        tire_img = self.extract_tire(results)
        
        if tire_img is not None:
            return "Stop", tire_img
        else:
            return "Go", None

    def extract_tire(self, results):
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 536 and conf >=0.5:  # tire class ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_img = result.orig_img[y1:y2, x1:x2]
                    return cropped_img
        return None

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

