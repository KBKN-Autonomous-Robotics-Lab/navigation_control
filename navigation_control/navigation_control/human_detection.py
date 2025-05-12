import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

class HumanDetection(Node):
    def __init__(self):
        super().__init__('human_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # サブスクライブする画像トピック
            self.image_callback,
            1)
        self.publisher = self.create_publisher(String, 'human_status', 10)
        self.bridge = CvBridge()
        
        # YOLOv8モデルのロード
        self.model = YOLO('yolov8x.pt')
        self.model.classes = [0]  # クラス0（human）のみ検出対象

    def image_callback(self, msg):
        self.get_logger().info('Received image')
        
        # ROS Imageメッセージ -> OpenCV画像
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # human detect
        status, human_img = self.detect_human(cv_image)
        
        self.publisher.publish(String(data=status))
        self.get_logger().info(f'Published human status: {status}')
        
        if human_img is not None:
            cv2.imshow('human Detection', human_img)
            cv2.waitKey(1)  # イメージウィンドウ更新

    def detect_human(self, image):
        # 推論実行（classes指定不要、self.model.classesで制御されている）
        results = self.model(image)

        human_img = self.extract_human(results)
        
        if human_img is not None:
            return "Stop", human_img
        else:
            return "Go", None

    def extract_human(self, results):
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                # size and priority
                if cls_id == 0 and conf >= 0.5 and (w > 50 or h > 230): # width , height -> pixel
                    cropped_img = result.orig_img[y1:y2, x1:x2]
                    return cropped_img

        return None
        
def main(args=None):
    rclpy.init(args=args)
    human_detector = HumanDetection()
    try:
        rclpy.spin(human_detector)
    except KeyboardInterrupt:
        pass
    finally:
        human_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()  # ウィンドウを確実に閉じる

if __name__ == '__main__':
    main()

