import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO
import easyocr  # read stop
import re

# YOLOv8 モデルの読み込み
model = YOLO('yolov8x.pt')  # モデルサイズに応じて調整可
model.classes = [11]  # クラス11（stop sign）のみを検出する

class StopSignDetection(Node):
    def __init__(self):
        super().__init__('stop_sign_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # トピック名を指定
            self.image_callback,
            1)
        self.publisher = self.create_publisher(String, 'stop_sign_status', 10)
        self.bridge = CvBridge()
        self.reader = easyocr.Reader(['en'])  # ★EasyOCRのリーダー（英語のみ）

    def image_callback(self, msg):
        self.get_logger().info('Received image')

        # ROS Image メッセージを OpenCV 画像に変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # stop signを検出し、文字認識を行う
        status, stop_sign_img = self.detect_stop_sign(cv_image)

        # 結果をpublish
        self.publisher.publish(String(data=status))
        self.get_logger().info(f'Published stop sign status: {status}')

        if stop_sign_img is not None:
            cv2.imshow('Detected Stop Sign', stop_sign_img)
            cv2.waitKey(1)

    def detect_stop_sign(self, image):
        results = model.predict(image, classes=[11])
        cropped_img = self.return_stop_sign(results)

        if cropped_img is not None:
            ocr_results = self.reader.readtext(cropped_img)

            for (bbox, text, confidence) in ocr_results:
                # 文字列の前後を除去し、小文字化し、記号を除去
                normalized_text = re.sub(r'[^a-z]', '', text.strip().lower())
                self.get_logger().info(f'OCR検出結果: {text} → 正規化: {normalized_text} (信頼度: {confidence:.2f})')
            
                if normalized_text == "stop":
                    return "Stop", cropped_img
            return "Go", None
        else:
            return "Go", None

    def return_stop_sign(self, results):
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == 11:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_img = result.orig_img[y1:y2, x1:x2]
                    if cropped_img.shape[0] > cropped_img.shape[1]:  # 縦長かもだけど一旦使う
                        return cropped_img
                    else:
                        return cropped_img  # 横長でも返す
        return None

def main(args=None):
    rclpy.init(args=args)
    stop_sign_detector = StopSignDetection()
    rclpy.spin(stop_sign_detector)
    stop_sign_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

