import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

# YOLOv8 モデルの読み込み
model = YOLO('yolov8x.pt')  # 小さなモデルの例
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
        
    def image_callback(self, msg):
        self.get_logger().info('Received image')
        
        # ROS Image メッセージを OpenCV 画像に変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # stop signを検出する
        status, stop_sign_img = self.detect_stop_sign(cv_image)
        
        if status:
            self.publisher.publish(String(data=status))
            self.get_logger().info(f'Published stop sign status: {status}')
            
            if stop_sign_img is not None:
                # stop signの画像を表示
                cv2.imshow('Pedestrian Traffic Light', stop_sign_img)
                cv2.waitKey(1)  # ウィンドウの更新
        else:
            self.publisher.publish(String(data="None"))
            self.get_logger().info('No stop sign  detected')
            
    def detect_stop_sign(self, image):
        results = model.predict(image, classes=[11])
        img = self.return_stop_sign(results)
        if img is not None:
            return "Stop", img
        else:
            return "Go", None
    
    def return_stop_sign(self, results):
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # バウンディングボックス
            confidences = result.boxes.conf.cpu().numpy()  # 信頼度
            classes = result.boxes.cls.cpu().numpy()  # クラス

            for box, cls, conf in zip(boxes, classes, confidences):
                if int(cls) == 11:  # クラス11（stop sign）
                    x1, y1, x2, y2 = map(int, box)
                    cropped_img = result.orig_img[y1:y2, x1:x2]
                    if cropped_img.shape[0] > cropped_img.shape[1]:  # 縦長の画像なら
                        return cropped_img
        return None
    
def main(args=None):
    rclpy.init(args=args)
    stop_sign_detector = StopSignDetection()
    rclpy.spin(stop_sign_detector)
    stop_sign_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()  # ウィンドウを閉じる

if __name__ == '__main__':
    main()
