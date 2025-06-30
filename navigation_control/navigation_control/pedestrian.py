import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloV8StopSignNode(Node):
    def __init__(self):
        super().__init__('yolov8_stop_sign_node')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8x.pt')  # 小さいモデルでOK（COCO学習済み）
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # カメラ画像のトピックに合わせて変更
            self.image_callback,
            1)
    
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 0:  # stopsign class ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"pedestrian {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('YOLOv8 - Stop Sign Detection', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8StopSignNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

