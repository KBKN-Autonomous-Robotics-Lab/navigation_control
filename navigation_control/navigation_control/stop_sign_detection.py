import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO
import easyocr
import re
import tkinter as tk
from threading import Thread

# YOLOv8 モデルの読み込み
model = YOLO('yolov8x.pt')
model.classes = [11]

# グローバル変数（GUIで表示するため）
recognized_text  = "Initializing..."

# GUI
def start_gui():
    global recognized_text
    root = tk.Tk()
    root.title("Detected Text on Stop Sign")
    label = tk.Label(root, text=recognized_text, font=("Helvetica", 32))
    label.pack(padx=20, pady=20)

    def update_label():
        label.config(text=recognized_text)
        root.after(200, update_label)

    update_label()
    root.mainloop()

class StopSignDetection(Node):
    def __init__(self):
        super().__init__('stop_sign_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            1)
        self.publisher = self.create_publisher(String, 'stop_sign_status', 10)
        self.bridge = CvBridge()
        self.reader = easyocr.Reader(['en'])

    def image_callback(self, msg):
        global recognized_text

        self.get_logger().info('Received image')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        status, raw_text, stop_sign_img = self.detect_stop_sign(cv_image)

        # send topic
        self.publisher.publish(String(data=status))
        self.get_logger().info(f'Published stop sign status: {status}')

        # GUI words
        recognized_text = raw_text if raw_text else "No sign"

        if stop_sign_img is not None:
            cv2.imshow('Detected Stop Sign', stop_sign_img)
            cv2.waitKey(1)
        
    def detect_stop_sign(self, image):
        results = model.predict(image, classes=[11])
        cropped_img = self.return_stop_sign(results)

        if cropped_img is not None:
            ocr_results = self.reader.readtext(cropped_img)

            for (bbox, text, confidence) in ocr_results:
                normalized_text = re.sub(r'[^a-z]', '', text.strip().lower())
                self.get_logger().info(f'OCR検出結果: {text} → 正規化: {normalized_text} (信頼度: {confidence:.2f})')

                if normalized_text == "stop":
                     return "Stop", normalized_text, cropped_img  # topic stop GUI words

            return "Go", normalized_text if ocr_results else "", None  # OCR not stop
        else:
            return "Go", "", None  
        
    def return_stop_sign(self, results):
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            height, width = result.orig_img.shape[:2]
            min_width = 50   # size pixel
            min_height = 50

            for box, cls in zip(boxes, classes):
                if int(cls) == 11:
                    x1, y1, x2, y2 = map(int, box)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # area and size
                    if center_x > width * 0.5 and center_y < height * 0.5:
                        if box_width >= min_width and box_height >= min_height:
                            cropped_img = result.orig_img[y1:y2, x1:x2]
                            return cropped_img
        return None

def main(args=None):
    rclpy.init(args=args)

    # GUI
    gui_thread = Thread(target=start_gui, daemon=True)
    gui_thread.start()

    stop_sign_detector = StopSignDetection()
    rclpy.spin(stop_sign_detector)

    stop_sign_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

