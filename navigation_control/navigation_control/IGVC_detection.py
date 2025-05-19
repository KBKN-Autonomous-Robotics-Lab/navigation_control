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


# グローバル変数（GUIで表示するため）
recognized_text  = "Initializing..."

class IGVCDetection(Node):
    def __init__(self):
        super().__init__('igvc_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # サブスクライブする画像トピック
            self.image_callback,
            1)
        self.tire_publisher = self.create_publisher(String, 'tire_status', 10)
        self.human_publisher = self.create_publisher(String, 'human_status', 10)
        self.stopsign_publisher = self.create_publisher(String, 'stop_sign_status', 10)
        self.bridge = CvBridge()
        
        # easyocr
        self.reader = easyocr.Reader(['en'])
        
        # YOLOv8モデルのロード
        self.model = YOLO('yolov8x-oiv7.pt')
        self.stop_sign_model = YOLO('yolov8x.pt')
        #self.model.classes = [536]  # クラス0（tire）のみ検出対象

    def image_callback(self, msg):
        self.get_logger().info('Received image')
        
        # ROS Imageメッセージ -> OpenCV画像
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # タイヤ検出
        tire_status, tire_img = self.detect_tire(cv_image)
        # human detect
        human_status, human_img = self.detect_human(cv_image)
        # stop sign detect
        stopsign_status, raw_text, stop_sign_img = self.detect_stop_sign(cv_image)
        
        self.tire_publisher.publish(String(data=tire_status))
        self.human_publisher.publish(String(data=human_status))
        self.stopsign_publisher.publish(String(data=stopsign_status))
        self.get_logger().info(f'Published tire status: {tire_status}')
        self.get_logger().info(f'Published human status: {human_status}')
        self.get_logger().info(f'Published stopsign status: {stopsign_status}')
        
        if tire_img is not None:
            cv2.imshow('Tire Detection', tire_img)
            cv2.waitKey(1)  # イメージウィンドウ更新
        if human_img is not None:
            cv2.imshow('Human Detection', human_img)
            cv2.waitKey(1)  # イメージウィンドウ更新
        if stop_sign_img is not None:
            cv2.imshow('Stop Sign Detection', stop_sign_img)
            cv2.waitKey(1)  # イメージウィンドウ更新

    def detect_tire(self, image):
        # 推論実行（classes指定不要、self.model.classesで制御されている）
        #results = self.model(image)
        results = self.model.predict(image, classes=[536])

        tire_img = self.extract_tire(results)
        
        if tire_img is not None:
            return "Stop", tire_img
        else:
            return "Go", None
    
    def detect_human(self, image):
        # 推論実行（classes指定不要、self.model.classesで制御されている）
        #results = self.model(image)
        results = self.model.predict(image, classes=[381])

        human_img = self.extract_human(results)
        
        if human_img is not None:
            return "Stop", human_img
        else:
            return "Go", None
    
    def detect_stop_sign(self, image):
        results = self.stop_sign_model.predict(image, classes=[11]) #model:495 stop_sign_mode:11
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
    
    def extract_human(self, results):
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                # size and priority
                # w > 50 h > 470 -> stop 5feet , w > 50 h > 240 -> change lane 10feet
                if cls_id == 381 and conf >= 0.5 and (w > 50 and h > 400): # width , height -> pixel
                    cropped_img = result.orig_img[y1:y2, x1:x2]
                    return cropped_img    

    def return_stop_sign(self, results):
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            height, width = result.orig_img.shape[:2]
            min_width = 100   # size pixel
            min_height = 100

            for box, cls in zip(boxes, classes):
                if int(cls) == 11: # model:495 stop_sign_mode:11
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

def main(args=None):
    rclpy.init(args=args)
    # GUI
    gui_thread = Thread(target=start_gui, daemon=True)
    gui_thread.start()
    igvc_detector = IGVCDetection()
    rclpy.spin(igvc_detector)
    igvc_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

