import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

import cv2
import numpy as np


class StopLineDetector(Node):

    def __init__(self):
        super().__init__("stop_line_detector")

        #############################################
        # Camera
        #############################################

        self.use_test_image = False
        self.test_image_path = "/home/ubuntu/test.png"

        if not self.use_test_image:
            self.cap = cv2.VideoCapture("/dev/sensors/webcam")

        #############################################
        # Publisher
        #############################################

        self.stop_pub = self.create_publisher(
            Bool,
            "/stop_line",
            10
        )

        #############################################
        # Timer
        #############################################

        self.timer = self.create_timer(
            0.05,
            self.timer_callback
        )

        #############################################
        # Camera calibration
        #############################################

        self.DIM = (736, 736)

        self.K = np.array([
            [230.42134889608045, 0.0, 366.90152469610496],
            [0.0, 230.45799046281923, 367.1826387781316],
            [0.0, 0.0, 1.0]
        ])

        self.D = np.array([
            [-0.02057316569139076],
            [0.0027174786060516474],
            [-0.0033945666290717286],
            [0.0005725592135935245]
        ])

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K,
            self.D,
            np.eye(3),
            self.K,
            self.DIM,
            cv2.CV_16SC2
        )

    ############################################################

    def timer_callback(self):

        #############################################
        # Image
        #############################################

        if self.use_test_image:

            frame = cv2.imread(self.test_image_path)

        else:

            ret, frame = self.cap.read()

            if not ret:
                return

            frame = frame[:, :736]
            frame = self.undistort_image(frame)

        #############################################

        roi, mask = self.preprocess_image(frame)

        detected, contour = self.extract_stop_line(mask)

        #############################################
        # Debug
        #############################################

        if detected:

            cv2.drawContours(
                roi,
                [contour],
                -1,
                (0,255,0),
                3
            )

            rect = cv2.minAreaRect(contour)

            box = cv2.boxPoints(rect)
            box = np.int32(box)

            cv2.drawContours(
                roi,
                [box],
                0,
                (255,0,0),
                2
            )

            (cx,cy),(w,h),angle = rect

            cv2.circle(
                roi,
                (int(cx),int(cy)),
                5,
                (0,0,255),
                -1
            )

            cv2.putText(
                roi,
                "STOP LINE",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )

        #############################################
        # Publish
        #############################################

        msg = Bool()
        msg.data = detected

        self.stop_pub.publish(msg)

        #############################################

        cv2.imshow("Mask",mask)
        cv2.imshow("StopLine",roi)

        cv2.waitKey(1)

    ############################################################

    def preprocess_image(self, frame):

        h,w = frame.shape[:2]

        roi = frame[:int(h*0.8), :]

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        #############################################
        # White extraction
        #############################################

        lower = np.array([0,0,180])
        upper = np.array([180,40,255])

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        #############################################

        kernel = np.ones((5,5),np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        mask = cv2.GaussianBlur(
            mask,
            (5,5),
            0
        )

        return roi,mask

    ############################################################

    def extract_stop_line(self, mask):

        contours,_ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 1000:
                continue

            rect = cv2.minAreaRect(contour)

            (_, _), (w,h), angle = rect

            long_side = max(w,h)
            short_side = min(w,h)

            #########################################
            # Stop line condition
            #########################################

            if long_side < 250:
                continue

            if short_side > 50:
                continue

            return True, contour

        return False, None

    ############################################################

    def undistort_image(self, frame):

        if frame.shape[1::-1] != self.DIM:
            frame = cv2.resize(frame,self.DIM)

        return cv2.remap(
            frame,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )


############################################################

def main():

    rclpy.init()

    node = StopLineDetector()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()