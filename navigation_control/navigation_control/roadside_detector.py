import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2
import numpy as np
from my_msgs.msg import RoadsideInfo

class RoadsideDetector(Node):
    def __init__(self):
        super().__init__("roadside_detector")

        # True: PNG画像を使用
        # False: カメラを使用
        self.use_test_image = False
        self.test_image_path = "/home/ubuntu/ros2_ws/src/navigation_control/navigation_control/test/test1.png"

        if not self.use_test_image:
            self.cap = cv2.VideoCapture("/dev/sensors/webcam")
        
        # publisher
        self.roadside_pub = self.create_publisher(RoadsideInfo, "/roadside_info", 10)

        # timer
        self.timer = self.create_timer(0.05, self.timer_callback)

        # parameter
        self.pixel_to_meter = 0.002
        self.detected = False
        self.distance = 0.0
        self.angle = 0.0

        # caribrate parameter
        self.DIM=(1504, 1504)
        self.K=np.array([[467.94972918063576, 0.0, 751.230452623187], [0.0, 468.05613091465483, 750.9019494139914], [0.0, 0.0, 1.0]])
        self.D=np.array([[-0.014641575667383097], [-0.010755035156452033], [0.003932361337988872], [-0.0007419693374374312]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2)
    
    def timer_callback(self):
        #############################################
        # Camera
        #############################################
        if self.use_test_image:

            frame = cv2.imread(self.test_image_path)

            if frame is None:
                self.get_logger().error("Cannot load test image.")
                return

        else:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Camera Error")
                return
        
        front = frame[:, :1504]
        rear = frame[:, 1504:]
        frame = self.undistort_image(front)

        roi, mask = self.preprocess_image(frame)

        #############################################
        # Boundary Extraction
        #############################################
        success, xs, ys, contour = self.extract_boundary_points(mask, roi)
        detected = self.detected
        boundary_distance = self.distance
        boundary_angle = self.angle

        #############################################
        # Curve Fitting
        #############################################

        if success:

            cv2.drawContours(
                roi,
                [contour],
                -1,
                (0, 255, 0),
                2
            )

            success, boundary_distance, boundary_angle, x_bottom, coef = \
                self.fit_boundary_curve(
                    xs,
                    ys,
                    roi
                )

            if success:

                detected = True

                ##################################################
                # 画像中心
                ##################################################

                image_center = roi.shape[1] // 2

                cv2.line(
                    roi,
                    (image_center, 0),
                    (image_center, roi.shape[0]),
                    (255, 0, 0),
                    2
                )

                ##################################################
                # 境界までの距離表示
                ##################################################

                cv2.putText(
                    roi,
                    f"{boundary_distance:.2f} m",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                ##################################################
                # 角度表示
                ##################################################

                cv2.putText(
                    roi,
                    f"{np.degrees(boundary_angle):.1f} deg",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

        #############################################
        # Publish
        #############################################

        msg = RoadsideInfo()

        msg.detected = detected
        msg.boundary_distance = float(boundary_distance)
        msg.boundary_angle = float(boundary_angle)

        self.roadside_pub.publish(msg)

        #############################################
        # Debug View
        #############################################

        cv2.imshow("Mask", mask)
        cv2.imshow("Roadside", roi)

        cv2.waitKey(1)
    
    def preprocess_image(self, frame):
        """
        カメラ画像から路側帯検出用のROIとマスク画像を生成する

        Parameters
        ----------
        frame : np.ndarray
            カメラ画像(BGR)

        Returns
        -------
        roi : np.ndarray
            切り出したROI画像

        mask : np.ndarray
            HSV二値化後のマスク画像
        """

        #############################################
        # ROI
        #############################################
        h, w = frame.shape[:2]
        front = frame[:, :w//2]
        #front = frame[:, w//2:]
        h, w = front.shape[:2]
        #cv2.imshow("Front", front)
        #cv2.imshow("Frame", frame)
        print(f"width = {w}, height = {h}")

        roi = front[int(h * 0.55):h, :]

        #############################################
        # HSV
        #############################################

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        #############################################
        # 朱色抽出
        #############################################

        lower1 = np.array([0, 80, 80])
        upper1 = np.array([15, 255, 255])

        lower2 = np.array([160, 80, 80])
        upper2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(
            hsv,
            lower1,
            upper1
        )

        mask2 = cv2.inRange(
            hsv,
            lower2,
            upper2
        )

        mask = cv2.bitwise_or(
            mask1,
            mask2
        )

        #############################################
        # Morphology
        #############################################

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

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

        #############################################
        # GaussianBlur
        #############################################

        mask = cv2.GaussianBlur(
            mask,
            (5, 5),
            0
        )

        return roi, mask

    def undistort_image(self, frame):
        """
        Fish-eye画像を歪み補正する
        """

        if frame.shape[1::-1] != self.DIM:
            frame = cv2.resize(frame, self.DIM)

        undistorted = cv2.remap(
            frame,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        return undistorted

    def extract_boundary_points(self, mask, roi):
        """
        HSVマスクから車道側境界点を抽出する

        Parameters
        ----------
        mask : np.ndarray
            HSV二値画像

        Returns
        -------
        success : bool
            境界抽出できたか

        xs : np.ndarray
            境界点x座標

        ys : np.ndarray
            境界点y座標

        contour : np.ndarray
            最大輪郭（デバッグ用）
        """

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            return False, None, None, None

        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < 500:
            return False, None, None, None

        height = mask.shape[0]

        # ROI下40%
        y_min = int(height * 0.6)

        boundary_points = {}

        for p in contour:

            x = int(p[0][0])
            y = int(p[0][1])

            if y < y_min:
                continue

            ####################################################
            # 左側路側帯なら車道側境界は「右端」
            ####################################################

            if y not in boundary_points:

                boundary_points[y] = x

            else:

                boundary_points[y] = min(
                    boundary_points[y],
                    x
                )

        if len(boundary_points) < 20:
            return False, None, None, None

        ys = np.array(
            sorted(boundary_points.keys()),
            dtype=np.float32
        )

        xs = np.array(
            [boundary_points[y] for y in ys],
            dtype=np.float32
        )
        # 境界点を描画
        for x, y in zip(xs, ys):
            cv2.circle(
                roi,
                (int(x), int(y)),
                2,
                (0, 0, 255),   # 赤色
                -1
            )

        return True, xs, ys, contour
    
    def fit_boundary_curve(self, xs, ys, roi):
        """
        境界点を2次多項式で近似する

        Parameters
        ----------
        xs : np.ndarray
            境界点x座標

        ys : np.ndarray
            境界点y座標

        roi : np.ndarray
            デバッグ描画用

        Returns
        -------
        success : bool

        boundary_distance : float

        boundary_angle : float

        x_bottom : float

        coef : np.ndarray
        """

        #############################################
        # 点数不足
        #############################################

        if len(xs) < 20:
            return False, 0.0, 0.0, 0.0, None

        #############################################
        # 2次近似
        #############################################

        coef = np.polyfit(
            ys,
            xs,
            2
        )

        #############################################
        # 画像下端
        #############################################

        y_bottom = roi.shape[0] - 1

        x_bottom = (
            coef[0] * y_bottom**2
            + coef[1] * y_bottom
            + coef[2]
        )

        #############################################
        # 接線
        #
        # x = ay²+by+c
        #
        # dx/dy = 2ay+b
        #############################################

        dxdy = (
            2.0 * coef[0] * y_bottom
            + coef[1]
        )

        boundary_angle = np.arctan(dxdy)

        #############################################
        # 境界までの距離(pixel)
        #############################################

        image_center = roi.shape[1] / 2

        distance_pixel = image_center - x_bottom

        boundary_distance = (
            distance_pixel
            * self.pixel_to_meter
        )

        #############################################
        # デバッグ描画
        #############################################

        curve = []

        for y in range(int(min(ys)), int(max(ys))):

            x = (
                coef[0] * y**2
                + coef[1] * y
                + coef[2]
            )

            curve.append(
                (
                    int(x),
                    int(y)
                )
            )

        for i in range(len(curve)-1):

            cv2.line(
                roi,
                curve[i],
                curve[i+1],
                (0,255,255),
                2
            )

        #############################################
        # 下端位置
        #############################################

        cv2.circle(
            roi,
            (
                int(x_bottom),
                int(y_bottom)
            ),
            6,
            (0,0,255),
            -1
        )

        #############################################

        ys_draw = np.arange(int(min(ys)), roi.shape[0])

        xs_draw = np.polyval(coef, ys_draw)

        for x, y in zip(xs_draw, ys_draw):

            cv2.circle(
                roi,
                (int(x), int(y)),
                1,
                (255, 0, 0),   # 青色
                -1
            )

        return (
            True,
            boundary_distance,
            boundary_angle,
            x_bottom,
            coef
        )

def main():
    rclpy.init()
    node = RoadsideDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()