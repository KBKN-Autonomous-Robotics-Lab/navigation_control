import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import Bool
import cv2
import numpy as np
from my_msgs.msg import RoadsideInfo
import nav_msgs.msg as nav_msgs

class TsukubaController(Node):
    def __init__(self):
        super().__init__("tsukuba_controller")

        # True: PNG画像を使用
        # False: カメラを使用
        self.use_test_image = False
        self.test_image_path = "/home/ubuntu/ros2_ws/src/navigation_control/navigation_control/test/test1.png"

        if not self.use_test_image:
            self.cap = cv2.VideoCapture("/dev/sensors/webcam")
            #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3008)
            #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1504)
        
        # subscriber
        self.odom_sub = self.create_subscription(nav_msgs.Odometry,'/fusion/odom', self.get_odom, 1)
        
        # publisher
        self.roadside_pub = self.create_publisher(RoadsideInfo, "/roadside_info", 10)
        self.stop_pub = self.create_publisher(Bool, "/stop_line", 10)
        self.braille_block_pub = self.create_publisher(Bool, "/stop_braille_block", 10)

        # timer
        self.timer = self.create_timer(0.05, self.timer_callback)

        # parameter
        #self.pixel_to_meter = 0.00075 # 1504*1504
        self.pixel_to_meter = 0.0015 # 736*736
        self.detected = False
        self.distance = 0.0
        self.angle = 0.0

        #positon init odom
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        self.yaw = 0.0
        self.orientation_z = 0.0
        self.orientation_w = 0.0

        # position init stop line
        self.stop_line_registered = False
        self.stop_line_x = 0.0
        self.stop_line_y = 0.0

        # caribrate parameter
        #self.DIM=(1504, 1504)
        #self.K=np.array([[467.94972918063576, 0.0, 751.230452623187], [0.0, 468.05613091465483, 750.9019494139914], [0.0, 0.0, 1.0]])
        #self.D=np.array([[-0.014641575667383097], [-0.010755035156452033], [0.003932361337988872], [-0.0007419693374374312]])
        #self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2)
        self.DIM=(736, 736)
        self.K=np.array([[230.42134889608045, 0.0, 366.90152469610496], [0.0, 230.45799046281923, 367.1826387781316], [0.0, 0.0, 1.0]])
        self.D=np.array([[-0.02057316569139076], [0.0027174786060516474], [-0.0033945666290717286], [0.0005725592135935245]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2)

        # image
        #cv2.namedWindow("test")
        #cv2.setMouseCallback("test", self.mouse_callback)
        self.mouse_x = 0
        self.mouse_y = 0

        img_pts = np.array([
            [358,337], [361,279], [358,230], [364,149], [363, 87], [364, 34],
            [  0,223], [115,224], [232,225], [349,227], [461,230], [580,240],
            [  0,325], [100,324], [224,330], [356,334], [492,334], [624,338],
            [ 50,142], [153,149], [255,150], [360,155], [459,154], [570,153],
            [ 92, 76], [176, 77], [269, 79], [362, 82], [455, 86], [542, 82],
            [115, 28], [196, 29], [276, 34], [364, 31], [442, 32], [516, 31]], dtype=np.float32)

        world_pts = np.array([
            [0,30],
            [0,45],
            [0,61],
            [0,90],
            [0,120],
            [0,150],

            [-90,60],
            [-60,60],
            [-30,60],
            [0,60],
            [30,60],
            [60,60],

            [-90,30],
            [-60,30],
            [-30,30],
            [0,30],
            [30,30],
            [60,30],

            [-90,90],
            [-60,90],
            [-30,90],
            [0,90],
            [30,90],
            [60,90],

            [-90,120],
            [-60,120],
            [-30,120],
            [0,120],
            [30,120],
            [60,120],

            [-90,150],
            [-60,150],
            [-30,150],
            [0,150],
            [30,150],
            [60,150]
        ], dtype=np.float32)

        self.H, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC)
    
    def timer_callback(self):
        frame = self.get_camera_image()
        if frame is None:
            return
        self.detect_roadside(frame)
        self.detect_stop_line(frame)
        self.detect_braille_block(frame)
        cv2.waitKey(1)
    
    def get_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.theta_x = 0 #roll /math.pi*180
        self.theta_y = 0 #pitch /math.pi*180
        self.theta_z = yaw /math.pi*180
        self.yaw = yaw
        self.orientation_z = flio_q_z
        self.orientation_w = flio_q_w
    
    def get_camera_image(self):
        if self.use_test_image:
            frame = cv2.imread(self.test_image_path)
            if frame is None:
                self.get_logger().error("Cannot load test image.")
                return None
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Camera Error")
                return None
            frame = frame[:, :736]
            frame = self.undistort_image(frame)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"CLICK : ({x},{y})")
    
    def detect_roadside(self, frame):
        roi, mask = self.preprocess_roadside(frame)
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

        cv2.circle(
            roi,
            (self.mouse_x, self.mouse_y),
            5,
            (0,0,255),
            -1
        )

        cv2.putText(
            roi,
            f"({self.mouse_x},{self.mouse_y})",
            (10, roi.shape[0]-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        # show image
        cv2.imshow("Roadside", roi)
    
    def detect_stop_line(self, frame):
        roi, mask = self.preprocess_stopline(frame)
        detected, contour, cx, cy = self.extract_stop_line(mask)
        stop_line = False

        #############################################
        # Debug
        #############################################
        if detected:
            cv2.drawContours(roi, [contour], -1, (0,255,0), 3)
            #rect = cv2.minAreaRect(contour)
            #box = cv2.boxPoints(rect)
            #box = np.int32(box)
            #cv2.drawContours(roi, [box], 0, (255,0,0), 2)
            dx, dy = self.image_to_world(cx, cy)
            cv2.circle(roi, (int(cx),int(cy)), 5, (0,0,255), -1)
            yaw = self.yaw
            self.stop_line_x = (self.position_x + dx * np.cos(yaw) - dy * np.sin(yaw))
            self.stop_line_y = (self.position_y + dx * np.sin(yaw) + dy * np.cos(yaw))
            self.stop_line_registered = True
            self.get_logger().info(f"Stop line registered : ({self.stop_line_x:.2f}, {self.stop_line_y:.2f})")
        
        if self.stop_line_registered:
            distance = np.hypot(self.stop_line_x - self.position_x, self.stop_line_y - self.position_y)
            self.get_logger().info(f"Stop line distance : ({distance:.2f})")
            if distance < 0.5: # 50cm
                stop_line = True
                self.stop_line_registered = False
            cv2.putText(roi, f"{distance:.2f} m", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        
        #############################################
        # Publish
        #############################################
        msg = Bool()
        msg.data = stop_line
        self.stop_pub.publish(msg)

        # show image
        cv2.imshow("StopMask",mask)
        cv2.imshow("StopLine",roi)
    
    def detect_braille_block(self, frame):
        roi, mask = self.preprocess_braille_block(frame)
        detected, contour = self.extract_braille_block(mask)

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
        
        #############################################
        # Publish
        #############################################
        msg = Bool()
        msg.data = detected
        self.braille_block_pub.publish(msg)

        # show image
        cv2.imshow("braille_block Mask",mask)
        cv2.imshow("braille_block",roi)
    
    def preprocess_roadside(self, frame):
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
        #front = frame[:, :w//2]
        #front = frame[:, w//2:]
        #h, w = front.shape[:2]
        #cv2.imshow("Front", front)
        #cv2.imshow("Frame", frame)
        print(f"width = {w}, height = {h}")

        #roi = frame[int(h * 0.01):h, :]
        roi = frame[:int(h * 0.64), :]

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
    
    def preprocess_stopline(self, frame):

        h,w = frame.shape[:2]
        roi = frame[int(h * 0.3):h, :]
        #roi = frame[:int(h*0.7), :]

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        #############################################
        # White extraction
        #############################################

        lower = np.array([0,0,180])
        upper = np.array([180,25,255])

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        #############################################

        kernel = np.ones((9,9),np.uint8)

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
    
    def preprocess_braille_block(self, frame):

        h,w = frame.shape[:2]
        roi = frame[int(h * 0.3):h, :]
        #roi = frame[:int(h*0.8), :]

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        #############################################
        # Yellow extraction
        #############################################

        lower = np.array([20,80,80])
        upper = np.array([40,255,255])

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        #############################################

        kernel = np.ones((9,9),np.uint8)

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

    def undistort_image(self, frame):
        """
        Fish-eye画像を歪み補正する
        """
        if frame.shape[1::-1] != self.DIM:
            frame = cv2.resize(frame, self.DIM)

        undistorted = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

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

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
                boundary_points[y] = min( boundary_points[y], x)

        if len(boundary_points) < 20:
            return False, None, None, None

        ys = np.array(sorted(boundary_points.keys()), dtype=np.float32)
        xs = np.array([boundary_points[y] for y in ys], dtype=np.float32)
        # 境界点を描画
        for x, y in zip(xs, ys):
            cv2.circle(roi, (int(x), int(y)), 2, (0, 0, 255), -1)
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
        coef = np.polyfit( ys, xs, 2)

        #############################################
        # 画像下端
        #############################################
        y_bottom = roi.shape[0] - 1

        #############################################
        # Lookahead位置
        #############################################
        lookahead_pixel = 180     # 150～250くらいで調整
        y_predict = max(0, y_bottom - lookahead_pixel)
        x_bottom = (coef[0] * y_bottom**2 + coef[1] * y_bottom + coef[2])
        x_predict = (coef[0] * y_predict**2 + coef[1] * y_predict + coef[2])

        #############################################
        # 接線
        #
        # x = ay²+by+c
        #
        # dx/dy = 2ay+b
        #############################################
        dxdy = (2.0 * coef[0] * y_bottom + coef[1])
        boundary_angle = np.arctan(dxdy)
        #boundary_angle = np.arctan2(1.0, dxdy)

        #############################################
        # 境界までの距離(pixel)
        #############################################
        image_center = roi.shape[1] / 2
        distance_pixel = image_center - x_bottom
        predict_pixel = image_center - x_predict
        boundary_distance, forward_distance = self.image_to_world(x_predict, y_predict)
        '''
        predict_distance = (
            predict_pixel
            * self.pixel_to_meter
        )
        
        boundary_distance = (
            distance_pixel
            * self.pixel_to_meter
        )
        '''
        #############################################
        # デバッグ描画
        #############################################
        curve = []

        for y in range(int(min(ys)), int(max(ys))):
            x = (coef[0] * y**2 + coef[1] * y + coef[2])
            curve.append((int(x), int(y)))

        for i in range(len(curve)-1):
            cv2.line(roi, curve[i], curve[i+1], (0,255,255), 2)

        #############################################
        # 下端位置
        #############################################
        cv2.circle(roi, (int(x_bottom), int(y_bottom)), 6, (0,0,255), -1)
        cv2.circle(roi, (int(x_predict), int(y_predict)), 8, (255,0,255), -1)

        #############################################
        ys_draw = np.arange(int(min(ys)), roi.shape[0])
        xs_draw = np.polyval(coef, ys_draw)

        for x, y in zip(xs_draw, ys_draw):
            cv2.circle(roi, (int(x), int(y)), 1, (255, 0, 0), -1)

        return (True, boundary_distance, boundary_angle, x_bottom, coef)
    
    def extract_stop_line(self, mask):
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            print(area)
            #mennseki
            if area < 1000:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (w,h), angle = rect
            long_side = max(w,h)
            short_side = min(w,h)
            print(long_side)
            print(short_side)

            #########################################
            # Stop line condition
            #########################################
            if long_side < 250:
                continue

            if short_side > 60:
                continue

            return True, contour, cx, cy

        return False, None, None, None
        
    def extract_braille_block(self, mask):
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            print(area)
            #mennseki
            if area < 1000:
                continue

            rect = cv2.minAreaRect(contour)
            (_, _), (w,h), angle = rect
            long_side = max(w,h)
            short_side = min(w,h)
            print(long_side)
            print(short_side)

            return True, contour

        return False, None
    
    def image_to_world(self, x, y):
        p = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(p, self.H)
        return world[0,0,0] / 100.0, world[0,0,1] / 100.0

def main():
    rclpy.init()
    node = TsukubaController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()