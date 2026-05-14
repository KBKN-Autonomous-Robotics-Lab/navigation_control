import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import numpy as np
import tkinter as tk
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import queue
from my_msgs.srv import Avglatlon
from geometry_msgs.msg import PoseStamped
import threading
from rclpy.time import Time
from rclpy.action import ActionClient
from my_msgs.action import StopFlag
from std_msgs.msg import Int32
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseWithCovariance
import os
import yaml
from visualization_msgs.msg import Marker, MarkerArray
import std_msgs.msg as std_msgs
import struct
from std_msgs.msg import Bool
import rclpy.duration

def excel_like_degmin(dd: float) -> float:
    """
    Excelでやっていた変換を再現
    10進度 dd.dddddd の小数部に 60 を掛けて、100 で割って度に足す

    例:
        35.709649 -> 35 + (0.709649 * 60) / 100 = 35.4257894

    注意:
        これは一般的な10進度のままではなく、
        以前Excelで使っていた「度.分っぽい表記」への変換です。
    """
    sign = -1.0 if dd < 0 else 1.0
    x = abs(dd)
    deg = int(x)
    frac = x - deg
    minutes = frac * 60.0
    return sign * (deg + minutes / 100.0)

class GPSWaypointManager(Node):
    def __init__(self):
        super().__init__('gps_waypoint_manager')

        self.data = []
        self.start_time = None
        self.is_collecting = False
        self.waypoints = queue.Queue()
        self.count = 0
        self.theta = None

        # set parameter (launch can change this parameter)
        self.declare_parameter('Position_magnification', 1.675)
        self.Position_magnification = self.get_parameter(
            'Position_magnification'
        ).get_parameter_value().double_value

        # Excelでやっていた変換を自動適用するか
        # True  : YAMLのgps_pointsには「変換前の元の10進度」をそのまま書けばOK
        # False : すでにExcel等で変換済みの値をYAMLに書いている場合
        self.declare_parameter('apply_excel_like_conversion', True)
        self.apply_excel_like_conversion = self.get_parameter(
            'apply_excel_like_conversion'
        ).get_parameter_value().bool_value

        self.declare_parameter('waypoint_start_index', 0)  # start waypoint number
        self.waypoint_start_index = self.get_parameter(
            'waypoint_start_index'
        ).get_parameter_value().integer_value

        self.declare_parameter('odom', '/fusion/odom')
        odom_topic = self.get_parameter('odom').get_parameter_value().string_value

        self.declare_parameter(
            'waypoint_path',
            'kbkn_maps/waypoints/hosei/2025/nakaniwa.yaml'
        )
        waypoint_path = self.get_parameter(
            'waypoint_path'
        ).get_parameter_value().string_value

        self.avg_gps_service = self.create_service(
            Avglatlon,
            'send_avg_gps',
            self.receive_avg_gps_callback
        )

        # Waypoint YAMLファイルを読み込む
        waypoint_map_yaml_path_name = waypoint_path
        waypoint_map_yaml_path_name_xy = "kbkn_maps/waypoints/hosei/2025/nakaniwa.yaml"
        py_path = "/home/ubuntu/ros2_ws/src/"
        waypoint_map_yaml_file_path = os.path.join(py_path, waypoint_map_yaml_path_name)
        waypoint_map_yaml_file_path_xy = os.path.join(py_path, waypoint_map_yaml_path_name_xy)

        with open(waypoint_map_yaml_file_path, 'r') as yaml_file:
            waypoint_map_yaml_data = yaml.safe_load(yaml_file)

        with open(waypoint_map_yaml_file_path_xy, 'r') as yaml_file_xy:
            waypoint_map_yaml_data_xy = yaml.safe_load(yaml_file_xy)

        # YAML の 'gps_points' をロード
        # 形式想定:
        # gps_points:
        #   - [lat, lon, offset_x, offset_y]
        yaml_points = waypoint_map_yaml_data.get('gps_points', [])
        yaml_points_xy = waypoint_map_yaml_data_xy.get('waypoints', [])

        self.gps_points = [point[:2] for point in yaml_points]
        self.offset_points = [point[2:4] for point in yaml_points]

        self.get_logger().info(f"Loaded {len(self.gps_points)} gps_points from YAML.")
        self.get_logger().info(
            f"apply_excel_like_conversion: {self.apply_excel_like_conversion}"
        )

        self.xy_points = [point[:3] for point in yaml_points_xy]

        self.xy_point = np.array([
            [0.0, 0.0, 0.0]
        ])
        self.first_point = np.array([[0.0, 0.0, 0.0]])  # 開始点など not reverse
        self.last_point = np.array([[0.0, 0.0, 0.0]])   # 終了点など not reverse

        # xy flag
        self.xy_flag = 0  # gps:0 xy:1

        # Tkinter
        self.root = tk.Tk()
        self.root.title("Waypoint Yaml")
        self.button = tk.Button(self.root, text="Waypoint Yaml", command=self.button_callback)
        self.button.pack()

        # timer
        self.initialize = None
        self.init_timer = self.create_timer(0.1, self.button_callback)

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_pose_callback, qos_profile
        )
        self.initial_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, qos_profile
        )
        self.human_sub = self.create_subscription(
            String, '/human_status', self.human_callback, 10
        )
        self.odom_sub = self.create_subscription(
            nav_msgs.Odometry, odom_topic, self.get_odom, qos_profile
        )

        self.waypoint_pub = self.create_publisher(
            geometry_msgs.PoseArray, 'current_waypoint', qos_profile
        )
        self.waypoint_number_pub = self.create_publisher(
            Int32, 'waypoint_number', qos_profile
        )
        self.waypoint_path_publisher = self.create_publisher(
            nav_msgs.Path, 'waypoint_path', qos_profile
        )
        ##tuika##
        self.goal_reached_publisher = self.create_publisher(
            Bool, '/goal_reached', 10
        ) 
        self.init_gps_pub = self.create_publisher(
    NavSatFix,
    '/init_gps',
    10
        )
        self.goal_publisher = False
        

        self.manager_timer = self.create_timer(0.1, self.waypoint_manager)
        
        # Marker publisher
        self.marker_pub = self.create_publisher(Marker, 'waypoint_markers', qos_profile)
        self.label_marker_pub = self.create_publisher(MarkerArray, 'waypoint_labels', qos_profile)
        # rviz再接続対策：1Hzで再publish（確定済みの場合のみ）
        self.marker_timer = self.create_timer(1.0, self.republish_markers_if_ready)

        self.current_waypoint = 0
        self.stop_flag = 0
        self.position_x = 0.0
        self.position_y = 0.0
        self.position_z = 0.0
        self.theta_x = 0.0
        self.theta_y = 0.0
        self.theta_z = 0.0
        self.waypoints_array = np.array([[100.0], [0.0], [0.0]])
        self.waypoint_range_set = 3.5
        self.waypoints_local_set = 0
        self.previous_status = None
        self.determine_dist = 4.5 # waypoint range
        self.waypoints_initial_set = 0

        # waypointが確定したらTrueにする（それまでmarkerは出さない）
        self.waypoints_ready = False
        
        # goal flag
        self.goal_published = False
        
        # Action
        self.action_client = ActionClient(self, StopFlag, 'stop_flag')  # ActionClient
        self.action_sent = False  
        self.stop = False # True=stop, False=go

        self.waypoints_initial_set = 0

        # Action
        self.action_client = ActionClient(self, StopFlag, 'stop_flag')
        self.action_sent = False
        self.stop = False  # True=stop, False=go
    ######tuika###
    def publish_goal_reached(self):
        msg = Bool()
        msg.data = True
        self.goal_reached_publisher.publish(msg)
        self.get_logger().info('publishd /goal_reached = True')
     ########   
    # Action
    def send_action_request(self):
        goal_msg = StopFlag.Goal()

        if self.stop:
            goal_msg.a = 1  # stop
        else:
            goal_msg.a = 0  # go

        goal_msg.b = 2

        self.action_client.wait_for_server()
        self.future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self.future.add_done_callback(self.response_callback)

    def feedback_callback(self, feedback):
        self.get_logger().info(f"Received feedback: {feedback.feedback.rate}")

    def response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.sum}")

    def initial_pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        xyz = np.vstack((x, y, z))
        new_x = self.waypoints_array[0, :] - x
        new_y = self.waypoints_array[1, :] - y
        new_z = self.waypoints_array[2, :] - z
        new_xyz = np.vstack((new_x, new_y, new_z))

        if self.waypoints_initial_set == 0:
            self.waypoints_array = new_xyz
            self.waypoints_initial_set = 1

        self.get_logger().info(f"new_xyz:{new_xyz}")

    def goal_pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        xyz = np.vstack((x, y, z))
        waypoint_range = np.vstack((self.waypoint_range_set, 0.0, 0.0))
        yaw = self.orientation_to_yaw(qz, qw) * 180 / math.pi
        xyz_range, _ = rotation_xyz(waypoint_range, 0, 0, yaw)
        next_x = xyz[0] + xyz_range[0]
        next_y = xyz[1] + xyz_range[1]
        next_z = xyz[2] + xyz_range[2]
        next_xyz = np.vstack((next_x,next_y,next_z))
        
        if self.waypoints_local_set == 0:
            self.current_waypoint = 0;
            self.waypoints_array = xyz;
            self.waypoints_local_set = 1;
        else:
            self.waypoints_array = np.insert(self.waypoints_array, len(self.waypoints_array[0,:]), xyz.T, axis=1)

        #full_waypoints = np.concatenate([self.xy_points], axis=0)
        #self.waypoints_array = full_waypoints.T
        self.current_waypoint = self.waypoint_start_index
        self.get_logger().info(f"Start index set: {self.current_waypoint}")

        # waypointが確定したのでmarkerを表示
        self.waypoints_ready = True
        self.publish_waypoint_markers()
        
        self.get_logger().info(f"Received goal: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} deg")    
        self.get_logger().info(f"self.waypoints_array:{self.waypoints_array}")    
        self.get_logger().info(f"xyz_range:{xyz_range}")    
    
    def human_callback(self, msg):
        human_status = msg.data

        if self.current_waypoint == 0:
            if human_status == "Stop":
                if self.previous_status != "Stop":
                    self.get_logger().info("human detected")
                    self.current_waypoint += 1

        self.previous_status = msg.data

    def orientation_to_yaw(self, z, w):
        yaw = np.arctan2(2.0 * (w * z), 1.0 - 2.0 * (z ** 2))
        return yaw

    def conversion(self, avg_lat, avg_lon, theta):
        ido0 = avg_lat
        keido0 = avg_lon

        # Excelでやっていた変換を、基準点にも自動適用
        if self.apply_excel_like_conversion:
            ido0 = excel_like_degmin(ido0)
            keido0 = excel_like_degmin(keido0)

        self.get_logger().info(f"theta: {theta}")
        self.get_logger().info(f"base_lat: {ido0}, base_lon: {keido0}")

        a = 6378137
        f = 35 / 10439
        e1 = 734 / 8971
        e2 = 127 / 1547
        n = 35 / 20843
        a0 = 1
        a2 = 102 / 40495
        a4 = 1 / 378280
        a6 = 1 / 289634371
        a8 = 1 / 204422462123
        pi180 = 71 / 4068

        points = []

        for i, (ido, keido) in enumerate(self.gps_points):
            # Excelでやっていた変換を、各 waypoint 側にも自動適用
            if self.apply_excel_like_conversion:
                ido = excel_like_degmin(ido)
                keido = excel_like_degmin(keido)

            d_ido = ido - ido0
            d_keido = keido - keido0
            rd_ido = d_ido * pi180
            rd_keido = d_keido * pi180
            r_ido = ido * pi180
            r_keido = keido * pi180
            r_ido0 = ido0 * pi180
            W = math.sqrt(1 - (e1 ** 2) * (math.sin(r_ido) ** 2))
            N = a / W
            t = math.tan(r_ido)
            ai = e2 * math.cos(r_ido)

            # %===Y===%
            S = a * (
                a0 * r_ido
                - a2 * math.sin(2 * r_ido)
                + a4 * math.sin(4 * r_ido)
                - a6 * math.sin(6 * r_ido)
                + a8 * math.sin(8 * r_ido)
            ) / (1 + n)

            S0 = a * (
                a0 * r_ido0
                - a2 * math.sin(2 * r_ido0)
                + a4 * math.sin(4 * r_ido0)
                - a6 * math.sin(6 * r_ido0)
                + a8 * math.sin(8 * r_ido0)
            ) / (1 + n)

            m0 = S / S0
            B = S - S0
            y1 = (rd_keido ** 2) * N * math.sin(r_ido) * math.cos(r_ido) / 2
            y2 = (rd_keido ** 4) * N * math.sin(r_ido) * (math.cos(r_ido) ** 3) * (
                5 - (t ** 2) + 9 * (ai ** 2) + 4 * (ai ** 4)
            ) / 24
            y3 = (rd_keido ** 6) * N * math.sin(r_ido) * (math.cos(r_ido) ** 5) * (
                61 - 58 * (t ** 2) + (t ** 4) + 270 * (ai ** 2) - 330 * (ai ** 2) * (t ** 2)
            ) / 720
            gps_y = self.Position_magnification * m0 * (B + y1 + y2 + y3)

            # %===X===%
            x1 = rd_keido * N * math.cos(r_ido)
            x2 = (rd_keido ** 3) * N * (math.cos(r_ido) ** 3) * (
                1 - (t ** 2) + (ai ** 2)
            ) / 6
            x3 = (rd_keido ** 5) * N * (math.cos(r_ido) ** 5) * (
                5 - 18 * (t ** 2) + (t ** 4) + 14 * (ai ** 2) - 58 * (ai ** 2) * (t ** 2)
            ) / 120
            gps_x = self.Position_magnification * m0 * (x1 + x2 + x3)

            degree_to_radian = math.pi / 180
            r_theta = theta * degree_to_radian

            # 元コードのoffset処理をそのまま維持
            h_x = math.cos(r_theta) * gps_x - math.sin(r_theta) * gps_y - self.offset_points[i][1]
            h_y = math.sin(r_theta) * gps_x + math.cos(r_theta) * gps_y + self.offset_points[i][0]
            point = np.array([h_y, -h_x, 0.0])
            points.append(point)

        return points

    def button_callback(self):
        if self.initialize is None:
            gps_points = np.array(self.gps_points)
            init_lat = gps_points[0, 0]
            init_lon = gps_points[0, 1]
            theta = 0.0
            
            converted_lat = excel_like_degmin(float(init_lat))
            converted_lon = excel_like_degmin(float(init_lon))
            
            msg = NavSatFix()
            msg.latitude = converted_lat
            msg.longitude = converted_lon
            msg.altitude = 0.0

            self.init_gps_pub.publish(msg)

            self.get_logger().info(
                f"Published /init_gps:"
                f"raw_lat={init_lat}, raw_lon={init_lon},"
                f"converted_lat={converted_lat}, converted_lon={converted_lon}"
            )
                


            print("test")
            print(init_lat)
            print(init_lon)

            GPSxy = self.conversion(init_lat, init_lon, theta)
            gps_np = np.array(GPSxy)

            if self.xy_flag == 1:
                full_waypoints = np.concatenate([self.xy_point], axis=0)
            else:
                full_waypoints = np.concatenate([gps_np], axis=0)

            self.waypoints_array = full_waypoints.T
            self.get_logger().info(f"Start waypoints_array: {self.waypoints_array}")

            self.current_waypoint = self.waypoint_start_index
            self.get_logger().info(f"Start index set: {self.current_waypoint}")
            
            # waypointが確定したのでmarkerを表示
            self.waypoints_ready = True
            self.publish_waypoint_markers()

            self.initialize = True

    def receive_avg_gps_callback(self, request, response):
        avg_lat, avg_lon, theta = request.avg_lat, request.avg_lon, request.theta

        if theta is None:
            self.get_logger().warn("GPSからのthetaがまだ取得されていません。")
            response.success = False
            return response

        gps_points = np.array(self.gps_points)
        init_lat = gps_points[0,0]
        init_lon = gps_points[0,1]
        print(init_lat)
        print(init_lon)
        GPSxy = self.conversion(init_lat, init_lon, theta)
        gps_np = np.array(GPSxy)

        if self.xy_flag == 1:
            full_waypoints = np.concatenate([self.xy_point], axis=0)
        else:
            full_waypoints = np.concatenate([gps_np], axis=0)

        self.waypoints_array = full_waypoints.T
        self.current_waypoint = self.waypoint_start_index
        self.get_logger().info(f"Start index set: {self.current_waypoint}")

        # waypointが確定したのでmarkerを表示
        self.waypoints_ready = True
        self.publish_waypoint_markers()

        response.success = True
        return response

    def get_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
        self.theta_x = 0
        self.theta_y = 0
        self.theta_z = yaw * 180 / math.pi

    def waypoint_manager(self):
        position_x, position_y = self.position_x, self.position_y

        relative_x = self.waypoints_array[0, self.current_waypoint] - position_x
        relative_y = self.waypoints_array[1, self.current_waypoint] - position_y
        relative_point = np.vstack((
            relative_x,
            relative_y,
            self.waypoints_array[2, self.current_waypoint]
        ))

        rotated_point, _ = rotation_xyz(
            relative_point,
            self.theta_x,
            self.theta_y,
            -self.theta_z
        )

        waypoint_rad = math.atan2(rotated_point[1], rotated_point[0])
        waypoint_dist = math.hypot(relative_x, relative_y)
        waypoint_theta = abs(waypoint_rad * 180 / math.pi)

        if 194 <= self.current_waypoint <= 194:
            determine_dist = 8.5
        elif 153 <= self.current_waypoint <= 155:
            determine_dist = 6.5
        else:
            determine_dist = self.determine_dist
###tuika####
        self.get_logger().info(
            f"current={self.current_waypoint}, total={len(self.waypoints_array[0, :])},"
            f"dist={waypoint_dist:3f}, thresh={determine_dist:3f}"
        )
        ####   
        if waypoint_dist < determine_dist:
            if self.current_waypoint < len(self.waypoints_array[0, :]) - 1:
                self.current_waypoint += 1
            else:
            ##tuika##
                self.get_logger().info("Entered final waypoint branch")
                if not self.goal_published:
                    self.publish_goal_reached()
                    self.goal_published = True
            #####        
                self.stop = True
                self.get_logger().info("Stop flag reset to True")
                self.send_action_request()

        pose_array = self.current_waypoint_msg(
            self.waypoints_array[:, self.current_waypoint],
            'map'
        )
        self.waypoint_pub.publish(pose_array)
        self.waypoint_number_pub.publish(Int32(data=self.current_waypoint))

        waypoint_path = path_msg(
            self.waypoints_array,
            self.get_clock().now().to_msg(),
            'odom'
        )
        self.waypoint_path_publisher.publish(waypoint_path)

        try:
            self.publish_waypoint_markers()
        except Exception as e:
            self.get_logger().warn(f"publish_waypoint_markers error: {e}")

    def current_waypoint_msg(self, waypoint, set_frame_id):
        pose_array = geometry_msgs.PoseArray()
        pose_array.header.frame_id = set_frame_id
        pose_array.header.stamp = self.get_clock().now().to_msg()

        pose = geometry_msgs.Pose()
        pose.position.x = float(waypoint[0])
        pose.position.y = float(waypoint[1])
        pose.position.z = float(waypoint[2])
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)

        return pose_array

    def run(self):
        self.root.mainloop()

    def publish_waypoint_markers(self):
        if self.waypoints_array is None:
            return
        
        npts = self.waypoints_array.shape[1]  # waypointの総数
        now = self.get_clock().now().to_msg()

        # ① SPHERE_LIST：全waypointを球で表示
        # SPHERE_LISTは「1つのMarkerメッセージで複数の球をまとめて送れる」型
        sphere = Marker()
        sphere.header.frame_id = 'odom'
        sphere.header.stamp = now
        sphere.ns = 'waypoints'
        sphere.id = 0
        sphere.type = Marker.SPHERE_LIST
        sphere.action = Marker.ADD
        sphere.scale.x = 0.4          # 球の直径[m]
        sphere.scale.y = 0.4
        sphere.scale.z = 0.4
        sphere.color.r = 0.0
        sphere.color.g = 1.0          # 緑色
        sphere.color.b = 0.0
        sphere.color.a = 0.9          # 透明度
        # lifetime=0で明示的に消すまで永続表示
        sphere.lifetime = rclpy.duration.Duration(seconds=0).to_msg()

        for i in range(npts):
            p = geometry_msgs.Point()
            p.x = float(self.waypoints_array[0, i])
            p.y = float(self.waypoints_array[1, i])
            p.z = 0.0
            sphere.points.append(p)
        
        self.marker_pub.publish(sphere)

        # ② TEXT_VIEW_FACING：各waypointの上に番号を表示
        # MarkerArrayは「複数のMarkerをまとめて1トピックで送る」型
        label_array = MarkerArray()

        for i in range(npts):
            label = Marker()
            label.header.frame_id = 'odom'
            label.header.stamp = now
            label.ns = 'waypoint_labels'
            label.id = i                          # 各テキストに固有ID
            label.type = Marker.TEXT_VIEW_FACING  # 常にカメラ方向を向くテキスト
            label.action = Marker.ADD
            label.pose.position.x = float(self.waypoints_array[0, i])
            label.pose.position.y = float(self.waypoints_array[1, i])
            label.pose.position.z = 0.8           # 球の少し上に表示
            label.scale.z = 0.5                   # テキストの高さ[m]
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0                   # 白色
            label.color.a = 1.0
            label.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            label.text = str(i)                   # waypoint番号を文字列で
            label_array.markers.append(label)

        self.label_marker_pub.publish(label_array)

    def republish_markers_if_ready(self):
        # waypointが確定していなければ何もしない
        # 確定していれば1Hzで再publishしてrviz再接続に備える
        if not self.waypoints_ready:
            return
        self.publish_waypoint_markers()

def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    rad_x = math.radians(theta_x)
    rad_y = math.radians(theta_y)
    rad_z = math.radians(theta_z)

    rot_x = np.array([
        [1, 0, 0],
        [0, math.cos(rad_x), -math.sin(rad_x)],
        [0, math.sin(rad_x),  math.cos(rad_x)]
    ])

    rot_y = np.array([
        [math.cos(rad_y), 0, math.sin(rad_y)],
        [0, 1, 0],
        [-math.sin(rad_y), 0, math.cos(rad_y)]
    ])

    rot_z = np.array([
        [math.cos(rad_z), -math.sin(rad_z), 0],
        [math.sin(rad_z),  math.cos(rad_z), 0],
        [0, 0, 1]
    ])

    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    rot_pointcloud = rot_matrix.dot(pointcloud)
    return rot_pointcloud, rot_matrix


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def path_msg(waypoints, stamp, parent_frame):
    wp_msg = nav_msgs.Path()
    wp_msg.header.frame_id = parent_frame
    wp_msg.header.stamp = stamp

    for i in range(waypoints.shape[1]):
        waypoint = geometry_msgs.PoseStamped()
        waypoint.header.frame_id = parent_frame
        waypoint.header.stamp = stamp
        waypoint.pose.position.x = waypoints[0, i]
        waypoint.pose.position.y = waypoints[1, i]
        waypoint.pose.position.z = 0.0
        waypoint.pose.orientation.w = 1.0
        wp_msg.poses.append(waypoint)

    return wp_msg


def main(args=None):
    rclpy.init(args=args)
    node = GPSWaypointManager()
    rclpy_thread = threading.Thread(target=rclpy.spin, args=(node,))
    rclpy_thread.start()
    node.run()
    node.destroy_node()
    rclpy.shutdown()
    rclpy_thread.join()


if __name__ == '__main__':
    main()
