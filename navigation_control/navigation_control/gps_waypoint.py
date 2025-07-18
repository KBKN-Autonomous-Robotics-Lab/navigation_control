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
from my_msgs.action import StopFlag  # Actionメッセージのインポート
from std_msgs.msg import Int32
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseWithCovariance

class GPSWaypointManager(Node):
    def __init__(self):
        super().__init__('gps_waypoint_manager')

        self.data = []
        self.start_time = None
        self.is_collecting = False
        self.waypoints = queue.Queue()
        self.count = 0
        self.theta = None
        self.declare_parameter('Position_magnification', 1.675)
        self.Position_magnification = self.get_parameter('Position_magnification').get_parameter_value().double_value

        self.avg_gps_service = self.create_service(Avglatlon, 'send_avg_gps', self.receive_avg_gps_callback)

        self.ref_points = [
            #(35.42578984, 139.3138073), # waypoint 1 nakaniwakokokara
            #(35.42580947, 139.3138761), # waypoint 2
            #(35.42582577, 139.3139183), # waypoint 3
            #(35.42584276, 139.3139622), # waypoint 1
            #(35.42585746, 139.3139984), # waypoint 2
            #(35.42589533, 139.3139987), # waypoint 3
            #(35.42596721, 139.3139898), # waypoint 4
            #(35.42596884, 139.3139395)  # waypoint 3 nakaniwakokomade
            #(35.4265706, 139.3141858), # waypoint 1 asupharutokokokara
            #(35.4266018, 139.3141984), # waypoint 2
            #(35.4266132, 139.314226),  # waypoint 3
            #(35.4266162, 139.3142614), # waypoint 4 asupharutokokomade
            #(35.426273, 139.3141582),  # waypoint 1 higasikan kokokara
            #(35.4262964, 139.3141756), # waypoint 2
            #(35.4262772, 139.3141948), # waypoint 3
            #(35.4262508, 139.3141906), # waypoint 4
            #(35.4262238, 139.314187),  # waypoint 5
            #(35.4262472, 139.3141576)  # waypoint 6 higasikan kokomade
            
            #(42.4007796,-83.1312014), # waypoint -1 reverse goal
            #(42.4009032,-83.1312044), # waypoint 0 qualification
            #(42.4009824,-83.1312026),  # waypoint 1
            #(42.400974,-83.1311006),   # waypoint 2
            #(42.4010064,-83.1310778),   # waypoint 2
            #(42.4009866,-83.1309344), # waypoint 3
            #(42.4009026,-83.1309332),  # waypoint 4
            #(42.4008312,-83.130932),  # waypoint 5
            #(42.400767,-83.1309272) # waypoint 6 goal qualification
            #(42.4007808,-83.1310664) #gave waypoint south
            #(42.4008624,-83.1310682) #gave waypoint mid
            #(42.4009362,-83.131076) #gave waypoint north
          
            #(42.4009746,-83.1316094),  # waypoint  0 autonav
            #(42.4009188,-83.1316196), # waypoint  1
            #(42.400875,-83.1316172), # waypoint  2
            #(42.4008048,-83.131616), # waypoint  3
            #(42.400758,-83.1315986), # waypoint  4
            #(42.400755,-83.1314978), # waypoint  5
            #(42.4007592,-83.1313388), # waypoint  6
            #(42.400836,-83.1313418), # waypoint 7
            #(42.40092,-83.1313454),   # waypoint 8 
            #(42.4009836,-83.131391), # waypoint  10
            #(42.4009842,-83.1314852), # waypoint  11
            #(42.4009782,-83.1316208),  # waypoint  12
            #(42.4009188,-83.1316196) # waypoint  13 goal autonav
            
            (42.4010028,-83.1299594), # waypoint  0 selfdrive
            (42.4010226,-83.1299276),  # waypoint  1
            (42.4010238,-83.1298562), # waypoint  2
            (42.4010058,-83.1298478),  # waypoint  3
            (42.400979,-83.129819), # waypoint  4
            (42.400923,-83.1298166), # waypoint  5
            (42.4009248,-83.1298454), # waypoint  6
            (42.4008294,-83.12984), # waypoint  7
            (42.400788,-83.1298406), # waypoint  8
            (42.4007982,-83.1299282),  # waypoint  9
            (42.400785,-83.129963), # waypoint 10
            (42.400791,-83.1300164), # waypoint 11
            (42.4007916,-83.1300782), # waypoint 12
            (42.4008414,-83.1300932), # waypoint 13
            (42.4008552,-83.1301148),   # waypoint 14
            (42.4009134,-83.1301196),  # waypoint 15
            (42.4009428,-83.1300956), # waypoint 16
            (42.4010052,-83.1300926), # waypoint 17
            (42.4010142,-83.1300254), # waypoint 18
            (42.4010208,-83.129969), # waypoint 19
            (42.4010544,-83.1299738), # waypoint 20
            (42.4010742,-83.129975)   # waypoint goal selfdrive
            #(42.4010934,-83.1299774)   # waypoint goal2 selfdrive
            #(42.4011054,83.1299798)   # waypoint goal3 yobi selfdrive           
        ]        
        self.selfdrive_point = np.array([
                                    [  7.033100176667772, 0.08274034687780049, 0.0], # waypoint  0 selfdrive
                                    [ 13.173118908065078,  -2.730432114149303, 0.0], # waypoint  1
                                    [  12.05675923470665,  -13.98312139287495, 0.0], # waypoint  2
                                    [  5.805107434278974, -15.141482523390598, 0.0], # waypoint  3
                                    [ 1.5629253772056577, -18.947534882827163, 0.0], # waypoint  4
                                    [ -6.474890741995338, -19.195749415863894, 0.0], # waypoint  5
                                    [ -9.823980594169749,  -16.21709637533306, 0.0], # waypoint  6
                                    [-21.099213573820954, -16.217088734954327, 0.0], # waypoint  7
                                    [ -27.68572091210563,  -16.13434404488772, 0.0], # waypoint  8
                                    [-28.243907283937006,   -8.10854194576664, 0.0], # waypoint  9
                                    [ -28.80208795226711,  1.7375446631183613, 0.0], # waypoint 10
                                    [-28.578814313348136,   7.777580943140201, 0.0], # waypoint 11
                                    [-27.797356645921774,  16.051603747084968, 0.0], # waypoint 12
                                    [-13.842875804579132,   18.53382131829375, 0.0], # waypoint 13
                                    [ -9.489063203498068,  21.760695953281353, 0.0], # waypoint 14
                                    [-1.1163428067660146,  22.257145472542515, 0.0], # waypoint 15
                                    [  1.786198429362834,  19.030275388330832, 0.0], # waypoint 16
                                    [ 10.828760671234289,  18.037398270328385, 0.0], # waypoint 17
                                    [  11.94511899957441,   10.67350681442107, 0.0], # waypoint 18
                                    [ 12.168388136731767,  3.3923549106952664, 0.0], # waypoint 19
                                    [   17.4153188087425,  1.1583653470352688, 0.0], # waypoint 20                                  
                                    [ 21.434249536317843,  1.4065867276276431, 0.0]  # waypoint goal selfdrive
                                    ])
        self.first_point = np.array([
                                    #[13.396391995138222, 0.49644283645666365, 0.0],
                                    #[28.132476063633337, 0.24822157204453668, 0.0],
                                    #[28.913951270729036, -18.53387791329175, 0.0],
                                    #[28.913994202519294, -36.736793721033145, 0.0],
                                    #[13.284813304930127, -36.902250675075656, 0.0],
                                    #[5.863191364919523e-05, -37.067711043789885, 0.0],
                                    #[-11.945016813956837, -37.72961562551752, 0.0]
                                    [30.117440234501547, 1.674507116857148, 0.0],
                                    [31.524010305139367, 12.056671542389434, 0.0],
                                    [31.193038945310082, 20.206096956195747, 0.0],
                                    [31.02754169887974, 33.26746847727089, 0.0],
                                    [28.628068059304322, 41.97503112166111, 0.0],
                                    [14.72773409252283, 42.53323288120902, 0.0],
                                    [-7.198387122031218, 41.75179344445025, 0.0],
                                    [-6.784690762796083, 27.462459157365217, 0.0],
                                    [-6.288254081807971, 11.833440117712478, 0.0],
                                    [-0.0, -0.0, 0.0],
                                    [12.99021557653992, -0.1116436130550457, 0.0]
                                    #[17.0804080112078, 0.8274040980095129, 0.0], # waypoint 0 qualification
                                    #[17.526959660260776, 9.763368540639547, 0.0], # waypoint 1
                                    #[16.41060157894189, 18.69933209559049, 0.0], # waypoint 2
                                    #[-0.1116226498476147, 17.954656032059653, 0.0], # waypoint 3
                                    #[-23.33192845492907, 16.630794290177384, 0.0], # waypoint 4
                                    #[-23.66684449903005, 8.356767162695558, 0.0], # waypoint 5
                                    #[-22.773761483667965, -0.41370135909791217, 0.0], # waypoint 6
                                    #[8.596013016246047, 0.3309615218752802, 0.0], # waypoint 7 goal qualification
                                    #[10.0, 0.0, 0.0],
                                    #[10.0, -10.0, 0.0],
                                    #[0.0, -10.0, 0.0],
                                    #[0.0, 0.0, 0.0]
                                    ]) # 開始点など not reverse
        self.last_point = np.array([[0.0, 0.0, 0.0]])   # 終了点など not reverse
        
        # selfdrive flag
        self.selfdrive_flag = 0 # autonav:0 selfdrive:1
        
        # Tkinter
        self.root = tk.Tk()
        self.root.title("GPS Waypoint Manager")
        self.root.bind("<Key>", self.key_input_handler)
        self.reversed_flag = False
        # ラベルの追加
        self.instruction_label = tk.Label(self.root, text='waypointを反転したい場合は"r"キーを押してください', font=('Helvetica', 14))
        self.instruction_label.pack(pady=10)

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, qos_profile)
        self.initial_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, qos_profile)
        self.human_sub = self.create_subscription(String, '/human_status', self.human_callback, 10) # lanechange 
        self.odom_sub = self.create_subscription(nav_msgs.Odometry, '/fusion/odom', self.get_odom, qos_profile)
        self.waypoint_pub = self.create_publisher(geometry_msgs.PoseArray, 'current_waypoint', qos_profile)
        self.waypoint_number_pub = self.create_publisher(Int32, 'waypoint_number', qos_profile)
        self.waypoint_path_publisher = self.create_publisher(nav_msgs.Path, 'waypoint_path', qos_profile) 
        self.timer = self.create_timer(0.1, self.waypoint_manager)

        self.current_waypoint = 0
        self.stop_flag = 0
        self.position_x = 0.0
        self.position_y = 0.0
        self.position_z = 0.0
        self.theta_x = 0.0
        self.theta_y = 0.0
        self.theta_z = 0.0
        self.waypoints_array = None
        self.waypoints_array = np.array([[100.0],[0.0],[0.0]])
        self.waypoint_range_set = 3.5
        self.waypoints_local_set = 0;
        self.previous_status = None
        self.determine_dist = 2.5 # waypoint range
        
        self.waypoints_initial_set = 0
        
        # Action
        self.action_client = ActionClient(self, StopFlag, 'stop_flag')  # ActionClient
        self.action_sent = False  
        self.stop = False # True=stop, False=go

    # Action
    def send_action_request(self):
        goal_msg = StopFlag.Goal()
        
        # stop変数の状態でaの値を決定
        if self.stop: # True
            goal_msg.a = 1  # stop
        else: # False
            goal_msg.a = 0  # go
            
        goal_msg.b = 2  # 任意の値を設定

        # アクションサーバーが利用可能になるまで待機
        self.action_client.wait_for_server()

        # アクションを非同期で送信
        self.future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.future.add_done_callback(self.response_callback)

    # フィードバックを受け取るコールバック関数
    def feedback_callback(self, feedback):
        self.get_logger().info(f"Received feedback: {feedback.feedback.rate}")

    # 結果を受け取るコールバック関数
    def response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")

        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.result_callback)

    # 結果のコールバック
    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.sum}")

    def key_input_handler(self, event):
        key = event.char.lower()
        if key == 'r':
            self.get_logger().info("キー入力: 'r' を受け取りました。waypointを反転します。")
            self.ref_points.reverse()
            self.reversed_flag = True
        elif key == 'a':
            self.get_logger().info("キー入力: 'a' を受け取りました。通常順で実行します。")
            self.reversed_flag = False
    

    def initial_pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        xyz = np.vstack((x,y,z))
        new_x = self.waypoints_array[0,:] - x
        new_y = self.waypoints_array[1,:] - y
        new_z = self.waypoints_array[2,:] - z
        new_xyz = np.vstack((new_x,new_y,new_z))
        
        if self.waypoints_initial_set == 0:
            self.waypoints_array = new_xyz;
            self.waypoints_initial_set = 1
        
        self.get_logger().info(f"new_xyz:{new_xyz}")    

    def goal_pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        
        xyz = np.vstack((x,y,z))
        waypoint_range = np.vstack((self.waypoint_range_set,0.0,0.0))
        yaw = self.orientation_to_yaw(qz, qw) *180/math.pi
        xyz_range , _ = rotation_xyz(waypoint_range, 0, 0, yaw)
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
            
        #self.waypoints_array = np.array([[xyz[0], xyz[1], xyz[2]], [next_xyz[0], next_xyz[1], next_xyz[2]]])
        #self.waypoints_array = np.array([[xyz[0], next_xyz[0]],[xyz[1], next_xyz[1]], [xyz[2], next_xyz[2]]])
        #self.waypoints_array = np.stack((xyz,next_xyz))

        # クォータニオン → ヨー角（Z軸の回転）に変換
        #angle = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
        
        #pose_array = self.current_waypoint_msg(self.waypoints_array[:, self.current_waypoint], 'map')
        #self.waypoint_pub.publish(pose_array)
        
        full_waypoints = np.concatenate([self.first_point], axis=0)
        self.waypoints_array = full_waypoints.T
        
        self.get_logger().info(f"Received goal: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} deg")    
        self.get_logger().info(f"self.waypoints_array:{self.waypoints_array}")    
        self.get_logger().info(f"xyz_range:{xyz_range}")    
    
    def human_callback(self, msg):
        human_status = msg.data
        
        # waypoint 0 and stop -> add current_waypoint number
        if self.current_waypoint == 0:
            if human_status == "Stop":
                if self.previous_status != "Stop":
                    self.get_logger().info("human detected")
                    self.current_waypoint += 1

        # 状態の更新
        self.previous_status = msg.data
    
    def orientation_to_yaw(self, z, w):
        yaw = np.arctan2(2.0 * (w * z), 1.0 - 2.0 * (z ** 2))
        return yaw
    
    def conversion(self, avg_lat, avg_lon, theta):
        #ido = self.ref_points[0]
        #keido = self.ref_points[1]
        ido0 = avg_lat
        keido0 = avg_lon

        self.get_logger().info(f"theta: {theta}")

        a = 6378137
        f = 35/10439
        e1 = 734/8971
        e2 = 127/1547
        n = 35/20843
        a0 = 1
        a2 = 102/40495
        a4 = 1/378280
        a6 = 1/289634371
        a8 = 1/204422462123
        pi180 = 71/4068
        
        points=[] # list
        
        for i, (ido, keido) in enumerate(self.ref_points):     
            # %math.pi/180
            d_ido = ido - ido0
            self.get_logger().info(f"d_ido: {d_ido}")
            d_keido = keido - keido0
            self.get_logger().info(f"d_keido: {d_keido}")
            rd_ido = d_ido * pi180
            rd_keido = d_keido * pi180
            r_ido = ido * pi180
            r_keido = keido * pi180
            r_ido0 = ido0 * pi180
            W = math.sqrt(1-(e1**2)*(math.sin(r_ido)**2))
            N = a / W
            t = math.tan(r_ido)
            ai = e2*math.cos(r_ido)

            # %===Y===%
            S = a*(a0*r_ido - a2*math.sin(2*r_ido)+a4*math.sin(4*r_ido) -
                   a6*math.sin(6*r_ido)+a8*math.sin(8*r_ido))/(1+n)
            S0 = a*(a0*r_ido0-a2*math.sin(2*r_ido0)+a4*math.sin(4*r_ido0) -
                    a6*math.sin(6*r_ido0)+a8*math.sin(8*r_ido0))/(1+n)
            m0 = S/S0
            B = S-S0
            y1 = (rd_keido**2)*N*math.sin(r_ido)*math.cos(r_ido)/2
            y2 = (rd_keido**4)*N*math.sin(r_ido) * \
                (math.cos(r_ido)**3)*(5-(t**2)+9*(ai**2)+4*(ai**4))/24
            y3 = (rd_keido**6)*N*math.sin(r_ido)*(math.cos(r_ido)**5) * \
                (61-58*(t**2)+(t**4)+270*(ai**2)-330*(ai**2)*(t**2))/720
            gps_y = self.Position_magnification * m0 * (B + y1 + y2 + y3)

            # %===X===%
            x1 = rd_keido*N*math.cos(r_ido)
            x2 = (rd_keido**3)*N*(math.cos(r_ido)**3)*(1-(t**2)+(ai**2))/6
            x3 = (rd_keido**5)*N*(math.cos(r_ido)**5) * \
                (5-18*(t**2)+(t**4)+14*(ai**2)-58*(ai**2)*(t**2))/120
            gps_x = self.Position_magnification * m0 * (x1 + x2 + x3)

            # point = (gps_x, gps_y)Not match

            degree_to_radian = math.pi / 180
            r_theta = theta * degree_to_radian
            h_x = math.cos(r_theta) * gps_x - math.sin(r_theta) * gps_y
            h_y = math.sin(r_theta) * gps_x + math.cos(r_theta) * gps_y
            point = np.array([h_y, -h_x, 0.0])
            #point = np.array([-h_y, h_x, 0.0])
            # point = (h_y, -h_x)
            self.get_logger().info(f"point: {point}")         
            points.append(point)

        return points

    def receive_avg_gps_callback(self, request, response):
        avg_lat, avg_lon, theta = request.avg_lat, request.avg_lon, request.theta
        if theta is None:
            self.get_logger().warn("GPSからのthetaがまだ取得されていません。")
            response.success = False
            return response

        GPSxy = self.conversion(avg_lat, avg_lon, theta)
        gps_np = np.array(GPSxy)
        if self.reversed_flag:
            self.first_point[:, 1] *= -1
            self.last_point[:, 1] *= -1

        if self.selfdrive_flag == 1:
            full_waypoints = np.concatenate([self.selfdrive_point], axis=0)
        else:
            full_waypoints = np.concatenate([gps_np], axis=0) #self.first_point, gps_np, self.last_point
        self.waypoints_array = full_waypoints.T
        response.success = True
        return response

    def get_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        x, y, z, w = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
        self.theta_x, self.theta_y, self.theta_z = 0, 0, yaw * 180 / math.pi

    def waypoint_manager(self):
        #self.get_logger().info(f"test: {self.current_waypoint}")
        #if self.waypoints_array is None or self.stop_flag:
        #    return
            
        position_x, position_y = self.position_x, self.position_y
        relative_x = self.waypoints_array[0, self.current_waypoint] - position_x
        relative_y = self.waypoints_array[1, self.current_waypoint] - position_y
        relative_point = np.vstack((relative_x, relative_y, self.waypoints_array[2, self.current_waypoint]))
        rotated_point, _ = rotation_xyz(relative_point, self.theta_x, self.theta_y, -self.theta_z)
        waypoint_rad = math.atan2(rotated_point[1], rotated_point[0])
        waypoint_dist = math.hypot(relative_x, relative_y)
        waypoint_theta = abs(waypoint_rad * 180 / math.pi)

        # determine_dist = 1.5 if abs(waypoint_theta) > 90 else 1.5
        determine_dist = self.determine_dist

        #check if the waypoint reached
        if waypoint_dist < determine_dist:
            #self.current_waypoint += 1
            if self.current_waypoint < len(self.waypoints_array[0,:])-1:
                self.current_waypoint += 1
            else:
                # goal:stopをtrueにしてアクションを再送信
                #a=1;
                self.stop = True
                self.get_logger().info("Stop flag reset to True")
                self.send_action_request()

        pose_array = self.current_waypoint_msg(self.waypoints_array[:, self.current_waypoint], 'map')
        self.waypoint_pub.publish(pose_array)
        self.waypoint_number_pub.publish(Int32(data=self.current_waypoint))
        waypoint_path = path_msg(self.waypoints_array, self.get_clock().now().to_msg(), 'odom')
        self.waypoint_path_publisher.publish(waypoint_path) 

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

def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    rad_x = math.radians(theta_x)
    rad_y = math.radians(theta_y)
    rad_z = math.radians(theta_z)
    rot_x = np.array([[ 1,               0,                0],
                      [ 0, math.cos(rad_x), -math.sin(rad_x)],
                      [ 0, math.sin(rad_x),  math.cos(rad_x)]])
    
    rot_y = np.array([[ math.cos(rad_y), 0,  math.sin(rad_y)],
                      [               0, 1,                0],
                      [-math.sin(rad_y), 0,  math.cos(rad_y)]])
    
    rot_z = np.array([[ math.cos(rad_z), -math.sin(rad_z), 0],
                      [ math.sin(rad_z),  math.cos(rad_z), 0],
                      [               0,                0, 1]])
    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    #print(f"rot_matrix ={rot_matrix}")
    #print(f"pointcloud ={pointcloud.shape}")
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
        
    # ウェイポイントを追加
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

