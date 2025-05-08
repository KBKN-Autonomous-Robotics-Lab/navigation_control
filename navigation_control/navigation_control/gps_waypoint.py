# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
import numpy as np
import time
import threading
from geopy.distance import geodesic  
import tkinter as tk
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import yaml
import os
import queue
from orange_gnss.get_lonlat_movingbase_quat_ttyUSB import GPSData
from my_msgs.srv import Avglatlon

class GPSAverageNode(Node):
    def __init__(self):
        super().__init__('gps_average_node')
        #self.subscription = self.create_subscription(NavSatFix, '/fix', self.gps_callback, 10)
        self.movingase_sub = self.create_subscription(Imu, "movingbase/quat", self.movingbase_callback, 1)

        self.data = []
        self.start_time = None
        self.is_collecting = False
        self.waypoints = queue.Queue()  
        self.theta = None
        self.count = 0
        self.declare_parameter('Position_magnification', 1.675)
        self.Position_magnification = self.get_parameter('Position_magnification').get_parameter_value().double_value
        
        # service
        self.avg_gps_service = self.create_service(Avglatlon, 'send_avg_gps', self.receive_avg_gps_callback)
        
        # define waypoints
        self.ref_points = [
            (35.42578984, 139.3138073), # waypoint 1
            (35.42580947, 139.3138761), # waypoint 2
            (35.42582577, 139.3139183), # waypoint 3
            (35.42584276, 139.3139622), # waypoint 1
            (35.42585746, 139.3139984), # waypoint 2
            (35.42589533, 139.3139987), # waypoint 3
            (35.42596721, 139.3139898), # waypoint 4
            (35.42596884, 139.3139395) # waypoint 3
        ]        
        self.first_point = np.array([[0.0, 0.0, 0.0],
                                    [5.0, 0.0, 0.0],
                                    [10.0, 0.0, 0.0]])  # 開始点など                        
        self.last_point = np.array([[0.0, 0.0, 0.0]])   # 終了点など
        
        # Tkinter
        self.root = tk.Tk()
        self.root.bind("<Key>", self.key_input_handler)
        self.reversed_flag = False  # bが入力されたかどうかを記録
        
    def key_input_handler(self, event):
        key = event.char.lower()
        if key == 'b':
            self.get_logger().info("キー入力: 'b' を受け取りました。ref_pointsを反転します。")
            self.ref_points.reverse()
            self.reversed_flag = True
        elif key == 'a':
            self.get_logger().info("キー入力: 'a' を受け取りました。通常順で実行します。")
            self.reversed_flag = False
    
    def movingbase_callback(self, msg):
        if self.count == 0:
            self.theta = msg.orientation_covariance[0]
            self.count = 1
    
    # copy lonlat_to_odom function 
    '''
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
     '''
            
    def receive_avg_gps_callback(self, request, response):
        avg_lat = request.avg_lat
        avg_lon = request.avg_lon

        self.get_logger().info(f"サービス受信: 平均緯度={avg_lat}, 経度={avg_lon}")

        if self.theta is None:
            self.get_logger().warn("GPSからのthetaがまだ取得されていません。")
            response.success = False
            return response

        GPSxy = GPSData.conversion(avg_lat, avg_lon, self.theta)
        #GPSxy = self.conversion(avg_lat, avg_lon, self.theta)

        gps_np = np.array(GPSxy)

        first_point = self.first_point
        last_point = self.last_point

        if self.reversed_flag:
            first_point[:, 1] *= -1
            last_point[:, 1] *= -1

        full_waypoints = np.concatenate([first_point, gps_np, last_point], axis=0)
        self.waypoints.put(full_waypoints.T)

        self.launch_waypoint_manager()

        response.success = True
        return response

            
    def launch_waypoint_manager(self):
        self.get_logger().info("WaypointManager を起動します。")
        # キューからリストに変換し、WaypointManager に渡す
        #waypoints_list = []
        while not self.waypoints.empty():
            #waypoints_list.append(self.waypoints.get())
            waypoints_array = self.waypoints.get()

        #waypoints_array = np.array(waypoints_list).T  # 形状を [3, N] に変換

        waypoint_manager = WaypointManager(waypoints_array)
        rclpy.spin(waypoint_manager)
        waypoint_manager.destroy_node()

    def run(self):
        self.root.mainloop()

# C++と同じく、Node型を継承します。
class WaypointManager(Node):
    # コンストラクタです、PcdRotationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self, waypoints_array):
        # 継承元のクラスを初期化します。
        super().__init__('waypoint_manager_node')
        
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 10
        )
        
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 10
        )
        
        # Subscriptionを作成。
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_wheel', self.get_odom, qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # タイマーを0.1秒（100ミリ秒）ごとに呼び出す
        self.timer = self.create_timer(0.1, self.waypoint_manager)
        
        # Publisherを作成
        self.current_waypoint_publisher = self.create_publisher(geometry_msgs.PoseArray, 'current_waypoint', qos_profile) #set publish pcd topic name
        
        #パラメータ
        #waypoint init
        self.current_waypoint = 0
        self.stop_flag = 0
        
        #positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        
        self.waypoints = waypoints_array
        print(f"self.waypoints ={self.waypoints}")
        
    def waypoint_manager(self):
        self.time_stamp = self.get_clock().now().to_msg()
        #self.get_logger().info('waypoint manager cntl')
        position_x=self.position_x; position_y=self.position_y; 
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        
        #waypoint theta & dist
        relative_point_x = self.waypoints[0,self.current_waypoint] - position_x
        relative_point_y = self.waypoints[1,self.current_waypoint] - position_y
        relative_point = np.vstack((relative_point_x, relative_point_y, self.waypoints[2,self.current_waypoint]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
        waypoint_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
        waypoint_dist = math.sqrt(relative_point_x**2 + relative_point_y**2)
        waypoint_theta = abs(waypoint_rad * (180 / math.pi))
        
        #set judge dist
        if abs(waypoint_theta) > 90:
            determine_dist = 8
        else:
            determine_dist = 3
        #check if the waypoint reached
        if waypoint_dist < determine_dist:
            self.current_waypoint += 1
            if self.current_waypoint > (len(self.waypoints[0,:]) - 1):
                self.stop_flag = 1
                self.get_logger().info('GOAL : stop_flag = %f' % (stop_flag))
        self.get_logger().info('current_waypoint:x = %f, y = %f : waypoint_no = %f' % (self.waypoints[0,self.current_waypoint], self.waypoints[1,self.current_waypoint], self.current_waypoint))
        
        #publish
        pose_array = self.current_waypoint_msg(self.waypoints[:,self.current_waypoint], 'map')
        self.current_waypoint_publisher.publish(pose_array)
        
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
        
    def current_waypoint_msg(self, waypoint, set_frame_id):
        pose_array = geometry_msgs.PoseArray()
        pose_array.header.frame_id = set_frame_id
        pose_array.header.stamp = self.time_stamp
        pose = geometry_msgs.Pose()
        pose.position.x = float(waypoint[0])
        pose.position.y = float(waypoint[1])
        pose.position.z = float(waypoint[2])
        
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)
        
        return pose_array

def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])
    
    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])
    
    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])
    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    #print(f"rot_matrix ={rot_matrix}")
    #print(f"pointcloud ={pointcloud.shape}")
    rot_pointcloud = rot_matrix.dot(pointcloud)
    return rot_pointcloud, rot_matrix
    
def quaternion_to_euler(x, y, z, w):
    # クォータニオンから回転行列を計算
    rot_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])

    # 回転行列からオイラー角を抽出
    roll = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
    pitch = np.arctan2(-rot_matrix[2, 0], np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2))
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return roll, pitch, yaw
        

def main(args=None):
    rclpy.init(args=args)
    node = GPSAverageNode()
    rclpy_thread = threading.Thread(target=rclpy.spin, args=(node,))
    rclpy_thread.start()
    node.run()  
    node.destroy_node()
    rclpy.shutdown()
    rclpy_thread.join()

if __name__ == '__main__':
    main()

