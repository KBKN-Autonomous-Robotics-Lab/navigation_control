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
import yaml

class GPSWaypointManager(Node):
    def __init__(self):
        super().__init__('gps_waypoint_manager')

        self.data = []
        self.start_time = None
        self.is_collecting = False
        self.count = 0
        self.theta = None
        self.declare_parameter('Position_magnification', 1.675)
        self.Position_magnification = self.get_parameter('Position_magnification').get_parameter_value().double_value

        self.init_lat = 42.4008888
        self.init_lon = -83.5310724
        self.init_theta = 0.0
        
        self.ref_points = [
            #(35.42578984, 139.3138073), # waypoint 1 nakaniwakokokara
            #(35.42580947, 139.3138761), # waypoint 2
            #(35.42582577, 139.3139183), # waypoint 3
            #(35.42584276, 139.3139622), # waypoint 1
            #(35.42585746, 139.3139984), # waypoint 2
            #(35.42589533, 139.3139987), # waypoint 3
            #(35.42596721, 139.3139898), # waypoint 4
            #(35.42596884, 139.3139395)  # waypoint 3 nakaniwakokomade
            #(35.4265706, 139.3141858),  # waypoint 1 asupharutokokokara
            #(35.4266018, 139.3141984),  # waypoint 2
            #(35.4266132, 139.314226),   # waypoint 3
            #(35.4266162, 139.3142614),  # waypoint 4 asupharutokokomade
            #(35.426273, 139.3141582),   # waypoint 1 higasikan kokokara
            #(35.4262964, 139.3141756),  # waypoint 2
            #(35.4262772, 139.3141948),  # waypoint 3
            #(35.4262508, 139.3141906),  # waypoint 4
            #(35.4262238, 139.314187),   # waypoint 5
            #(35.4262472, 139.3141576)   # waypoint 6 higasikan kokomade
            
            #(42.4009488, -83.5314714), # waypoint  0 selfdrive
            #(42.4009818, -83.531451),  # waypoint  1
            #(42.4009758, -83.5313694), # waypoint  2
            #(42.4009422, -83.531361),  # waypoint  3
            #(42.4009194, -83.5313334), # waypoint  4
            #(42.4008762, -83.5313316), # waypoint  5
            #(42.4008582, -83.5313532), # waypoint  6
            #(42.4007976, -83.5313532), # waypoint  7
            #(42.4007622, -83.5313538), # waypoint  8
            #(42.4007592, -83.531412),  # waypoint  9
            #(42.4007562, -83.5314834), # waypoint 10
            #(42.4007574, -83.5315272), # waypoint 11
            #(42.4007616, -83.5315872), # waypoint 12
            #(42.4008366, -83.5316052), # waypoint 13
            #(42.40086, -83.5316286),   # waypoint 14
            #(42.400905, -83.5316322),  # waypoint 15
            #(42.4009206, -83.5316088), # waypoint 16
            #(42.4009692, -83.5316016), # waypoint 17
            #(42.4009752, -83.5315482), # waypoint 18
            #(42.4009764, -83.5314954), # waypoint 19
            #(42.4010046, -83.5314792), # waypoint 20
            #(42.4010262, -83.531481)   # waypoint goal selfdrive
            
            #(42.4009188,-83.531346),  # waypoint  0 autonav yobi
            #(42.40098,-83.5313508),   # waypoint  1
            #(42.4009788,-83.5314744), # waypoint  2
            #(42.400977,-83.5316148),  # waypoint  3
            #(0), # waypoint  4
            #(0), # waypoint  5
            #(0), # waypoint  6
            #(0), # waypoint  7
            #(42.4007526,-83.5316082), # waypoint  8
            #(42.4007562,-83.531475),  # waypoint  9
            #(42.4007628,-83.5313478), # waypoint 10
            #(42.4008246,-83.5313418), # waypoint 11
            #(42.4009152,-83.5313436)  # waypoint 12 goal autonav yobi
            
            (42.4009806,-83.5310784), # waypoint 0 qualification
            (42.400983,-83.5311432),  # waypoint 1
            (42.400977,-83.531208),   # waypoint 2
            (42.4008882,-83.5312026), # waypoint 3
            (42.4007634,-83.531193),  # waypoint 4
            (42.4007616,-83.531133),  # waypoint 5
            (42.4007664,-83.5310694), # waypoint 6
            (42.400935,-83.5310748)   # waypoint 7 goal qualification
        ]        
        self.first_point = np.array([[0.0, 0.0, 0.0],
                                    #[10.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0]])  # 開始点など                        
        self.last_point = np.array([[0.0, 0.0, 0.0]])   # 終了点など

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
        
        # Tkinterウィンドウとボタンの設定
        self.root = tk.Tk()
        self.root.title("GPS Data Yaml")
        self.button = tk.Button(self.root, text="Start Yaml", command=self.start_collection)
        self.button.pack()

    # ボタンが押されたときにデータ収集を開始
    def start_collection(self):
        if not self.is_collecting:  # すでに開始していない場合のみ実行
            self.get_logger().info("ボタンが押されました。")
            self.receive_avg_gps_callback()
            self.is_collecting = True
    
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

    def receive_avg_gps_callback(self):
        avg_lat, avg_lon, theta = self.init_lat, self.init_lon, self.init_theta
        
        GPSxy = self.conversion(avg_lat, avg_lon, theta)
        gps_np = np.array(GPSxy)

        self.waypoints_array = gps_np.T
        # YAML に保存
        self.save_to_yaml(self.waypoints_array)

    def save_to_yaml(self, waypoints):
        waypoint_list = []

        for i in range(waypoints.shape[1]):
            waypoint = [
                float(waypoints[0][i]),
                float(waypoints[1][i]),
                float(waypoints[2][i])
            ]
            waypoint_list.append(waypoint)

        yaml_data = {"waypoints": waypoint_list}

        with open("IGVC_waypoints.yaml", "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=None, allow_unicode=True)

        self.get_logger().info("IGVC_waypoints.yaml に保存しました。")
    
    def run(self):
        # Tkinterのメインループを開始
        self.root.mainloop()

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

