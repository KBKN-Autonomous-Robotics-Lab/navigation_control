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
            (35.42578984, 139.3138073), # nakaniwa hajime
            (35.42580947, 139.3138761),
            (35.42582577, 139.3139183),
            (35.42584276, 139.3139622),
            (35.42585746, 139.3139984),
            (35.42589533, 139.3139987),
            (35.42596721, 139.3139898),
            (35.42596884, 139.3139395) # nakaniwa owari
        ]

        self.first_point = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.last_point = np.array([[0.0, 0.0, 0.0]])

        self.root = tk.Tk()
        self.root.bind("<Key>", self.key_input_handler)
        self.reversed_flag = False

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )

        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, qos_profile)       
        self.odom_sub = self.create_subscription(nav_msgs.Odometry, '/fusion/odom', self.get_odom, qos_profile)
        self.waypoint_pub = self.create_publisher(geometry_msgs.PoseArray, 'current_waypoint', qos_profile)
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
        self.waypoints_array = np.array([[0.0],[0.0],[0.0]])
        self.waypoint_range_set = 3.5

    def key_input_handler(self, event):
        key = event.char.lower()
        if key == 'b':
            self.get_logger().info("キー入力: 'b' を受け取りました。ref_pointsを反転します。")
            self.ref_points.reverse()
            self.reversed_flag = True
        elif key == 'a':
            self.get_logger().info("キー入力: 'a' を受け取りました。通常順で実行します。")
            self.reversed_flag = False

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
        
        
        self.current_waypoint = 0;
        #self.waypoints_array = np.array([[xyz[0], xyz[1], xyz[2]], [next_xyz[0], next_xyz[1], next_xyz[2]]])
        self.waypoints_array = np.array([[xyz[0], next_xyz[0]],[xyz[1], next_xyz[1]], [xyz[2], next_xyz[2]]])
        #self.waypoints_array = np.stack((xyz,next_xyz))
        
        # クォータニオン → ヨー角（Z軸の回転）に変換
        #angle = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
        
        #pose_array = self.current_waypoint_msg(self.waypoints_array[:, self.current_waypoint], 'map')
        #self.waypoint_pub.publish(pose_array)
        
        self.get_logger().info(f"Received goal: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} deg")    
        self.get_logger().info(f"self.waypoints_array:{self.waypoints_array}")    
        self.get_logger().info(f"xyz_range:{xyz_range}")    
    
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
            #point = np.array([h_y, -h_x, 0.0])
            point = np.array([-h_y, h_x, 0.0])
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

        full_waypoints = np.concatenate([self.first_point, gps_np, self.last_point], axis=0)
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

        determine_dist = 4.5 if abs(waypoint_theta) > 90 else 4.5

        #check if the waypoint reached
        if waypoint_dist < determine_dist:
            #self.current_waypoint += 1
            if self.current_waypoint < len(self.waypoints_array[0,:])-1:
                self.current_waypoint += 1
            else:
                # goal:stopをtrueにしてアクションを再送信
                a=1;
                #self.stop_flag = True
                #self.get_logger().info("Stop flag reset to True")
                #self.send_action_request()

        pose_array = self.current_waypoint_msg(self.waypoints_array[:, self.current_waypoint], 'map')
        self.waypoint_pub.publish(pose_array)

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

