import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Odometry
import subprocess
import yaml
from my_msgs.action import StopFlag  # Actionメッセージのインポート
import tkinter as tk
import threading
from std_msgs.msg import Bool

class WaypointMonitor(Node):
    def __init__(self):
        super().__init__('waypoint_monitor')
        self.action_client = ActionClient(self, StopFlag, 'stop_flag')  # ActionClientの設定
        self.action_sent = False  # アクションが送信されたかを追跡
        self.stop = False # stopするかの変数(True=stop, False=go)
        self.nav_state = False
        
        # publish /nav_state
        self.nav_state_publisher = self.create_publisher(Bool, '/nav_state', 10)
        self.create_timer(1, self.publish_nav_state)
        
        # Tkinterウィンドウとボタンの設定
        self.root = tk.Tk()
        self.root.title("Waypoint Monitor Control")
        self.button = tk.Button(self.root, text="Resume", command=self.resume_action)
        self.button.pack()

    # サーバーにアクションを送信する関数
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

    def publish_nav_state(self):
        # /nav_state publish
        msg = Bool()
        msg.data = self.nav_state
        self.nav_state_publisher.publish(msg)
        
    def resume_action(self):
        # ボタンを押したときにstopをFalseにしてアクションを再送信
        self.stop = False
        self.get_logger().info("Stop flag reset to False")
        self.nav_state = True
        self.send_action_request()

    def run(self):
        # Tkinterのメインループを開始
        self.root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    waypoint_monitor = WaypointMonitor()
    # rclpyスピンを別スレッドで実行
    rclpy_thread = threading.Thread(target=rclpy.spin, args=(waypoint_monitor,))
    rclpy_thread.start()
    # Tkinter GUIのループを実行
    waypoint_monitor.run()
    # スレッド終了後にノードを破棄
    waypoint_monitor.destroy_node()
    rclpy.shutdown()
    rclpy_thread.join()

if __name__ == '__main__':
    main()

