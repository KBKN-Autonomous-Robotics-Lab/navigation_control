import rclpy
import time
from rclpy.action import ActionServer
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String  # 追加: トピックの型
from my_msgs.action import StopFlag

# サーバーノード
class HumanControl(Node):
    # 初期化
    def __init__(self):
        super().__init__('human_control')

        # トピックの購読設定
        self.subscription = self.create_subscription(
            String,
            '/human_status',
            self.human_callback,
            10
        )

        # ActionClientの設定
        self.action_client = ActionClient(self, StopFlag, 'stop_flag')
        
        # 初期値の設定
        #self.traffic_action = False # traffic_actionするかの変数
        self.stop = True
        self.previous_status = None

    # human_status トピックのコールバック
    def human_callback(self, msg):
        # 現在の状態が "Stop" になったら stop_flag を True にする
        if msg.data == "Stop" and self.previous_status != "Stop":
            self.get_logger().info("human detected")
            self.stop = True
            #self.traffic_action = True
            self.send_action_request()
            # 1秒後にアクションを送信
            #self.create_timer(1.0, self.delayed_action_send, callback_group=None)
        # 状態の更新
        self.previous_status = msg.data
        
    def delayed_action_send(self):
        self.send_action_request()
    
    # サーバーにアクションを送信する関数
    def send_action_request(self):
        goal_msg = StopFlag.Goal()
        
        # stop変数の状態でaの値を決定
        if self.stop: # True
            goal_msg.a = 1  # stop
        else: # False
            goal_msg.a = 0  # go
            
        # traffic_action変数の状態でbの値を決定
        #if self.traffic_action:
        #    goal_msg.b = 0  # start judge
        #else:
        #    goal_msg.b = 1  # 

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
    
# メイン関数
def main(args=None):
    # ROS通信の初期化
    rclpy.init(args=args)

    # サーバーノードの生成
    human_control = HumanControl()

    # ノード終了の待機
    rclpy.spin(human_control)

if __name__ == '__main__':
    main()

