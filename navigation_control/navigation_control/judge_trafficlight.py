import rclpy
import time
from rclpy.action import ActionServer
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String  # 追加: トピックの型
from my_msgs.action import StopFlag

# サーバーノード
class JudgeTrafficLight(Node):
    # 初期化
    def __init__(self):
        super().__init__('judge_trafficlight')

        # トピックの購読設定
        self.subscription = self.create_subscription(
            String,
            '/traffic_light_status',
            self.traffic_light_callback,
            10
        )

        # アクションサーバーの生成
        self.server = ActionServer(
            self,
            StopFlag,
            "traffic_flag",
            self.listener_callback
        )
        
        # ActionClientの設定
        self.action_client = ActionClient(self, StopFlag, 'stop_flag')
        self.action_client2 = ActionClient(self, StopFlag, 'traffic_flag')
        
        # 初期値の設定
        self.traffic_action = False # traffic_actionするかの変数
        self.traffic_flag = 0
        self.stop = True
        self.previous_status = None

    # traffic_light_status トピックのコールバック
    def traffic_light_callback(self, msg):
        if self.traffic_flag == 1:
            # 現在の状態が "red" で、トピックの内容が "blue" になったら stop_flag を False にする
            if self.previous_status == "red" and msg.data == "blue":
                self.get_logger().info("Traffic light changed from red to blue.")
                self.stop = False
                self.traffic_flag = 0  # フラグをリセット
                self.traffic_action = True
                self.send_action_request()
            # 状態の更新
            self.previous_status = msg.data

    # アクションリクエストのコールバック
    def listener_callback(self, goal_handle):
        self.get_logger().info(f"Received goal with a: {goal_handle.request.a}, b: {goal_handle.request.b}")
        
        # クライアントから送られた a を traffic_flag に代入
        self.traffic_flag = goal_handle.request.a
        print(f"traffic_flag set to: {self.traffic_flag}")
        
        # フィードバックの返信
        for i in range(1):
            feedback = StopFlag.Feedback()
            feedback.rate = i * 0.1
            goal_handle.publish_feedback(feedback)
            time.sleep(0.5)

        # レスポンスの返信
        goal_handle.succeed()
        result = StopFlag.Result()
        result.sum = goal_handle.request.a + goal_handle.request.b  # 結果の計算
        return result
        
    # サーバーにアクションを送信する関数
    def send_action_request(self):
        goal_msg = StopFlag.Goal()
        
        # stop変数の状態でaの値を決定
        if self.stop: # True
            goal_msg.a = 1  # stop
        else: # False
            goal_msg.a = 0  # go
            
        # traffic_action変数の状態でbの値を決定
        if self.traffic_action:
            goal_msg.b = 0  # start judge
        else:
            goal_msg.b = 1  # 

        # アクションサーバーが利用可能になるまで待機
        self.action_client.wait_for_server()
        self.action_client2.wait_for_server()

        # アクションを非同期で送信
        self.future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.future2 = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.future.add_done_callback(self.response_callback)
        self.future2.add_done_callback(self.response_callback2)

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
        
    def response_callback2(self, future2):
        goal_handle = future2.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")

        self.result_future2 = goal_handle.get_result_async()
        self.result_future2.add_done_callback(self.result_callback2)

    # 結果のコールバック
    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.sum}")
        
    def result_callback2(self, future2):
        result = future2.result().result
        self.get_logger().info(f"Result: {result.sum}")
    
# メイン関数
def main(args=None):
    # ROS通信の初期化
    rclpy.init(args=args)

    # サーバーノードの生成
    judge_trafficlight = JudgeTrafficLight()

    # ノード終了の待機
    rclpy.spin(judge_trafficlight)

if __name__ == '__main__':
    main()

