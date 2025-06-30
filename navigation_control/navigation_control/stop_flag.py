import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from my_msgs.action import StopFlag  # アクションのメッセージ型
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.task import Future

class TrafficLightMonitor(Node):
    def __init__(self):
        super().__init__('traffic_light_monitor')

        # QoS設定
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

        # /traffic_light_statusのサブスクリプション
        self.subscription = self.create_subscription(
            String,
            '/traffic_light_status',
            self.traffic_light_callback,
            qos_profile
        )
        self.subscription  # 無警告のため

        # StopFlagのアクションクライアント
        self._action_client = ActionClient(self, StopFlag, 'set_stop_flag')

    def traffic_light_callback(self, msg):
        self.get_logger().info(f'Received traffic light status: "{msg.data}"')

        if msg.data == 'red':
            self.send_stop_flag(True)  # 赤信号でstop_flag=1を送る
        else:
            self.send_stop_flag(False)  # 赤以外ならstop_flag=0を送る

    def send_stop_flag(self, flag_value):
        goal_msg = StopFlag.Goal()
        goal_msg.data = bool(flag_value)

        self.get_logger().info('Waiting for action server...')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Action server not available!')
            return

        self.get_logger().info(f'Sending stop flag: {1 if flag_value else 0}')
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future):
        # ゴールが受け入れられたかどうか確認
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server')
            return
        self.get_logger().info('Goal accepted by action server')

        # 結果が来るのを待つ
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future: Future):
        result = future.result()
        if result is None:
            self.get_logger().error('Result is None')
            return
        self.get_logger().info(f'Action result received: stop_flag = {result.result.success}')


def main(args=None):
    rclpy.init(args=args)
    traffic_light_monitor = TrafficLightMonitor()
    try:
        rclpy.spin(traffic_light_monitor)
    except KeyboardInterrupt:
        traffic_light_monitor.get_logger().info('Shutting down via keyboard interrupt.')
    finally:
        traffic_light_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

