import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
from rclpy.action import ActionServer
from rclpy.action import ActionClient
from std_msgs.msg import String  # 追加: トピックの型
from my_msgs.action import StopFlag

class MJPGCameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.cap = cv2.VideoCapture('/dev/sensors/webcam')
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # アクションサーバーの生成
        self.server = ActionServer(
            self,
            StopFlag,
            "traffic_flag",
            self.listener_callback
        )
        self.action_client = ActionClient(self, StopFlag, 'traffic_flag')
        
        # 初期値の設定
        self.traffic_action = False # traffic_actionするかの変数
        self.traffic_flag = 1
        self.stop = True

        # MJPG フォーマットの設定
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not self.cap.isOpened():
            self.get_logger().error('カメラデバイスを開けません')
            rclpy.shutdown()

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

    def timer_callback(self):
        ret, frame = self.cap.read()
        if self.traffic_flag == 1:
            if ret:
                # フレームを RGB に変換してパブリッシュ
                image_message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.publisher_.publish(image_message)
            else:
                self.get_logger().warning('フレームを取得できません')

def main(args=None):
    rclpy.init(args=args)
    node = MJPGCameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

