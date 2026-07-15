#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState


RIGHT_ARM_JOINTS = [
    "rshd_pitch_joint",
    "rshd_yaw_joint",
    "rshd_roll_joint",
    "relbow_joint",
    "rwrist_pitch_joint",
    "rwrist_yaw_joint",
]


class VrRightArmTest(Node):
    def __init__(self):
        super().__init__("vr_right_arm_test")

        self.pub = self.create_publisher(JointState, "/joint_command", 10)

        self.sub = self.create_subscription(
            PoseArray,
            "/xr_teleop/ee_poses",
            self.ee_pose_callback,
            10,
        )

        self.last_cmd = [0.0] * len(RIGHT_ARM_JOINTS)

        self.get_logger().info("VR right arm test node started")

    def ee_pose_callback(self, msg: PoseArray):
        if len(msg.poses) < 2:
            return

        right = msg.poses[1].position

        # IsaacTeleop -> 간단 command 매핑 테스트
        # 아직 IK 아님. 손 움직임이 오른팔 관절 command로 들어가는지 확인하는 단계.
        #
        # y는 높이처럼 보였으므로 shoulder pitch에 작게 매핑
        # x는 좌우 움직임이므로 shoulder yaw에 작게 매핑
        # z는 앞뒤 움직임이므로 elbow에 작게 매핑
        cmd = [0.0] * len(RIGHT_ARM_JOINTS)

        cmd[0] = self.clamp((right.y - 1.45) * 0.8, -0.5, 0.5)   # rshd_pitch_joint
        cmd[1] = self.clamp(right.x * 1.0, -0.5, 0.5)            # rshd_yaw_joint
        cmd[2] = 0.0                                             # rshd_roll_joint
        cmd[3] = self.clamp((-right.z - 0.50) * 0.8, -0.5, 0.5)  # relbow_joint
        cmd[4] = 0.0                                             # rwrist_pitch_joint
        cmd[5] = 0.0                                             # rwrist_yaw_joint

        # 저역통과 필터: 갑자기 튀는 것 방지
        alpha = 0.1
        self.last_cmd = [
            (1.0 - alpha) * old + alpha * new
            for old, new in zip(self.last_cmd, cmd)
        ]

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = RIGHT_ARM_JOINTS
        out.position = self.last_cmd

        self.pub.publish(out)

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))


def main():
    rclpy.init()
    node = VrRightArmTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
