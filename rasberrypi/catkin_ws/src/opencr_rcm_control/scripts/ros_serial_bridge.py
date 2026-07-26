#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
import serial
import math

RAD_TO_DXL_STEP = 2048.0 / math.pi
INVERT_IDS = {100}  # {2, 3,4,5,6}


class ROSSerialBridge:
    def __init__(self):
        rospy.init_node('ros_serial_bridge', anonymous=False)
        
        self.port = "/dev/ttyACM0"
        self.baud = 115200
        
        try:
            self.py_serial = serial.Serial(port=self.port, baudrate=self.baud, timeout=0.01)
            rospy.loginfo("➔ Connected to OpenCR successfully!")
        except Exception as e:
            rospy.logerr(f"➔ Failed to connect OpenCR: {e}")
            return

        self.last_sent_steps = [0] * 28

        self.joint_id_map = {
            "lbase_joint": 1,
            "ll1_joint":   2,
            "ll2_joint":   3,
            "ll3_joint":   4,
            "ll4_joint":   5,
            "ll5_joint":   6,
            
            "rbase_joint": 7,
            "rl1_joint":   8,
            "rl2_joint":   9,
            "rl3_joint":   10,
            "rl4_joint":   11,
            "rl5_joint":   12,
        

            # Neck
            "neck_roll_joint": 13,
            "neck_pitch_joint": 14,

            # Left arm
            "lshd_pitch_joint": 15,
            "lshd_yaw_joint": 16,
            "lshd_roll_joint": 17,
            "lelbow_joint": 18,
            "lwrist_pitch_joint": 19,
            "lwrist_yaw_joint": 20,
            "lgripper_joint": 21,

            # Right arm
            "rshd_pitch_joint": 22,
            "rshd_yaw_joint": 23,
            "rshd_roll_joint": 24,
            "relbow_joint": 25,
            "rwrist_pitch_joint": 26,
            "rwrist_yaw_joint": 27,
            "rgripper_joint": 28,


        }

        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback, queue_size=1)
        rospy.loginfo("➔ ROS-Isaac Serial Bridge Node Running...")

    def joint_state_callback(self, msg):
        cmd_tokens = []
        any_changed = False

        for i, name in enumerate(msg.name):
            if name in self.joint_id_map:
                joint_id = self.joint_id_map[name]
                radian = msg.position[i]
                
                if joint_id in INVERT_IDS:
                    radian = -radian
                
                step_offset = int(radian * RAD_TO_DXL_STEP)

                if abs(step_offset - self.last_sent_steps[joint_id - 1]) > 3:
                    cmd_tokens.append(f"{joint_id},{step_offset}")
                    self.last_sent_steps[joint_id - 1] = step_offset
                    any_changed = True

        if any_changed and cmd_tokens:
            serial_cmd = "/".join(cmd_tokens) + "\n"
            
            rospy.loginfo(f"➔ Sending to OpenCR: {serial_cmd.strip()}")
            
            self.py_serial.write(serial_cmd.encode())

    def shutdown_hook(self):
        rospy.loginfo("Shutting down Bridge. Stopping motors safely...")
        if hasattr(self, 'py_serial') and self.py_serial.is_open:
            reset_cmd = "/".join([f"{i},0" for i in range(1, 29)]) + "\n"
            self.py_serial.write(reset_cmd.encode())
            self.py_serial.close()
         


if __name__ == '__main__':
    bridge = ROSSerialBridge()
    rospy.on_shutdown(bridge.shutdown_hook)
    rospy.spin()
