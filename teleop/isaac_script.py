import numpy as np

import omni.usd
import omni.kit.app
from pxr import UsdGeom, Gf

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped 
from pxr import UsdGeom, Gf, Gf 
import math

try:
    from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
except Exception:
    from omni.isaac.motion_generation.lula import LulaKinematicsSolver


# =========================
# paths
# =========================
URDF_PATH = "/home/dongseon/low_cost_humanoid_demo/rasberrypi/catkin_ws/src/humanoid_urdf_description/urdf/humanoid_urdf.urdf"
RIGHT_DESC_PATH = "/home/dongseon/right_arm_lula.yaml" 
LEFT_DESC_PATH = "/home/dongseon/left_arm_lula.yaml"

TARGET_PRIM_PATH = "/World/vr_target_right"
RIGHT_EE_FRAME = "rgripper_1"
LEFT_EE_FRAME = "lgripper_1"

LEFT_ARM_JOINTS = [ "lshd_pitch_joint", "lshd_yaw_joint", "lshd_roll_joint", "lelbow_joint", "lwrist_pitch_joint", "lwrist_yaw_joint", ]

RIGHT_ARM_JOINTS = [
    "rshd_pitch_joint",
    "rshd_yaw_joint",
    "rshd_roll_joint",
    "relbow_joint",
    "rwrist_pitch_joint",
    "rwrist_yaw_joint",
]

# =========================
# tuning
# =========================
ALPHA = 0.03          # smoothing
MAX_DELTA = 1.0     # rad per update
RATE_SKIP = 4

# VR pose -> Isaac Sim 좌표 변환
# 네가 앞에서 썼던 변환과 동일 계열
HEIGHT_OFFSET = 1.20

JOINT_LIMIT_LOW = np.array([ -0.6, -0.6, -0.6, -0.8, -0.6, -0.6 ], dtype=np.float64) 
JOINT_LIMIT_HIGH = np.array([ 0.6, 0.6, 0.6, 0.2, 0.6, 0.6 ], dtype=np.float64)

# =========================
# stage / sphere
# =========================
stage = omni.usd.get_context().get_stage()

from pxr import UsdGeom



def get_world_pos(prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print("Invalid prim:", prim_path)
        return None

    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(0)
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)

def ensure_target_sphere(path, pos):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(0.04)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.3, 1.0)])
        sphere.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


def set_translation(prim_path, xyz):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()

    translate_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break

    if translate_op is None:
        translate_op = xform.AddTranslateOp()

    translate_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def get_world_position(prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def teleop_to_sim_position(p):
    """
    IsaacTeleop:
      x: 좌우
      y: 높이 축처럼 들어옴
      z: 앞뒤 축처럼 들어옴

    Isaac Sim:
      x: 좌우
      y: 앞뒤
      z: 높이
    """
    sim_x = p.x
    sim_y = -p.z
    sim_z = p.y - HEIGHT_OFFSET

    return np.array([sim_x, sim_y, sim_z], dtype=np.float64)

def quat_to_rot_matrix_xyzw(x, y, z, w):
    # normalize
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-8:
        return np.eye(3)

    x /= n
    y /= n
    z /= n
    w /= n

    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def rot_matrix_to_euler_xyz_deg(R):
    # USD RotateXYZ용 degree
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    return np.degrees(np.array([x, y, z], dtype=np.float64))

def set_camera_pose(camera_path, pos, euler_deg):
    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        print("Invalid camera:", camera_path)
        return

    xform = UsdGeom.Xformable(prim)

    translate_op = None
    rotate_op = None

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op

    if translate_op is None:
        translate_op = xform.AddTranslateOp()

    if rotate_op is None:
        rotate_op = xform.AddRotateXYZOp()

    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    rotate_op.Set(Gf.Vec3f(float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])))

# =========================
# Lula solver
# =========================
right_solver = LulaKinematicsSolver( robot_description_path=RIGHT_DESC_PATH, urdf_path=URDF_PATH, ) 
left_solver = LulaKinematicsSolver( robot_description_path=LEFT_DESC_PATH, urdf_path=URDF_PATH, )

print("R Lula joints:", right_solver.get_joint_names())
print("L Lula joints:", left_solver.get_joint_names())


# =========================
# ROS2 node
# =========================
if not rclpy.ok():
    rclpy.init(args=None)


def get_fk_position(q):
    try:
        fk_result = solver.compute_forward_kinematics(
            frame_name=EE_FRAME_NAME,
            joint_positions=q,
        )
    except TypeError:
        fk_result = solver.compute_forward_kinematics(
            EE_FRAME_NAME,
            q,
        )

    if isinstance(fk_result, tuple):
        return np.asarray(fk_result[0], dtype=np.float64)

    return np.asarray(fk_result, dtype=np.float64)

class ArmLulaTeleop(Node):
    def __init__(self):
        super().__init__("right_arm_lula_teleop")

        self.sub = self.create_subscription(
            PoseArray,
            "/xr_teleop/ee_poses",
            self.pose_callback,
            10,
        )

        self.pub = self.create_publisher(
            JointState,
            "/joint_command",
            10,
        )

        self.head_sub = self.create_subscription(
	        PoseStamped,
	        "/xr_teleop/head_pose",
	        self.head_callback,
	        10,
        )
		
        self.head_origin_pos = None
        self.head_origin_quat = None
		
        self.camera_origin_pos = np.array([0.0, 0.0, 1.3], dtype=np.float64)
        self.head_scale = 1.0
		
        self.left_ee_origin = np.array([-0.000076, 0.277106, 0.379216], dtype=np.float64)
        self.right_ee_origin = np.array([0.036924, -0.280904, 0.379216], dtype=np.float64)
		
        self.left_target_pos = None
        self.right_target_pos = None
		
        self.left_vr_origin = None
        self.right_vr_origin = None
		
        self.left_q = np.zeros(len(LEFT_ARM_JOINTS), dtype=np.float64)
        self.right_q = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float64)

        self.scale = 0.25
        self.frame_count = 0
        self.left_last_vr_pos = None 
        self.right_last_vr_pos = None
        self.left_last_valid_target = None
        self.right_last_valid_target = None
		
        self.get_logger().info("Right arm Lula teleop node started")
	
        self.last_valid_cmd = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float64)
        self.has_valid_cmd = False
        ensure_target_sphere("/World/vr_target_left", self.left_ee_origin) 
        ensure_target_sphere("/World/vr_target_right", self.right_ee_origin)

			
    def pose_callback(self, msg):
	    if len(msg.poses) < 2:
	        return
	
	    lp = msg.poses[0].position
	    rp = msg.poses[1].position
	
	    left_vr_pos = np.array([lp.x, lp.y, lp.z], dtype=np.float64)
	    right_vr_pos = np.array([rp.x, rp.y, rp.z], dtype=np.float64)
	
	    self.update_arm_target(
	        side="left",
	        vr_pos=left_vr_pos,
	        ee_origin=self.left_ee_origin,
	    )
	
	    self.update_arm_target(
	        side="right",
	        vr_pos=right_vr_pos,
	        ee_origin=self.right_ee_origin,
	    )
	    
    def solve_ik(self, solver, frame_name, target_pos, warm_start):
	    try:
	        q_sol, success = solver.compute_inverse_kinematics(
	            frame_name=frame_name,
	            target_position=target_pos,
	            warm_start=warm_start,
	        )
	    except TypeError:
	        q_sol, success = solver.compute_inverse_kinematics(
	            frame_name,
	            target_pos,
	            None,
	            warm_start,
	        )
	
	    return np.asarray(q_sol, dtype=np.float64), bool(success)
	

    def publish_command(self):
	    msg = JointState()
	    msg.header.stamp = self.get_clock().now().to_msg()
	
	    msg.name = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
	    msg.position = (
	        [float(v) for v in self.left_q]
	        + [float(v) for v in self.right_q]
	    )
	
	    self.pub.publish(msg)

    def update(self):
	    self.frame_count += 1
	    if self.frame_count % RATE_SKIP != 0:
	        return
	
        # left IK
	    if self.left_target_pos is not None:
		    q_sol, success = self.solve_ik(
		        left_solver,
		        LEFT_EE_FRAME,
		        self.left_target_pos,
		        self.left_q,
		    )
		
		    if success:
		        self.left_q = (1.0 - ALPHA) * self.left_q + ALPHA * q_sol
		        self.left_last_valid_target = self.left_target_pos.copy()
		    else:
		        print("[LEFT IK] failed, hold last valid target")
		        if self.left_last_valid_target is not None:
		            self.left_target_pos = self.left_last_valid_target.copy()
	
	    # right IK
	    if self.right_target_pos is not None:
	        q_sol, success = self.solve_ik(
	            right_solver,
	            RIGHT_EE_FRAME,
	            self.right_target_pos,
	            self.right_q,
	        )
	
	        if success:
	            self.right_q = (1.0 - ALPHA) * self.right_q + ALPHA * q_sol
	        else:
		        print("[RIGHT IK] failed, hold last valid target")
		        if self.right_last_valid_target is not None:
		            self.right_target_pos = self.right_last_valid_target.copy()	
	
	
	    self.publish_command()
	    
    def update_arm_target(self, side, vr_pos, ee_origin):
	    # 0 pose는 무조건 무시
	    if np.linalg.norm(vr_pos) < 1e-6:
	        print(f"[{side}] zero frame skipped")
	        return
	
	    # side별 상태 선택
	    if side == "left":
	        vr_origin = self.left_vr_origin
	        last_vr_pos = self.left_last_vr_pos
	        current_target = self.left_target_pos
	    else:
	        vr_origin = self.right_vr_origin
	        last_vr_pos = self.right_last_vr_pos
	        current_target = self.right_target_pos
	
	    # 최초 1회만 origin 설정
	    if vr_origin is None:
	        vr_origin = vr_pos.copy()
	        last_vr_pos = vr_pos.copy()
	        current_target = ee_origin.copy()
	        print(f"[{side}] origin set:", vr_origin)
	
	    # 너무 큰 순간 점프는 tracking glitch로 보고 무시
	    vr_jump = np.linalg.norm(vr_pos - last_vr_pos)
	    if vr_jump > 0.25:
	        print(f"[{side}] VR jump skipped: {vr_jump:.3f}")
	        last_vr_pos = vr_pos.copy()
	
	        if side == "left":
	            self.left_last_vr_pos = last_vr_pos
	        else:
	            self.right_last_vr_pos = last_vr_pos
	        return
	
	    dvr = vr_pos - vr_origin
	
	    # controller 움직임 대비 target 움직임 줄이기
	    robot_delta = np.array([
	        dvr[0],
	        -dvr[2],
	        dvr[1],
	    ], dtype=np.float64) * self.scale
	
	    # delta 자체를 제한
	    max_delta = 0.08
	    robot_delta = np.clip(robot_delta, -max_delta, max_delta)
	
	    raw_target = ee_origin + robot_delta
	
	    # workspace 제한
	    limit = 0.08
	    raw_target[0] = np.clip(raw_target[0], ee_origin[0] - limit, ee_origin[0] + limit)
	    raw_target[1] = np.clip(raw_target[1], ee_origin[1] - limit, ee_origin[1] + limit)
	    raw_target[2] = np.clip(raw_target[2], ee_origin[2] - limit, ee_origin[2] + limit)
	
	    # target sphere 자체도 smoothing
	    if current_target is None:
	        target_pos = raw_target
	    else:
	        target_alpha = 0.15
	        target_pos = (1.0 - target_alpha) * current_target + target_alpha * raw_target
	
	    # 상태 저장
	    if side == "left":
	        self.left_vr_origin = vr_origin
	        self.left_last_vr_pos = vr_pos.copy()
	        self.left_target_pos = target_pos
	        set_translation("/World/vr_target_left", target_pos)
	    else:
	        self.right_vr_origin = vr_origin
	        self.right_last_vr_pos = vr_pos.copy()
	        self.right_target_pos = target_pos
	        set_translation("/World/vr_target_right", target_pos)
	
    def head_callback(self, msg):
	    p = msg.pose.position
	    q = msg.pose.orientation
	
	    head_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
	    head_quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
	
	    if np.linalg.norm(head_pos) < 1e-6:
	        return
	
	    if self.head_origin_pos is None:
	        self.head_origin_pos = head_pos.copy()
	        self.head_origin_quat = head_quat.copy()
	        print("[HEAD] origin set:", self.head_origin_pos)
	        return
	
	    dpos = head_pos - self.head_origin_pos
	
	    # VR 좌표 -> Isaac 카메라 좌표 매핑
	    # 필요하면 부호는 나중에 조정
	    camera_delta = np.array([
	        dpos[0],
	        -dpos[2],
	        dpos[1],
	    ], dtype=np.float64) * self.head_scale
	
	    camera_pos = self.camera_origin_pos + camera_delta
	
	    # 너무 멀리 튀지 않게 제한
	    camera_pos[0] = np.clip(camera_pos[0], -0.5, 0.5)
	    camera_pos[1] = np.clip(camera_pos[1], -0.5, 0.5)
	    camera_pos[2] = np.clip(camera_pos[2], 0.8, 1.8)
	
	    R = quat_to_rot_matrix_xyzw(
	        head_quat[0],
	        head_quat[1],
	        head_quat[2],
	        head_quat[3],
	    )
	
	    euler_deg = rot_matrix_to_euler_xyz_deg(R)
	
	    # 카메라 방향이 이상하면 여기 부호/축 조정
	    set_camera_pose(
	        "/World/VRHeadCamera",
	        camera_pos,
	        euler_deg,
	    )
	
	    if self.frame_count % 30 == 0:
	        print("[HEAD DEBUG]")
	        print("  dpos:", dpos)
	        print("  camera_pos:", camera_pos)
	        print("  euler_deg:", euler_deg)


node = ArmLulaTeleop()


def on_update(event):
    rclpy.spin_once(node, timeout_sec=0.0)
    node.update()


sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    on_update,
    name="right_arm_lula_teleop_update",
)

print("Started:")
print("  subscribe: /xr_teleop/ee_poses")
print("  publish:   /joint_command")
print("  target:    /World/vr_target_right")
```





```python
from pxr import UsdGeom, Gf
import omni.usd
import omni.kit.viewport.utility as vp_utils

stage = omni.usd.get_context().get_stage()

CAMERA_PATH = "/World/VRHeadCamera"

prim = stage.GetPrimAtPath(CAMERA_PATH)

if not prim.IsValid():
    cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.3))
    cam.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    print("Created camera:", CAMERA_PATH)
else:
    print("Camera already exists:", CAMERA_PATH)

viewport = vp_utils.get_active_viewport()
viewport.camera_path = CAMERA_PATH

print("Viewport camera set to:", CAMERA_PATH)