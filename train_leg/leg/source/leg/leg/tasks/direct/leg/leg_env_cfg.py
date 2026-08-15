# leg_env_cfg.py

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .legs_cfg import LEGS_CFG

from isaaclab.sensors import ContactSensorCfg

@configclass
class LegEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 10.0

    action_space = 12
    observation_space = 48 #48
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    robot_cfg: ArticulationCfg = LEGS_CFG.replace(
        prim_path="/World/envs/env_.*/legs"   # / 로 시작
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=2.0,
        replicate_physics=True,
    )

    joint_names = [
        "lbase_joint",
        "rbase_joint",
        "ll1_joint",
        "rl1_joint",
        "ll2_joint",
        "rl2_joint",
        "ll3_joint",
        "rl3_joint",
        "ll4_joint",
        "rl4_joint",
        "ll5_joint",
        "rl5_joint",
    ]
    gait_period = 1.0
    action_scale = 0.3
    torque_limit = 8.0

    # 베이스 높이
    base_height_target = 0.16
    min_base_height = 0.10
    max_base_pitch = 1.0
    max_base_roll = 1.0

    # reward scales
    rew_scale_alive = 0.2      # 1.0 너무 크지 않게, 그래도 살아있으면 + 보상
    rew_scale_terminated = -5.0     # -10 넘어지면 꽤 큰 음수
    rew_scale_upright = 0.5         # 3 자세 잘 유지하면 꽤 보상
    rew_scale_forward_vel = 12.0     #25 앞으로 가면 많이 보상
    rew_vel_track_rate = 8.0 # 5 속도 증가
    rew_heading_rate = -8.0  #  15.0

    # penalties는 일단 아주 약하게 시작
    rew_scale_joint_vel = -0.005
    rew_scale_action_rate = -0.2
    rew_scale_energy = -0.0001          # 처음엔 꺼버려도 됨
    rew_leg_pose_rate = -0.1
    rew_lateral_vel_rate = -20.0 # 50
    rew_vertical_vel_rate = -5.0 # 0.2
    rew_body_ang_vel_rate = -0.5

    # 스텝용
    foot_target_width = 0.06
    ground_z = 0.035
    swing_z = 0.08 # 0.11
    rew_fheight_rate = 2.0 #0.25
    rew_foot_flat_rate = 0.5 #0.5

    # ---- 접촉 센서는 미사용
    GROUND_FILTER = ["/World/GroundPlane/GroundPlane/CollisionPlane"]

    scene.left_foot_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/legs/ll6_.*",
        filter_prim_paths_expr=GROUND_FILTER,   # 추가
        update_period=0.0,
        history_length=1,
    )
    scene.right_foot_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/legs/rl6_.*",
        filter_prim_paths_expr=GROUND_FILTER,   # 추가
        update_period=0.0,
        history_length=1,
    )

    # ---- 발 간섭 ----
    scene.left_leg_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/legs/ll(4|5|6)_.*",
        filter_prim_paths_expr=["/World/envs/env_.*/legs/rl(4|5|6)_.*"],
        update_period=0.0,
        history_length=1,
    )
    scene.right_leg_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/legs/rl(4|5|6)_.*",
        filter_prim_paths_expr=["/World/envs/env_.*/legs/ll(4|5|6)_.*"], #ll.*
        update_period=0.0,
        history_length=1,
    )
