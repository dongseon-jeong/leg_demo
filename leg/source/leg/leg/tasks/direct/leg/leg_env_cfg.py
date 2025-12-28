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
    observation_space = 48
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    robot_cfg: ArticulationCfg = LEGS_CFG.replace(
        prim_path="/World/envs/env_.*/legs"   # ✅ 반드시 / 로 시작
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

    base_height_target = 0.25
    min_base_height = 0.10
    max_base_pitch = 1.0
    max_base_roll = 1.0

    # reward scales
    rew_scale_alive = 2       # 너무 크지 않게, 그래도 살아있으면 + 보상
    rew_scale_terminated = -6.0     # 넘어지면 꽤 큰 음수
    rew_scale_forward_vel = 1.0     # 앞으로 가면 많이 보상
    rew_scale_upright = 3.0         # 자세 잘 유지하면 꽤 보상

    # penalties는 일단 아주 약하게 시작
    rew_scale_joint_vel = -1e-4
    rew_scale_action_rate = -1e-4
    rew_scale_energy = -0.0005          # 처음엔 꺼버려도 됨

    action_scale = 0.5
    torque_limit = 3.0

    rew_scale_leg_interference = -0.005   # ✅ 마이너스 보상(패널티)
    leg_interference_force_threshold = 100.0  # ✅ N 단위(대충 시작값)

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
