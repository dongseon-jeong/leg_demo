# leg_env_cfg.py

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .legs_cfg import LEGS_CFG


@configclass
class LegEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 10.0

    action_space = 12
    observation_space = 48
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    # ⬇ 여기서도 env 복수 개를 대상으로 /world/legs 를 패턴으로 지정
    robot_cfg: ArticulationCfg = LEGS_CFG.replace(
        prim_path="/World/envs/env_.*/legs"
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
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

    # action_scale = 1.0
    # torque_limit = 3.0

    # rew_scale_alive = 1.0
    # rew_scale_terminated = -10.0
    # rew_scale_forward_vel = 2.0
    # rew_scale_upright = 1.0
    # rew_scale_joint_vel = -0.001
    # rew_scale_action_rate = -0.001
    # rew_scale_energy = -0.0005

    base_height_target = 0.25
    min_base_height = 0.10
    max_base_pitch = 0.7
    max_base_roll = 0.7

    # reward scales
    rew_scale_alive = 0.1           # 너무 크지 않게, 그래도 살아있으면 + 보상
    rew_scale_terminated = -3.0     # 넘어지면 꽤 큰 음수
    rew_scale_forward_vel = 2.0     # 앞으로 가면 많이 보상
    rew_scale_upright = 3.0         # 자세 잘 유지하면 꽤 보상

    # penalties는 일단 아주 약하게 시작
    rew_scale_joint_vel = -1e-4
    rew_scale_action_rate = -1e-4
    rew_scale_energy = 0.0          # 처음엔 꺼버려도 됨

    action_scale = 5.0
    torque_limit = 5.0