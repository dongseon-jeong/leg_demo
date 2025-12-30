# legs_cfg.py

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import os

BASE_PATH = os.environ.get("BASE_PATH")

LEGS_CFG = ArticulationCfg(
    # ⬇ 최종 스테이지에서 로봇의 prim 경로 패턴
    #    /World/envs/env_0/world/legs, /World/envs/env_1/world/legs, ...
    prim_path="/World/envs/env_.*/legs",   
    spawn=sim_utils.UsdFileCfg(
        # 여긴 그대로 네 USD 파일 경로
        usd_path=BASE_PATH+"/leg_w.usd",
        activate_contact_sensors=True,   # 추가
        copy_from_source=True,   # 추가/수정
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=False,   # 반드시 False로
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        # copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={".*": 0.0},

    ),
    actuators={
        "legs_dxl": ImplicitActuatorCfg(
            joint_names_expr=[
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
            ],
            stiffness={".*": 800.0},
            damping={".*": 40.0},
            velocity_limit_sim={".*": 10.0},
        ),
    },
)
