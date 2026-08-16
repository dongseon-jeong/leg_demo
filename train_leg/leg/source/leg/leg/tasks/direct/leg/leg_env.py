# leg_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor  # <-- make sure your IsaacLab version provides this

from .leg_env_cfg import LegEnvCfg

from typing import Tuple

class LegEnv(DirectRLEnv):
    cfg: LegEnvCfg

    def __init__(self, cfg: LegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # # --- Contact sensors (must exist in cfg/scene) ---
        # # Foot contact (ground reaction)
        # self._left_foot_contact: ContactSensor = self.scene.sensors["left_foot_contact"]
        # self._right_foot_contact: ContactSensor = self.scene.sensors["right_foot_contact"]

        # # Leg-leg interference (optional but used in reward)
        # self._left_leg_interf: ContactSensor = self.scene.sensors["left_leg_contact"]
        # self._right_leg_interf: ContactSensor = self.scene.sensors["right_leg_contact"]

        # DOF indices for controlled joints
        self._dof_indices, _ = self.robot.find_joints(self.cfg.joint_names)
        self._num_dofs = len(self.cfg.joint_names)

        # State views (updated each step)
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.joint_nominal = (
            self.robot.data.default_joint_pos[:, self._dof_indices]
            .clone()
        )

        # knee 기본 굽힘
        self.joint_nominal[:, 6] = -0.3
        self.joint_nominal[:, 7] = 0.3

        # Action buffers
        self.actions = torch.zeros(self.num_envs, self._num_dofs, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        # Torque proxy (we use position targets; keep zeros unless you add explicit torque control)
        self.joint_torques = torch.zeros_like(self.actions)

        # Termination debounce counters
        self._low_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._tilt_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # foot body ids (exact names)
        lf_ids, _ = self.robot.find_bodies("ll6_1")
        rf_ids, _ = self.robot.find_bodies("rl6_1")

        self._lf_body_id = int(lf_ids[0])
        self._rf_body_id = int(rf_ids[0])

        # 몸통 높이 대체
        ll1_ids, _ = self.robot.find_bodies("ll1_1")
        rl1_ids, _ = self.robot.find_bodies("rl1_1")
        self._ll1_body_id = int(ll1_ids[0])
        self._rl1_body_id = int(rl1_ids[0])

        self.prev_base_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.imu_bias_lin = torch.zeros(
            (self.num_envs,3),
            device=self.device
        )

        self.imu_bias_ang = torch.zeros(
            (self.num_envs,3),
            device=self.device
        )

        self._q_log_counter = 0

        self.lf_z_sum = torch.zeros(self.num_envs, device=self.device)
        self.rf_z_sum = torch.zeros(self.num_envs, device=self.device)
        self.z_count = torch.zeros(self.num_envs, device=self.device)
        self.phase_offset = torch.zeros(self.num_envs, device=self.device)

    # ---------------------------------------------------------------------
    # Scene
    # ---------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/GroundPlane", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------------
    # RL hooks
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Keep last actions for action-rate penalty
        self.last_actions[:] = self.actions

        # Smooth + clamp incoming actions
        alpha = float(self.cfg.action_smoothing_alpha) if hasattr(self.cfg, "action_smoothing_alpha") else 0.2
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions[:] = (1.0 - alpha) * self.actions + alpha * actions

    def _apply_action(self) -> None:
        # Clamp final action
        act = torch.clamp(self.actions, -1.0, 1.0)

        # Ramp-in at episode start (avoid immediate explosions)
        ramp_seconds = float(self.cfg.action_ramp_seconds) if hasattr(self.cfg, "action_ramp_seconds") else 0.5
        ramp_steps = int(ramp_seconds / (self.cfg.sim.dt * self.cfg.decimation))
        ramp = torch.clamp(
            self.episode_length_buf.float() / float(max(ramp_steps, 1)),
            0.0,
            1.0,
        ).unsqueeze(-1)  # [N,1]

        # Default pose for controlled joints
        # q0 = self.robot.data.default_joint_pos[:, self._dof_indices]  # [N,D]
        q0 = self.joint_nominal

        # action_scale is "radian delta"
        q_target = q0 + (act * float(self.cfg.action_scale)) * ramp

        # Position targets (implicit PD from actuators)
        self.robot.set_joint_position_target(q_target, joint_ids=self._dof_indices)

        # # Torque proxy (kept at zero unless you compute/measure real torques)
        # self.joint_torques.zero_()

    def _get_observations(self) -> dict:
        # Refresh joint states
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        root_state = self.robot.data.root_state_w
        base_pos = root_state[:, 0:3]
        base_quat = root_state[:, 3:7]     # expected (qw,qx,qy,qz)
        base_lin_vel = root_state[:, 7:10]
        base_ang_vel = root_state[:, 10:13]
       # =========================
        # IMU noise simulation
        # =========================

        lin_vel_noise_std = 0.02
        ang_vel_noise_std = 0.01
        base_lin_vel = base_lin_vel + torch.randn_like(base_lin_vel) * lin_vel_noise_std + self.imu_bias_lin
        base_ang_vel = base_ang_vel + torch.randn_like(base_ang_vel) * ang_vel_noise_std + self.imu_bias_ang

        # Prefer COM height if available
        # if hasattr(self.robot.data, "root_com_pos_w"):
        #     base_height = self.robot.data.root_com_pos_w[:, 2]
        # elif hasattr(self.robot.data, "com_pos_w"):
        #     base_height = self.robot.data.com_pos_w[:, 2]
        # else:
        #     base_height = base_pos[:, 2]
        bs = self.robot.data.body_state_w
        base_height = 0.5 * (
            bs[:, self._ll1_body_id, 2] + bs[:, self._rl1_body_id, 2]
        )

        tilt_angle = compute_tilt_from_quat_wxyz(base_quat)

        # Controlled joints
        q = self.joint_pos[:, self._dof_indices]
        qd = self.joint_vel[:, self._dof_indices]

        # Action history
        last_act = self.last_actions

        # Extra scalars
        forward_vel = base_lin_vel[:, 0].unsqueeze(-1)
        height_err = (base_height - float(self.cfg.base_height_target)).unsqueeze(-1)

        gait_period = float(self.cfg.gait_period)
        phase = torch.fmod(
            self.episode_length_buf.float() * (self.cfg.sim.dt * self.cfg.decimation) / gait_period
            + self.phase_offset,
            1.0,
        )

        phase_sin = torch.sin(2.0 * torch.pi * phase).unsqueeze(-1)
        phase_cos = torch.cos(2.0 * torch.pi * phase).unsqueeze(-1)



        obs = torch.cat(
            (
                q,                       # D
                qd,                      # D
                base_lin_vel,            # 3
                base_ang_vel,            # 3
                base_height.unsqueeze(-1),  # 1
                tilt_angle.unsqueeze(-1),   # 1
                last_act,                # D
                forward_vel,             # 1
                # x_pos,                   # 1
                # y_pos,                   # 1
                height_err,              # 1
                phase_sin,
                phase_cos,
            ),
            dim=-1,
        )

        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, -100.0, 100.0)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Refresh states
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        root_state = self.robot.data.root_state_w
        base_pos = root_state[:, 0:3] # position world
        base_quat = root_state[:, 3:7]     # quaternion (qw,qx,qy,qz)
        base_lin_vel = root_state[:, 7:10] # linear velocity world
        base_ang_vel = root_state[:, 10:13] # angular velocity world
 

        # # Prefer COM height if available
        # if hasattr(self.robot.data, "root_com_pos_w"):
        #     base_height = self.robot.data.root_com_pos_w[:, 2]
        # elif hasattr(self.robot.data, "com_pos_w"):
        #     base_height = self.robot.data.com_pos_w[:, 2]
        # else:
        #     base_height = base_pos[:, 2]
        bs = self.robot.data.body_state_w
        base_height = 0.5 * (
            bs[:, self._ll1_body_id, 2] + bs[:, self._rl1_body_id, 2]
        )

        tilt_angle = compute_tilt_from_quat_wxyz(base_quat)

        q = self.joint_pos[:, self._dof_indices]
        qd = self.joint_vel[:, self._dof_indices]
        action_rate = self.actions - self.last_actions

        self.joint_torques.copy_(
            self.robot.data.applied_torque[:, self._dof_indices]
        )

        # --- Foot flatness shaping (only when in contact) ---
        bs = self.robot.data.body_state_w

        # 다리사이 간격
        # lf_pos = bs[:, 10, 0:3]
        # rf_pos = bs[:, 11, 0:3]
        lf_state = bs[:, self._lf_body_id]
        rf_state = bs[:, self._rf_body_id]

        lf_pos = lf_state[:, 0:3] # position world
        rf_pos = rf_state[:, 0:3]

        # 다리 높이
        lf_z = lf_pos[:,2]
        rf_z = rf_pos[:,2]

        self.lf_z_sum += lf_z.detach()
        self.rf_z_sum += rf_z.detach()
        self.z_count += 1.0

        lf_z_avg = self.lf_z_sum / torch.clamp(self.z_count, min=1.0)
        rf_z_avg = self.rf_z_sum / torch.clamp(self.z_count, min=1.0)

        gait_period = float(self.cfg.gait_period)
        phase = torch.fmod(
            self.episode_length_buf.float() * (self.cfg.sim.dt * self.cfg.decimation) / gait_period
            + self.phase_offset,
            1.0,
        )

        # -------------------------
        # Reward
        # -------------------------
        (
            total_reward,

            r_alive,
            r_term,
            r_upright,
            r_forward,
            r_vel_track,
            r_heading,
            r_foot_height,
            rew_yaw_rate,
            rew_lateral_world,
            rew_foot_height_usage_balance

        ) = compute_rewards(
            # scales
            float(self.cfg.rew_scale_alive),
            float(self.cfg.rew_scale_terminated),
            float(self.cfg.rew_scale_forward_vel),
            float(self.cfg.rew_scale_upright),
            float(self.cfg.rew_scale_joint_vel),
            float(self.cfg.rew_scale_action_rate),
            float(self.cfg.rew_scale_energy),
            float(self.cfg.rew_vel_track_rate),
            float(self.cfg.rew_fheight_rate),
            float(self.cfg.rew_leg_pose_rate),

            # robot state tensors
            base_quat,
            base_ang_vel,
            base_lin_vel,
            base_height,
            tilt_angle,
            q,
            qd,
            self.joint_torques,
            action_rate,
            self.joint_nominal,
            float(self.cfg.base_height_target),
            self.reset_terminated,
            float(self.cfg.rew_heading_rate),

            lf_z,
            rf_z,
            lf_z_avg,
            rf_z_avg,
            float(self.cfg.ground_z),
            float(self.cfg.swing_z),
            phase,

        )


        if not hasattr(self, "extras"):
            self.extras = {}

        self.extras["episode"] = {
            "rew_alive": r_alive.mean(),
            "rew_termination": r_term.mean(),

            "rew_upright": r_upright.mean(),
            "rew_forward": r_forward.mean(),
            "rew_vel_track": r_vel_track.mean(),
            "rew_heading": r_heading.mean(),
            "rew_yaw_rate":rew_yaw_rate.mean(),
            "rew_lateral_world": rew_lateral_world.mean(),

            "rew_foot_height": r_foot_height.mean(),
            "rew_foot_height_usage_balance":rew_foot_height_usage_balance.mean(),

            # 전체
            "rew_total": total_reward.mean(),
        }
        self._q_log_counter += 1
        if self._q_log_counter % 500 == 0:
            print("lf_body_id:", self._lf_body_id, "rf_body_id:", self._rf_body_id)
            print("lf_z mean/min/max:",
                lf_z.mean().detach().cpu().item(),
                lf_z.min().detach().cpu().item(),
                lf_z.max().detach().cpu().item(),
            )
            print("rf_z mean/min/max:",
                rf_z.mean().detach().cpu().item(),
                rf_z.min().detach().cpu().item(),
                rf_z.max().detach().cpu().item(),
            )
        self.extras["episode"]["base_height"] = base_height.mean()

        return total_reward

    def _get_dones(self):
        root = self.robot.data.root_state_w
        pos = root[:, 0:3]
        linvel = root[:, 7:10]
        angvel = root[:, 10:13]
        base_quat = root[:, 3:7]  # expected (qw,qx,qy,qz)

        # Prefer COM height if available
        if hasattr(self.robot.data, "root_com_pos_w"):
            base_height = self.robot.data.root_com_pos_w[:, 2]
        elif hasattr(self.robot.data, "com_pos_w"):
            base_height = self.robot.data.com_pos_w[:, 2]
        else:
            base_height = pos[:, 2]

        tilt_angle = compute_tilt_from_quat_wxyz(base_quat)

        max_tilt = float(max(self.cfg.max_base_pitch, self.cfg.max_base_roll))

        # Safety
        bad_nan = torch.isnan(root).any(dim=1) | torch.isinf(root).any(dim=1)
        bad_oob = (pos.abs().max(dim=1).values > 100.0)
        bad_vel = (linvel.abs().max(dim=1).values > 20.0) | (angvel.abs().max(dim=1).values > 50.0)
        bad = bad_nan | bad_oob | bad_vel

        # Raw conditions
        low = base_height < float(self.cfg.min_base_height)
        tilt = tilt_angle > max_tilt

        # Debounce
        self._low_count = torch.where(low, self._low_count + 1, torch.zeros_like(self._low_count))
        self._tilt_count = torch.where(tilt, self._tilt_count + 1, torch.zeros_like(self._tilt_count))

        low_term_steps = int(getattr(self.cfg, "low_term_steps", 12))
        tilt_term_steps = int(getattr(self.cfg, "tilt_term_steps", 8))

        low_term = self._low_count >= low_term_steps
        tilt_term = self._tilt_count >= tilt_term_steps

        fallen = low_term | tilt_term | bad

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = time_out & ~fallen

        return fallen, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset joints
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Reset root over env origins
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Buffers
        self.actions[env_ids].zero_()
        self.last_actions[env_ids].zero_()
        self.joint_torques[env_ids].zero_()

        self._low_count[env_ids] = 0
        self._tilt_count[env_ids] = 0

        # Write to sim
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.prev_base_pos[env_ids] = self.robot.data.root_state_w[env_ids, 0:3]

        self.imu_bias_lin[env_ids] = (
            torch.randn((len(env_ids),3),device=self.device)*0.01
        )

        self.imu_bias_ang[env_ids] = (
            torch.randn((len(env_ids),3),device=self.device)*0.005
        )

        self.lf_z_sum[env_ids] = 0.0
        self.rf_z_sum[env_ids] = 0.0
        self.z_count[env_ids] = 0.0
        self.phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device)

# =============================================================================
# Helpers (TorchScript-safe where it matters)
# =============================================================================

@torch.jit.script
def quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: [N,4] = (qw,qx,qy,qz), v: [N,3]
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    vpx = vx + qw * tx + (qy * tz - qz * ty)
    vpy = vy + qw * ty + (qz * tx - qx * tz)
    vpz = vz + qw * tz + (qx * ty - qy * tx)

    return torch.stack((vpx, vpy, vpz), dim=1)


@torch.jit.script
def compute_tilt_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    # quat_wxyz: [N,4] = (qw, qx, qy, qz) -> tilt angle [0..pi]
    qw = quat_wxyz[:, 0]
    qx = quat_wxyz[:, 1]
    qy = quat_wxyz[:, 2]
    qz = quat_wxyz[:, 3]

    # R33 = 1 - 2*(x^2 + y^2) for quaternion (w,x,y,z)
    z_wz = 1.0 - 2.0 * (qx * qx + qy * qy)
    return torch.acos(torch.clamp(z_wz, -1.0, 1.0))


@torch.jit.script
def compute_rewards(
    # scales
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_forward_vel: float,
    rew_scale_upright: float,
    rew_scale_joint_vel: float,
    rew_scale_action_rate: float,
    rew_scale_energy: float,
    rew_vel_track_rate: float,
    rew_fheight_rate: float,
    rew_leg_pose_rate: float,

    # robot state
    base_quat: torch.Tensor,            # [N,4] (qw,qx,qy,qz)
    base_ang_vel: torch.Tensor,         # [N,3]
    base_lin_vel: torch.Tensor,         # [N,3]
    base_height: torch.Tensor,          # [N]
    tilt_angle: torch.Tensor,           # [N]
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_torques: torch.Tensor,
    action_rate: torch.Tensor,
    joint_nominal: torch.Tensor,        # [N,D]
    base_height_target: float,
    reset_terminated: torch.Tensor,     # [N] bool/int
    rew_heading_rate:float,

    lf_z: torch.Tensor,
    rf_z: torch.Tensor,
    lf_z_avg: torch.Tensor,
    rf_z_avg: torch.Tensor,
    ground_z:float,
    swing_z:float,
    phase:torch.Tensor,

) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

# ) -> torch.Tensor:

    # Alive / termination
    alive_term = (1.0 - reset_terminated.float())
    rew_alive = rew_scale_alive * alive_term
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # Robot frame axes in world
    N = base_quat.shape[0]
    axis_x = base_quat.new_zeros((N, 3))
    axis_y = base_quat.new_zeros((N, 3))
    axis_z = base_quat.new_zeros((N, 3))
    axis_x[:, 0] = 1.0
    axis_y[:, 1] = 1.0
    axis_z[:, 2] = 1.0

    fwd_w = quat_apply_wxyz(base_quat, axis_x)

    # If true forward is -X
    forward_sign = -1.0
    fwd_true = forward_sign * fwd_w
    # v_fwd = torch.sum(base_lin_vel * fwd_true, dim=1)
    v_cmd = 0.18 # 0.2
    v_min = 0.04
    # Forward reward (clipped)
    target_dir = base_quat.new_zeros((N, 3))
    target_dir[:, 0] = -1.0

    v_fwd_world = torch.sum(base_lin_vel * target_dir, dim=1)
    v_eff = torch.clamp(v_fwd_world - v_min, 0.0, v_cmd - v_min)
    rew_forward = rew_scale_forward_vel * v_eff

    fwd_xy = fwd_true[:, 0:2]
    tgt_xy = target_dir[:, 0:2]
    fwd_xy = fwd_xy / (torch.linalg.norm(fwd_xy, dim=1, keepdim=True) + 1e-6)
    tgt_xy = tgt_xy / (torch.linalg.norm(tgt_xy, dim=1, keepdim=True) + 1e-6)

    heading_cos = torch.sum(fwd_xy * tgt_xy, dim=1)
    rew_heading = rew_heading_rate * (1.0 - heading_cos).square()

    # Velocity tracking (gaussian peak at v_cmd, baseline-shifted to ~0 at v=0)
    k = 40.0 # 2.0 클수록 목표 속도 주변만 좁게 포함
    track = torch.exp(-k * (v_fwd_world - v_cmd).square())
    baseline =  torch.exp(base_quat.new_full((), -k * v_cmd * v_cmd))

    rew_vel_track = rew_vel_track_rate * torch.clamp(
        (track - baseline) / (1.0 - baseline + 1e-6),
        0.0,
        1.0,
    )

    # Upright / height stabilization
    height_err = base_height - base_height_target
    upright_term = torch.exp(-50.0 * height_err * height_err - 4.0 * tilt_angle * tilt_angle)
    rew_upright = rew_scale_upright * upright_term

    # Small penalties 잔발 억제 / 값은 작게, 크면 걷기를 방해함
    # rew_joint_vel = rew_scale_joint_vel * torch.sum(joint_vel * joint_vel, dim=1)
    # rew_action_rate = rew_scale_action_rate * torch.sum(action_rate * action_rate, dim=1)
    # rew_energy = rew_scale_energy * torch.sum(torch.abs(joint_torques * joint_vel), dim=1)

    # ==========================
    # Straight leg penalty
    # ==========================

    leg_error = joint_pos - joint_nominal

    rew_leg_pose = (
        rew_leg_pose_rate *
        torch.sum(leg_error * leg_error, dim=1)
    )

    # 발 높이
    stance_sigma = 0.015
    swing_sigma = 0.030
    left_stance = phase < 0.5

    stance_z = torch.where(left_stance, lf_z, rf_z)
    swing_z_now = torch.where(left_stance, rf_z, lf_z)

    stance_err = stance_z - ground_z
    swing_err = swing_z_now - swing_z

    rew_gait_height = (
        torch.exp(-((stance_err) / stance_sigma).square())
        * torch.exp(-((swing_err) / swing_sigma).square())
    )

    rew_foot_height_usage_balance = -2.0 * (100.0 * (lf_z_avg - rf_z_avg)).square()

    rew_foot_height = rew_fheight_rate * rew_gait_height

    foot_z_axis = base_quat.new_zeros((N, 3))
    foot_z_axis[:, 2] = 1.0

    rew_yaw_rate = -0.3 * base_ang_vel[:, 2].square()

    # 대각선 진행 패널티
    v_side_world = base_lin_vel[:, 1]
    rew_lateral_world =  -10.0 * v_side_world.square()


    # 게이트 품질에 따른 보상
    gait_quality = rew_gait_height  # 0~1 근처

    rew_vel_track = rew_vel_track * gait_quality #(0.5 + 0.5 * gait_quality)
    rew_forward = rew_forward * gait_quality #(0.7 + 0.3 * gait_quality)

    total = (
        rew_alive
        + rew_termination
        + rew_upright
        + rew_forward
        + rew_vel_track
        + rew_heading

        + rew_foot_height

        + rew_yaw_rate
        + rew_lateral_world
        # + rew_joint_vel
        # + rew_action_rate

        + rew_foot_height_usage_balance

    )
    return (
        total,

        rew_alive,
        rew_termination,
        rew_upright,

        rew_forward,
        rew_vel_track,
        rew_heading,

        rew_foot_height,

        rew_yaw_rate,
        rew_lateral_world,
        # rew_joint_vel,
        # rew_action_rate,

        rew_foot_height_usage_balance
    )