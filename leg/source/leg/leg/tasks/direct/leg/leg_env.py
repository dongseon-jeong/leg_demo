# leg_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .leg_env_cfg import LegEnvCfg


class LegEnv(DirectRLEnv):
    cfg: LegEnvCfg

    def __init__(self, cfg: LegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # find DOF indices for the controlled joints
        # cfg.joint_names: list of 12 joint names (lbase_joint, rbase_joint, ...)
        self._dof_indices, _ = self.robot.find_joints(self.cfg.joint_names)

        # convenient references to simulation buffers
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # buffer for actions / previous actions / torques
        num_dofs = len(self.cfg.joint_names)
        self.actions = torch.zeros(self.num_envs, num_dofs, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.joint_torques = torch.zeros_like(self.actions)

    # --------------------------------------------------------------------- #
    # Scene setup
    # --------------------------------------------------------------------- #

    def _setup_scene(self):
        # create robot articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # register articulation in the scene
        self.scene.articulations["robot"] = self.robot

        # add a light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------------------- #
    # RL interface
    # --------------------------------------------------------------------- #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # clip to [-1, 1] for safety
        actions = torch.clamp(actions, -1.0, 1.0)

        # store previous actions for action_rate penalty
        self.last_actions[:] = self.actions
        self.actions[:] = actions

    def _apply_action(self) -> None:
        # scale actions to torques
        torques = self.actions * self.cfg.action_scale

        # optional torque limit
        if self.cfg.torque_limit > 0.0:
            torques = torch.clamp(torques, -self.cfg.torque_limit, self.cfg.torque_limit)

        # apply torques only to controlled joints
        self.robot.set_joint_effort_target(torques, joint_ids=self._dof_indices)

        # save for energy penalty
        self.joint_torques[:] = torques

    def _get_observations(self) -> dict:
        # refresh joint states
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # root state: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        root_state = self.robot.data.root_state_w
        base_pos = root_state[:, 0:3]
        base_height = base_pos[:, 2]
        base_lin_vel = root_state[:, 7:10]
        base_ang_vel = root_state[:, 10:13]
        base_quat = root_state[:, 3:7]

        # tilt angle from quaternion (angle between base and world up)
        tilt_angle = _compute_tilt_from_quat(base_quat)  # [N]

        # joint states (only controlled joints)
        q = self.joint_pos[:, self._dof_indices]
        qd = self.joint_vel[:, self._dof_indices]

        # previous actions (for smoothness)
        act = self.actions
        last_act = self.last_actions
        action_rate = act - last_act

        # build 48-D observation:
        #  - 12 joint positions
        #  - 12 joint velocities
        #  - 3 base linear velocity
        #  - 3 base angular velocity
        #  - 1 base height
        #  - 1 tilt angle
        #  - 12 previous actions
        #  - 4 extra: forward velocity, x position, y position, height error
        forward_vel = base_lin_vel[:, 0].unsqueeze(-1)
        height_err = (base_height - self.cfg.base_height_target).unsqueeze(-1)
        x_pos = base_pos[:, 0].unsqueeze(-1)
        y_pos = base_pos[:, 1].unsqueeze(-1)

        obs = torch.cat(
            (
                q,                                     # 12
                qd,                                    # 12 -> 24
                base_lin_vel,                          # 3  -> 27
                base_ang_vel,                          # 3  -> 30
                base_height.unsqueeze(-1),             # 1  -> 31
                tilt_angle.unsqueeze(-1),              # 1  -> 32
                last_act,                              # 12 -> 44
                forward_vel,                           # 1  -> 45
                x_pos,                                 # 1  -> 46
                y_pos,                                 # 1  -> 47
                height_err,                            # 1  -> 48
            ),
            dim=-1,
        )

        # NOTE: obs.shape[-1] == 48, so LegEnvCfg.observation_space must be 48.
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # read states
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        root_state = self.robot.data.root_state_w
        base_pos = root_state[:, 0:3]
        base_height = base_pos[:, 2]
        base_lin_vel = root_state[:, 7:10]
        base_quat = root_state[:, 3:7]

        tilt_angle = _compute_tilt_from_quat(base_quat)  # [N]

        qd = self.joint_vel[:, self._dof_indices]
        torques = self.joint_torques
        action_rate = self.actions - self.last_actions

        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_forward_vel,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_energy,
            base_lin_vel,
            base_height,
            tilt_angle,
            qd,
            torques,
            action_rate,
            self.cfg.base_height_target,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.root_state_w
        base_pos = root_state[:, 0:3]
        base_height = base_pos[:, 2]
        base_quat = root_state[:, 3:7]

        tilt_angle = _compute_tilt_from_quat(base_quat)

        # termination if:
        #  - base too low
        #  - tilt too large
        max_tilt = max(self.cfg.max_base_pitch, self.cfg.max_base_roll)

        fallen = (base_height < self.cfg.min_base_height) | (tilt_angle > max_tilt)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # reset joint states to defaults (standing pose)
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # small joint noise (optional)
        # joint_pos += 0.02 * torch.randn_like(joint_pos)

        # reset root state: position over env origin
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # reset buffers
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.joint_torques[env_ids] = 0.0

        # write to sim
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


# ------------------------------------------------------------------------- #
# Helper & reward functions
# ------------------------------------------------------------------------- #


def _compute_tilt_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Compute tilt angle of base from quaternion.

    quat: [N, 4] with ordering [qx, qy, qz, qw] (Isaac convention).
    Returns angle in radians between base z-axis and world up.
    """
    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]

    # rotation angle from quaternion
    sin_half = torch.sqrt(qx * qx + qy * qy + qz * qz)
    cos_half = torch.clamp(qw, -1.0, 1.0)
    tilt = 2.0 * torch.atan2(sin_half, cos_half)
    return tilt


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_forward_vel: float,
    rew_scale_upright: float,
    rew_scale_joint_vel: float,
    rew_scale_action_rate: float,
    rew_scale_energy: float,
    base_lin_vel: torch.Tensor,      # [N, 3]
    base_height: torch.Tensor,       # [N]
    tilt_angle: torch.Tensor,        # [N]
    joint_vel: torch.Tensor,         # [N, D]
    joint_torques: torch.Tensor,     # [N, D]
    action_rate: torch.Tensor,       # [N, D]
    base_height_target: float,
    reset_terminated: torch.Tensor,  # [N]
):
    # alive / termination
    alive_term = (1.0 - reset_terminated.float())
    rew_alive = rew_scale_alive * alive_term
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # forward velocity along +x
    forward_vel = base_lin_vel[:, 0]
    rew_forward = rew_scale_forward_vel * forward_vel

    # upright reward: near target height and small tilt
    height_err = base_height - base_height_target
    # gaussian-like reward around (height_err=0, tilt=0)
    upright_term = torch.exp(-5.0 * height_err * height_err - 2.0 * tilt_angle * tilt_angle)
    rew_upright = rew_scale_upright * upright_term

    # joint velocity penalty
    rew_joint_vel = rew_scale_joint_vel * torch.sum(joint_vel * joint_vel, dim=1)

    # action rate penalty
    rew_action_rate = rew_scale_action_rate * torch.sum(action_rate * action_rate, dim=1)

    # energy penalty (|tau * qdot|)
    rew_energy = rew_scale_energy * torch.sum(torch.abs(joint_torques * joint_vel), dim=1)

    total_reward = (
        rew_alive
        + rew_termination
        + rew_forward
        + rew_upright
        + rew_joint_vel
        + rew_action_rate
        + rew_energy
    )
    return total_reward
