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

        self._left_contact: ContactSensor = self.scene.sensors["left_leg_contact"]
        self._right_contact: ContactSensor = self.scene.sensors["right_leg_contact"]
        self._low_height_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        self._dof_indices, _ = self.robot.find_joints(self.cfg.joint_names)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        num_dofs = len(self.cfg.joint_names)
        self.actions = torch.zeros(self.num_envs, num_dofs, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.joint_torques = torch.zeros_like(self.actions)

        # ✅ 추가: cfg에서 만든 센서 2개 핸들 가져오기
        self._left_leg_contact = self.scene["left_leg_contact"]
        self._right_leg_contact = self.scene["right_leg_contact"]

        self._low_count  = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._tilt_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    # --------------------------------------------------------------------- #
    # Scene setup
    # --------------------------------------------------------------------- #

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)

        # ✅ ContactReport API 디버그/강제 적용 (ContactSensor 초기화 전에!)
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics, PhysxSchema

            stage = omni.usd.get_context().get_stage()
            root_path = "/World/envs/env_0/legs/legs"
            root = stage.GetPrimAtPath(root_path)

            if not root.IsValid():
                print(f"[DBG] root not found: {root_path}", flush=True)
            else:
                total = 0
                rigid = 0
                had = 0
                applied = 0

                for prim in Usd.PrimRange(root):
                    total += 1
                    name = prim.GetName()

                    # ll* / rl* 링크 후보만 대상으로
                    if not (name.startswith("ll") or name.startswith("rl")):
                        continue

                    # 1) RigidBodyAPI 있는지
                    is_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    if is_rigid:
                        rigid += 1

                    # 2) ContactReportAPI 있는지 / 없으면 apply
                    if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                        had += 1
                    else:
                        # RigidBody가 아니더라도 일단 적용 시도 (Articulation link여도 붙는 경우가 있음)
                        PhysxSchema.PhysxContactReportAPI.Apply(prim)
                        applied += 1

                print(
                    f"[DBG] scanned={total}  rigid={rigid}  had_report={had}  applied_report={applied}  under={root_path}",
                    flush=True,
                )

        except Exception as e:
            print("[DBG] contact report patch failed:", e, flush=True)
        # ✅ 여기까지 추가

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        print("[DBG] robot prim:", self.cfg.robot_cfg.prim_path, flush=True)
        print("[DBG] left sensor prim:", self.cfg.scene.left_leg_contact.prim_path, flush=True)

    # --------------------------------------------------------------------- #
    # RL interface
    # --------------------------------------------------------------------- #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # # clip to [-1, 1] for safety
        # actions = torch.clamp(actions, -1.0, 1.0)
        # # store previous actions for action_rate penalty
        # self.last_actions[:] = self.actions
        # self.actions[:] = actions

        alpha = 0.2  # 0~1, 작을수록 부드러움
        actions = torch.clamp(actions, -1.0, 1.0)
        self.last_actions[:] = self.actions
        self.actions[:] = (1 - alpha) * self.actions + alpha * actions

    def _apply_action(self) -> None:
        # actions in [-1, 1]
        actions = torch.clamp(self.actions, -1.0, 1.0)

        # step-based ramp (0~1) : first 0.5s
        ramp_steps = int(0.5 / (self.cfg.sim.dt * self.cfg.decimation))  # 0.5초
        ramp = torch.clamp(
            self.episode_length_buf.float() / float(max(ramp_steps, 1)),
            0.0,
            1.0,
        ).unsqueeze(-1)  # [N,1]

        # default pose for controlled joints
        q0 = self.robot.data.default_joint_pos[:, self._dof_indices]  # [N, D]

        # action_scale is now "radian delta" (e.g., 0.25~0.5)
        q_target = q0 + (actions * self.cfg.action_scale) * ramp  # [N, D]

        # PD position target (ImplicitActuatorCfg stiffness/damping will be used)
        self.robot.set_joint_position_target(q_target, joint_ids=self._dof_indices)

        # (optional) If you still want "energy penalty", torque is not directly available.
        # Use a proxy: squared position error * ramp (or |qd|) etc.
        # Here we just store something compatible shape-wise:
        self.joint_torques[:] = 0.0


    def _get_observations(self) -> dict:
        # refresh joint states
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # root state: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        root_state = self.robot.data.root_state_w
        base_pos = root_state[:, 0:3]


        # (추천) COM 기반 height 우선
        if hasattr(self.robot.data, "root_com_pos_w"):
            base_height = self.robot.data.root_com_pos_w[:, 2]
        elif hasattr(self.robot.data, "com_pos_w"):
            # 보통 [N,3] 형태
            base_height = self.robot.data.com_pos_w[:, 2]
        else:
            # fallback: base_link 원점
            base_height = base_pos[:, 2]


        # base_height = base_pos[:, 2]
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

        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, -100.0, 100.0)   # 선택: 폭주 값 제한

        # NOTE: obs.shape[-1] == 48, so LegEnvCfg.observation_space must be 48.
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # read states
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        root_state = self.robot.data.root_state_w
        base_pos = root_state[:, 0:3]

        # (추천) COM 기반 height 우선
        if hasattr(self.robot.data, "root_com_pos_w"):
            base_height = self.robot.data.root_com_pos_w[:, 2]
        elif hasattr(self.robot.data, "com_pos_w"):
            # 보통 [N,3] 형태
            base_height = self.robot.data.com_pos_w[:, 2]
        else:
            # fallback: base_link 원점
            base_height = base_pos[:, 2]

        # base_height = base_pos[:, 2]
        base_lin_vel = root_state[:, 7:10]
        base_quat = root_state[:, 3:7]
        base_ang_vel = root_state[:, 10:13]

        tilt_angle = _compute_tilt_from_quat(base_quat)  # [N]

        qd = self.joint_vel[:, self._dof_indices]
        torques = self.joint_torques
        action_rate = self.actions - self.last_actions



        if int(self.common_step_counter) % 500 == 0:
            N = base_quat.shape[0]
            axis_x = base_quat.new_zeros((N, 3)); axis_x[:,0]=1.0
            axis_y = base_quat.new_zeros((N, 3)); axis_y[:,1]=1.0
            fwd_w = quat_apply_wxyz(base_quat, axis_x)
            lat_w = quat_apply_wxyz(base_quat, axis_y)
            v_fwd = torch.sum(base_lin_vel * fwd_w, dim=1)
            v_lat = torch.sum(base_lin_vel * lat_w, dim=1)

            # -x가 정면이면 여기서도 동일하게
            v_fwd = -v_fwd

            print("[reward DBG] v_fwd mean/max:", v_fwd.mean().item(), v_fwd.max().item(), flush=True)
            print("[reward DBG] v_lat mean/absmax:", v_lat.mean().item(), v_lat.abs().max().item(), flush=True)
            print("[reward DBG] linvel world x/y mean:", base_lin_vel[:,0].mean().item(), base_lin_vel[:,1].mean().item(), flush=True)

            base_pos = self.robot.data.root_state_w[:, 0:3]
            base_lin_vel = self.robot.data.root_state_w[:, 7:10]
            print("[reward DBG] base_x mean/min/max:",
                base_pos[:,0].mean().item(), base_pos[:,0].min().item(), base_pos[:,0].max().item(),
                flush=True)
            print("[reward DBG] v_fwd < 0 ratio:", (v_fwd < 0).float().mean().item(), flush=True)
            print("[reward DBG] |v_lat| > 0.2 ratio:", (v_lat.abs() > 0.2).float().mean().item(), flush=True)

        # ✅ leg-leg interference (use two scene sensors)
        fm_l = self._left_contact.data.force_matrix_w   # [N, L, F, 3] or [N,1,F,3]
        fm_r = self._right_contact.data.force_matrix_w

        # 안전장치: None이면 0
        if fm_l is None:
            max_l = torch.zeros(self.num_envs, device=self.device)
        else:
            mag_l = torch.linalg.norm(fm_l, dim=-1)                 # [N, L, F]
            max_l = mag_l.max(dim=-1).values.max(dim=-1).values     # [N]

        if fm_r is None:
            max_r = torch.zeros(self.num_envs, device=self.device)
        else:
            mag_r = torch.linalg.norm(fm_r, dim=-1)
            max_r = mag_r.max(dim=-1).values.max(dim=-1).values

        # env별 최대 간섭힘
        leg_interf_max_force = torch.maximum(max_l, max_r)          # [N]

        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_forward_vel,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_energy,


            # ✅ 추가 2개
            self.cfg.rew_scale_leg_interference,
            self.cfg.leg_interference_force_threshold,
            leg_interf_max_force,

            base_quat,
            base_ang_vel,

            base_lin_vel,
            base_height,
            tilt_angle,
            qd,
            torques,
            action_rate,
            self.cfg.base_height_target,
            self.reset_terminated,
        )

        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)
        total_reward = torch.clamp(total_reward, -100.0, 100.0)  # 선택

        return total_reward

    def _get_dones(self):
        root = self.robot.data.root_state_w
        pos = root[:, 0:3]
        linvel = root[:, 7:10]
        angvel = root[:, 10:13]


        # (추천) COM 기반 height 우선
        if hasattr(self.robot.data, "root_com_pos_w"):
            base_height = self.robot.data.root_com_pos_w[:, 2]
        elif hasattr(self.robot.data, "com_pos_w"):
            # 보통 [N,3] 형태
            base_height = self.robot.data.com_pos_w[:, 2]
        else:
            # fallback: base_link 원점
            base_height = pos[:, 2]


        # base_height = pos[:, 2]
        base_quat = root[:, 3:7]
        tilt_angle = _compute_tilt_from_quat(base_quat)

        max_tilt = max(self.cfg.max_base_pitch, self.cfg.max_base_roll)

        # safety
        bad_nan = torch.isnan(root).any(dim=1) | torch.isinf(root).any(dim=1)
        bad_oob = (pos.abs().max(dim=1).values > 100.0)
        bad_vel = (linvel.abs().max(dim=1).values > 20.0) | (angvel.abs().max(dim=1).values > 50.0)
        bad = bad_nan | bad_oob | bad_vel

        # raw conditions
        low  = base_height < self.cfg.min_base_height
        tilt = tilt_angle > max_tilt

        # debounce counters
        self._low_count  = torch.where(low,  self._low_count + 1, torch.zeros_like(self._low_count))
        self._tilt_count = torch.where(tilt, self._tilt_count + 1, torch.zeros_like(self._tilt_count))

        low_term_steps  = 3   # 너가 원한 low 3-step
        tilt_term_steps = 2   # 추천: tilt는 1~2면 충분 (누워있는 걸 빨리 끊어야 함)

        low_term  = self._low_count  >= low_term_steps
        tilt_term = self._tilt_count >= tilt_term_steps

        fallen = low_term | tilt_term | bad

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = time_out & ~fallen

        below0 = (base_height < 0.0).float().mean().item()
        belowMin = (base_height < self.cfg.min_base_height).float().mean().item()

        # debug (원하면 유지)
        if int(self.common_step_counter) % 200 == 0:
            print("[DONE] low/low_term/tilt/tilt_term/bad:",
                low.float().mean().item(),
                low_term.float().mean().item(),
                tilt.float().mean().item(),
                tilt_term.float().mean().item(),
                bad.float().mean().item(),
                flush=True)
            print("[DBG] max_tilt:", float(max_tilt),
                "tilt mean/min/max:",
                tilt_angle.mean().item(), tilt_angle.min().item(), tilt_angle.max().item(),
                flush=True)
            print("[DBG] base_quat mean:", 
                base_quat.mean(dim=0).tolist(), 
                flush=True)
            print(
                "[DBG] base_height mean/min/max:",
                base_height.mean().item(),
                base_height.min().item(),
                base_height.max().item(),
                "min_base_height:",
                float(self.cfg.min_base_height),
                flush=True,
            )
            print(f"[DBG] base_height < 0 ratio: {below0:.4f}, < min_base_height ratio: {belowMin:.4f}", flush=True)

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

        self._low_height_count[env_ids] = 0

        self._low_count[env_ids] = 0
        self._tilt_count[env_ids] = 0

        # write to sim
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


# ------------------------------------------------------------------------- #
# Helper & reward functions
# ------------------------------------------------------------------------- #

# “월드 x” 대신 “로봇 forward”
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


def _compute_tilt_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """quat_wxyz: [N,4] = (qw, qx, qy, qz). Returns tilt angle (0..pi)."""
    qw = quat_wxyz[:, 0]
    qx = quat_wxyz[:, 1]
    qy = quat_wxyz[:, 2]
    qz = quat_wxyz[:, 3]

    # Rotation matrix element R33 for body z-axis vs world up
    # R33 = 1 - 2*(x^2 + y^2) for quaternion (w,x,y,z)
    z_wz = 1.0 - 2.0 * (qx * qx + qy * qy)

    tilt = torch.acos(torch.clamp(z_wz, -1.0, 1.0))
    return tilt


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_forward_vel: float,   # 전진(클립) 보상 스케일
    rew_scale_upright: float,
    rew_scale_joint_vel: float,
    rew_scale_action_rate: float,
    rew_scale_energy: float,

    rew_scale_leg_interference: float,
    leg_interference_force_threshold: float,
    leg_interf_max_force: torch.Tensor,   # [N]

    base_quat: torch.Tensor,             # [N,4] (qw,qx,qy,qz)
    base_ang_vel: torch.Tensor,          # [N,3]  (world)
    base_lin_vel: torch.Tensor,          # [N,3]  (world)
    base_height: torch.Tensor,           # [N]
    tilt_angle: torch.Tensor,            # [N]
    joint_vel: torch.Tensor,             # [N,D]
    joint_torques: torch.Tensor,         # [N,D]
    action_rate: torch.Tensor,           # [N,D]
    base_height_target: float,
    reset_terminated: torch.Tensor,      # [N]
):
    # 1) alive / termination
    alive_term = (1.0 - reset_terminated.float())
    rew_alive = rew_scale_alive * alive_term
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # 2) robot-frame velocity decomposition (heading-aware)
    N = base_quat.shape[0]
    axis_x = base_quat.new_zeros((N, 3)); axis_x[:, 0] = 1.0
    axis_y = base_quat.new_zeros((N, 3)); axis_y[:, 1] = 1.0
    axis_z = base_quat.new_zeros((N, 3)); axis_z[:, 2] = 1.0

    fwd_w = quat_apply_wxyz(base_quat, axis_x)  # [N,3]
    lat_w = quat_apply_wxyz(base_quat, axis_y)  # [N,3]
    up_w  = quat_apply_wxyz(base_quat, axis_z)  # [N,3]

    v_fwd = torch.sum(base_lin_vel * fwd_w, dim=1)   # [N]
    v_lat = torch.sum(base_lin_vel * lat_w, dim=1)   # [N]
    yaw_rate = torch.sum(base_ang_vel * up_w, dim=1) # [N]  (robot up축 기준 yaw)

    # ✅ 로봇 정면이 -X라면 여기만 -1로
    forward_sign = -1.0
    v_fwd = forward_sign * v_fwd

    # (A) 전진 속도 보상(클립)
    v_fwd_clip = torch.clamp(v_fwd, 0.0, 2.0)
    rew_forward = rew_scale_forward_vel * v_fwd_clip

    # (B) 속도 트래킹 (정지에서 “별로” / v_cmd에서 “최고”)
    v_cmd = 0.4
    k = 4.0
    # v=0일 때도 확실히 불리하게 만들기 (baseline subtraction)
    baseline = torch.exp(-k * (0.0 - v_cmd) * (0.0 - v_cmd))
    rew_vel_track = 2.0 * (torch.exp(-k * (v_fwd - v_cmd) * (v_fwd - v_cmd)) - baseline)

    # (C) 정지 억제 (v_fwd가 너무 작으면 패널티)
    v_eps = 0.05
    stand_pen = torch.relu(v_eps - v_fwd)
    rew_stand_pen = -2.5 * (stand_pen * stand_pen)

    # (D) crab-walk / yaw 강 억제
    rew_lat_pen = -2.0 * (v_lat * v_lat)        # ← 기존 -1.0 보다 강하게
    rew_yaw_pen = -0.5 * (yaw_rate * yaw_rate)  # ← 기존 -0.1 보다 강하게

    # (E) heading 정렬 보상: "월드에서 지정된 방향"을 바라보게
    # 원하는 진행방향이 -X면 target_dir=[-1,0,0]
    target_dir = base_quat.new_zeros((N, 3))
    target_dir[:, 0] = -1.0

    # 수평 성분만 비교
    fwd_xy = fwd_w[:, 0:2]
    tgt_xy = target_dir[:, 0:2]
    fwd_xy = fwd_xy / (torch.linalg.norm(fwd_xy, dim=1, keepdim=True) + 1e-6)
    tgt_xy = tgt_xy / (torch.linalg.norm(tgt_xy, dim=1, keepdim=True) + 1e-6)
    heading_cos = torch.sum(fwd_xy * tgt_xy, dim=1)   # [-1,1]
    rew_heading = 1.0 * torch.clamp(heading_cos, 0.0, 1.0)  # 음수는 0

    # 3) upright reward
    height_err = base_height - base_height_target
    upright_term = torch.exp(-5.0 * height_err * height_err - 4.0 * tilt_angle * tilt_angle)
    rew_upright = rew_scale_upright * upright_term

    # 4) small penalties
    rew_joint_vel   = rew_scale_joint_vel   * torch.sum(joint_vel * joint_vel, dim=1)
    rew_action_rate = rew_scale_action_rate * torch.sum(action_rate * action_rate, dim=1)
    rew_energy      = rew_scale_energy      * torch.sum(torch.abs(joint_torques * joint_vel), dim=1)

    # 5) leg interference penalty (교차 억제용으로 조금 더 공격적으로)
    excess = torch.clamp(leg_interf_max_force - leg_interference_force_threshold, min=0.0)
    excess = torch.clamp(excess, max=200.0)
    rew_leg_interf = rew_scale_leg_interference * (excess / 200.0)

    total_reward = (
        rew_alive
        + rew_termination
        + rew_upright
        + rew_forward
        + rew_vel_track
        + rew_heading
        + rew_stand_pen
        + rew_lat_pen
        + rew_yaw_pen
        + rew_leg_interf
        + rew_joint_vel
        + rew_action_rate
        + rew_energy
    )
    return total_reward
