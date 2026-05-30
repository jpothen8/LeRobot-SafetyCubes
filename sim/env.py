"""SafeCubeEnv — MuJoCo environment for constraint-aware VLA training.

Observation (the policy sees this):
    image        uint8 (H, W, 3)   agentview (main) camera
    wrist_image  uint8 (H, W, 3)   wrist-mounted camera
    state        float32 (n_arm_joints + 1,)   joint positions + gripper

Privileged info (rides in info dict; only the *loss* may read it):
    cube_positions       float32 (n_red, 3)
    cube_half_extents    float32 (n_red, 3)
    blue_cube_pos        float32 (3,)
    goal_pos             float32 (3,)
    ee_pos               float32 (3,)
    grasped              bool

Step accepts either a single action vector (shape n_arm + 1) or an action chunk
(shape T x (n_arm + 1)). With a chunk, the env runs the whole chunk open-loop
and returns the *last* obs / aggregated termination flags. This matches π0's
action-chunking inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .configs import EnvConfig, SceneConfig
from .scene import Layout, build_scene, sample_layout


@dataclass
class EpisodeStats:
    steps: int = 0
    red_contact: bool = False
    ceiling_violation: bool = False
    blue_dropped: bool = False
    success: bool = False
    dwell: int = 0
    min_clearance: float = float("inf")
    closest_red_idx: int = -1
    contact_history: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "red_contact": bool(self.red_contact),
            "ceiling_violation": bool(self.ceiling_violation),
            "blue_dropped": bool(self.blue_dropped),
            "success": bool(self.success),
            "dwell": self.dwell,
            "min_clearance": float(self.min_clearance),
            "closest_red_idx": self.closest_red_idx,
        }


class SafeCubeEnv:
    """Bare-bones gym-like env. Not registered with gymnasium intentionally —
    keeps it independent of the lerobot.envs factory until we want to plug in."""

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self.layout: Layout | None = None
        self._stats = EpisodeStats()

        # Cached indices computed in reset().
        self._arm_qpos_idx: list[int] = []
        self._arm_ctrl_idx: list[int] = []
        self._gripper_qpos_idx: int = -1
        self._gripper_ctrl_idx: int = -1
        self._red_body_ids: list[int] = []
        self._blue_body_id: int = -1
        self._red_geom_ids: set[int] = set()
        self._blue_geom_ids: set[int] = set()
        self._arm_geom_ids: set[int] = set()
        self._gripper_geom_ids: set[int] = set()
        # Indices for the blue cube's free joint (used by the magnetic-grip
        # logic to kinematically attach the cube to the gripper).
        self._blue_qpos_start: int = -1
        self._blue_dof_start: int = -1
        # Magnetic-grip state.
        self._attached: bool = False
        # Tunables for the grip proxy. The grasp activates only once the
        # gripper joint has *physically* closed (qpos near grip_closed=-0.1),
        # not on the ctrl signal — so the cube doesn't snap to the gripper
        # before the jaws have moved.
        # With grip_open ≈ 0.6, set release halfway between closed (-0.1) and
        # open (0.6) → 0.40 rad. Hysteresis: attach < 0.20 closes; release > 0.40 opens.
        self._grip_qpos_attach_threshold: float = 0.20
        self._grip_qpos_release_threshold: float = 0.40
        self._grip_attach_radius: float = 0.04
        # Vertical offset between the grasp site (jaw-gap center) and the cube
        # center while held. The grasp site sits at the fingertip plane and the
        # cube center is essentially coincident with it, so this is ~0.
        self._grip_cube_offset_z: float = 0.0
        self._ee_site_id: int = -1
        self._grasp_site_id: int = -1
        self._goal_site_id: int = -1

    # ----- public API -----------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.layout = sample_layout(self.cfg.scene, self._rng)
        self.model, _ = build_scene(self.cfg.scene, self.layout)
        self.data = mujoco.MjData(self.model)

        H, W = self.cfg.scene.image_size
        # Recreate renderer per reset since the model changed.
        self.renderer = mujoco.Renderer(self.model, height=H, width=W)

        self._cache_indices()

        # Seed the arm to the configured home pose. Without this the URDF's
        # zero pose has the arm pointing straight up — useless for cube manip
        # and blocks the camera. Apply pose, ctrl, and forward-propagate
        # kinematics before stepping physics.
        sc = self.cfg.scene
        for i, qi in enumerate(self._arm_qpos_idx):
            self.data.qpos[qi] = sc.home_qpos[i]
        if self._gripper_qpos_idx >= 0:
            self.data.qpos[self._gripper_qpos_idx] = sc.home_gripper
        for i, ctrl_i in enumerate(self._arm_ctrl_idx):
            self.data.ctrl[ctrl_i] = sc.home_qpos[i]
        if self._gripper_ctrl_idx >= 0:
            self.data.ctrl[self._gripper_ctrl_idx] = sc.home_gripper
        mujoco.mj_forward(self.model, self.data)

        # Settle physics so cubes rest on the table before first observation.
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Reset magnetic-grip state for the new episode.
        self._attached = False

        self._stats = EpisodeStats()
        return self._observe(), self._privileged()

    def step(self, action: np.ndarray) -> tuple[
        dict[str, Any], float, bool, bool, dict[str, Any]
    ]:
        """Returns (obs, reward, terminated, truncated, info).
        Reward is task-shaping only; the policy's training loss is elsewhere.
        """
        assert self.model is not None and self.data is not None

        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            chunk = action[None, :]
        else:
            chunk = action

        terminated = truncated = False
        info: dict[str, Any] = {}

        for a in chunk:
            self._apply_action(a)
            for _ in range(self.cfg.sim_substeps):
                mujoco.mj_step(self.model, self.data)
                # Re-apply magnetic grip every substep so the cube tracks the
                # ee smoothly even during a single env-step's worth of motion.
                self._update_grasp()
            self._update_episode_state()
            self._stats.steps += 1

            if self._terminal():
                terminated = True
                break
            if self._stats.steps >= self.cfg.max_episode_steps:
                truncated = True
                break

        obs = self._observe()
        info.update(self._privileged())
        info["stats"] = self._stats.as_dict()
        reward = float(self._stats.success) - 0.5 * float(self._stats.red_contact)
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Render the agentview (main) camera — the policy's primary image."""
        return self._render_camera(self.cfg.scene.camera_name)

    def render_wrist(self) -> np.ndarray:
        """Render the wrist camera — the policy's secondary image."""
        return self._render_camera(self.cfg.scene.wrist_camera_name)

    def _render_camera(self, camera: str) -> np.ndarray:
        assert self.renderer is not None and self.data is not None
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # ----- internals ------------------------------------------------------

    def _cache_indices(self) -> None:
        assert self.model is not None
        m = self.model
        sc = self.cfg.scene

        def jname(j: int) -> str:
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""

        def aname(a: int) -> str:
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""

        joint_name_to_qpos: dict[str, int] = {}
        for j in range(m.njnt):
            joint_name_to_qpos[jname(j)] = int(m.jnt_qposadr[j])

        self._arm_qpos_idx = []
        for nm in sc.arm_joint_names:
            if nm not in joint_name_to_qpos:
                raise KeyError(
                    f"Arm joint '{nm}' not found in MJCF. "
                    f"Update SceneConfig.arm_joint_names to match your file."
                )
            self._arm_qpos_idx.append(joint_name_to_qpos[nm])
        self._gripper_qpos_idx = joint_name_to_qpos.get(sc.gripper_joint_name, -1)

        # Actuator index per joint name (assume one actuator per arm joint).
        actuator_for_joint: dict[str, int] = {}
        for a in range(m.nu):
            jid = int(m.actuator_trnid[a, 0])
            if jid >= 0:
                actuator_for_joint[jname(jid)] = a
        self._arm_ctrl_idx = [actuator_for_joint[nm] for nm in sc.arm_joint_names
                              if nm in actuator_for_joint]
        self._gripper_ctrl_idx = actuator_for_joint.get(sc.gripper_joint_name, -1)

        # Bodies / geoms.
        self._red_body_ids = []
        self._red_geom_ids = set()
        assert self.layout is not None
        for red in self.layout.red_cubes:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, red.name)
            self._red_body_ids.append(bid)
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{red.name}_geom")
            if gid >= 0:
                self._red_geom_ids.add(gid)
        self._blue_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.layout.blue_cube.name)
        bg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{self.layout.blue_cube.name}_geom")
        self._blue_geom_ids = {bg} if bg >= 0 else set()
        # Look up the blue cube's freejoint qpos/qvel slots for the magnetic grip.
        blue_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "blue_free")
        if blue_jid >= 0:
            self._blue_qpos_start = int(m.jnt_qposadr[blue_jid])
            self._blue_dof_start = int(m.jnt_dofadr[blue_jid])

        # Arm geoms = every geom whose body is *not* a cube, the table, or the floor.
        cube_bodies = set(self._red_body_ids) | {self._blue_body_id}
        self._arm_geom_ids = set()
        gripper_body_names = {"gripper_link", "moving_jaw_so101_v1_link", "gripper_frame_link"}
        gripper_body_ids = set()
        for nm in gripper_body_names:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid >= 0:
                gripper_body_ids.add(bid)
        self._gripper_geom_ids = set()
        for g in range(m.ngeom):
            bid = int(m.geom_bodyid[g])
            if bid in cube_bodies:
                continue
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if gname.startswith("table") or gname == "floor":
                continue
            if int(m.body_parentid[bid]) > 0:
                self._arm_geom_ids.add(g)
                if bid in gripper_body_ids:
                    self._gripper_geom_ids.add(g)

        self._ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sc.ee_site_name)
        self._grasp_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sc.grasp_site_name)
        self._goal_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "goal")

    def _apply_action(self, a: np.ndarray) -> None:
        assert self.data is not None
        n_arm = len(self._arm_ctrl_idx)
        if a.shape[0] < n_arm:
            raise ValueError(f"action has {a.shape[0]} dims, expected at least {n_arm}")
        for i, ctrl_i in enumerate(self._arm_ctrl_idx):
            self.data.ctrl[ctrl_i] = a[i]
        if self._gripper_ctrl_idx >= 0 and a.shape[0] > n_arm:
            self.data.ctrl[self._gripper_ctrl_idx] = a[n_arm]

    def _observe(self) -> dict[str, Any]:
        assert self.data is not None
        qpos = np.array([self.data.qpos[i] for i in self._arm_qpos_idx], dtype=np.float32)
        grip = np.float32(self.data.qpos[self._gripper_qpos_idx]) if self._gripper_qpos_idx >= 0 else np.float32(0.0)
        state = np.concatenate([qpos, np.array([grip], dtype=np.float32)])
        return {"image": self.render(), "wrist_image": self.render_wrist(),
                "state": state}

    def _privileged(self) -> dict[str, Any]:
        assert self.data is not None and self.layout is not None
        cube_pos = np.stack([self.data.xpos[bid].copy() for bid in self._red_body_ids]).astype(np.float32)
        half = self.cfg.scene.red_cube_half
        cube_half = np.full_like(cube_pos, half, dtype=np.float32)
        blue_pos = self.data.xpos[self._blue_body_id].astype(np.float32).copy()
        ee_pos = self.data.site_xpos[self._ee_site_id].astype(np.float32).copy() if self._ee_site_id >= 0 else np.zeros(3, np.float32)
        goal_pos = self.data.site_xpos[self._goal_site_id].astype(np.float32).copy() if self._goal_site_id >= 0 else np.zeros(3, np.float32)
        return {
            "cube_positions": cube_pos,
            "cube_half_extents": cube_half,
            "blue_cube_pos": blue_pos,
            "goal_pos": goal_pos,
            "ee_pos": ee_pos,
            "grasped": bool(self._blue_grasped()),
        }

    def _blue_grasped(self) -> bool:
        """Magnetic-grip state. Set/cleared by _update_grasp()."""
        return self._attached

    def _update_grasp(self) -> None:
        """Magnetic-grip controller. Attaches the blue cube to the gripper
        when the jaws are *physically* closed (gripper qpos has reached the
        close threshold) AND the cube is within range of the jaw-gap center
        (grasp site). Releases when the jaws physically open. Triggering on
        actual qpos (not the ctrl signal) means the cube doesn't snap to the
        gripper before the jaws have moved.

        The SO-101 jaw spacing (~55 mm closed) is too wide to physically
        pinch a 25 mm cube, so we sidestep contact physics for the held
        state — the policy's image observation still sees the cube moving
        naturally with the gripper. The cube is locked to the grasp site (the
        point between the fingers) rather than the ee_site (which sits on the
        fixed jaw), so the held cube stays centered in the jaws.
        """
        assert self.data is not None
        if self._gripper_qpos_idx < 0 or self._blue_qpos_start < 0:
            return

        grip_qpos = float(self.data.qpos[self._gripper_qpos_idx])
        grasp = self.grasp_position()
        blue_pos = self.data.xpos[self._blue_body_id]

        if not self._attached:
            if grip_qpos < self._grip_qpos_attach_threshold:
                if np.linalg.norm(grasp - blue_pos) < self._grip_attach_radius:
                    self._attached = True
        else:
            if grip_qpos > self._grip_qpos_release_threshold:
                self._attached = False

        if self._attached:
            # Kinematically lock the cube to the jaw-gap center (between the
            # fingers), with a small vertical offset to seat the cube center.
            tgt = grasp + np.array([0.0, 0.0, self._grip_cube_offset_z])
            qi = self._blue_qpos_start
            self.data.qpos[qi:qi + 3] = tgt
            # Preserve orientation (identity quat); zero out velocity.
            self.data.qpos[qi + 3:qi + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            vi = self._blue_dof_start
            self.data.qvel[vi:vi + 6] = 0.0

    def _update_episode_state(self) -> None:
        assert self.data is not None and self.layout is not None
        m, d = self.model, self.data
        sc = self.cfg.scene

        # 1. Red-cube contact.
        for ci in range(d.ncon):
            c = d.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            hits_red = g1 in self._red_geom_ids or g2 in self._red_geom_ids
            if not hits_red:
                continue
            other = g2 if g1 in self._red_geom_ids else g1
            if other in self._arm_geom_ids or other in self._blue_geom_ids:
                self._stats.red_contact = True
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or "?"
                self._stats.contact_history.append(name)

        # 2. Ceiling violation — only enforced once the cube is grasped (carry phase).
        ee_pos = d.site_xpos[self._ee_site_id] if self._ee_site_id >= 0 else None
        if ee_pos is not None and self._blue_grasped() and ee_pos[2] > sc.ee_height_ceiling:
            self._stats.ceiling_violation = True

        # 3. Blue dropped (fell off table or below threshold).
        blue_z = d.xpos[self._blue_body_id][2]
        if blue_z < sc.table_z - 0.05:
            self._stats.blue_dropped = True

        # 4. Success: blue cube center inside the square goal patch (L-inf
        # against goal_size/2), resting on the table, and released.
        if self._goal_site_id >= 0:
            goal_xy = d.site_xpos[self._goal_site_id][:2]
            blue_xy = d.xpos[self._blue_body_id][:2]
            half = sc.goal_size / 2
            in_zone = (abs(goal_xy[0] - blue_xy[0]) < half
                       and abs(goal_xy[1] - blue_xy[1]) < half)
            resting = abs(blue_z - (sc.table_z + sc.blue_cube_half)) < 0.01
            released = not self._blue_grasped()
            if in_zone and resting and released:
                self._stats.dwell += 1
                if self._stats.dwell >= self.cfg.success_dwell_steps:
                    self._stats.success = True
            else:
                self._stats.dwell = 0

        # 5. Min clearance (signed distance ee → nearest red cube surface).
        if ee_pos is not None and len(self._red_body_ids) > 0:
            for bid in self._red_body_ids:
                center = d.xpos[bid]
                half = sc.red_cube_half
                d_box = np.maximum(np.abs(ee_pos - center) - half, 0.0)
                outside = np.linalg.norm(d_box)
                inside = min(max(np.max(np.abs(ee_pos - center) - half), -1.0), 0.0)
                clearance = outside + inside
                if clearance < self._stats.min_clearance:
                    self._stats.min_clearance = float(clearance)
                    self._stats.closest_red_idx = self._red_body_ids.index(bid)

    def _terminal(self) -> bool:
        if self._stats.success:
            return True
        if self.cfg.terminate_on_red_contact and self._stats.red_contact:
            return True
        if self.cfg.terminate_on_ceiling_violation and self._stats.ceiling_violation:
            return True
        if self._stats.blue_dropped:
            return True
        return False

    # ----- exposed helpers for the expert / safety loss -------------------

    @property
    def action_dim(self) -> int:
        return len(self._arm_ctrl_idx) + (1 if self._gripper_ctrl_idx >= 0 else 0)

    def joint_positions(self) -> np.ndarray:
        assert self.data is not None
        return np.array([self.data.qpos[i] for i in self._arm_qpos_idx], dtype=np.float64)

    def ee_position(self) -> np.ndarray:
        assert self.data is not None
        if self._ee_site_id < 0:
            return np.zeros(3)
        return self.data.site_xpos[self._ee_site_id].copy()

    def grasp_position(self) -> np.ndarray:
        """World position of the tool-center-point (jaw-gap center). This is
        where a grasped cube sits, so the expert targets it for the
        approach/descend/grasp phases. Falls back to the ee_site if the grasp
        site is absent."""
        assert self.data is not None
        if self._grasp_site_id < 0:
            return self.ee_position()
        return self.data.site_xpos[self._grasp_site_id].copy()

    def gripper_qpos(self) -> float:
        """Current gripper joint position. Low = closed, high = open."""
        assert self.data is not None
        if self._gripper_qpos_idx < 0:
            return 0.0
        return float(self.data.qpos[self._gripper_qpos_idx])

    def ee_jacobian(self) -> np.ndarray:
        """3 x n_arm position Jacobian of the ee site wrt arm joints."""
        assert self.data is not None and self.model is not None
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._ee_site_id)
        return jacp[:, self._arm_dof_cols()]

    def _arm_dof_cols(self) -> list[int]:
        return [int(self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        ]) for nm in self.cfg.scene.arm_joint_names]

    def ik_solve(self, ee_target: np.ndarray, *, damping: float = 0.2,
                  max_iters: int = 30, pos_tol: float = 0.003,
                  max_dq: float = 0.4, target_grasp_site: bool = False) -> np.ndarray:
        """Damped least-squares IK on a *copy* of MjData so we can iterate
        without disturbing live physics. Returns absolute joint targets.

        When ``target_grasp_site`` is set, solves so the jaw-gap center
        (grasp site) reaches ``ee_target`` instead of the ee_site — used by
        the grasp phases so the cube lands between the fingers."""
        assert self.model is not None and self.data is not None
        m = self.model
        site_id = (self._grasp_site_id if target_grasp_site and self._grasp_site_id >= 0
                   else self._ee_site_id)
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        cols = self._arm_dof_cols()
        arm_q_idx = self._arm_qpos_idx
        jacp = np.zeros((3, m.nv))
        damp_eye = (damping ** 2) * np.eye(3)
        q = np.array([d.qpos[i] for i in arm_q_idx], dtype=np.float64)

        for _ in range(max_iters):
            ee = d.site_xpos[site_id].copy()
            err = np.asarray(ee_target, dtype=np.float64) - ee
            if np.linalg.norm(err) < pos_tol:
                break
            mujoco.mj_jacSite(m, d, jacp, None, site_id)
            J = jacp[:, cols]
            dq = J.T @ np.linalg.solve(J @ J.T + damp_eye, err)
            n = np.linalg.norm(dq)
            if n > max_dq:
                dq = dq * (max_dq / n)
            q = q + dq
            for k, qi in enumerate(arm_q_idx):
                d.qpos[qi] = q[k]
            mujoco.mj_forward(m, d)
        return q
