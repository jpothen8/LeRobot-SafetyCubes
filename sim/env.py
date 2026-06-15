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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Render headlessly on the GPU via EGL. Without this, MuJoCo binds a
# display-backed GLX context to $DISPLAY (e.g. an AnyDesk virtual display like
# :1); when that remote session disconnects or sleeps the framebuffer goes
# black, which silently corrupted ~60-68% of earlier collected datasets. EGL is
# independent of any X display. setdefault lets an interactive viewer override.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402  (must follow the MUJOCO_GL default above)
import numpy as np

from .configs import EnvConfig, SceneConfig
from .scene import Layout, build_scene, sample_layout


@dataclass
class EpisodeStats:
    steps: int = 0
    red_contact: bool = False
    ceiling_violation: bool = False
    fly_over: bool = False
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
            "fly_over": bool(self.fly_over),
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
        # Offset (cube_pos - grasp_site, in the grasp-site frame) captured at the
        # instant of attach, so the catch introduces no jump.
        self._grip_offset: np.ndarray = np.zeros(3)
        # Minimum grip qpos seen while attached — used to prevent the lerp from
        # running backwards (dragging the cube back toward the open-jaw position
        # as the jaw opens). frac is computed from this rather than live qpos, so
        # it only ever increases and the cube stays at _grip_hold_offset until release.
        self._min_grip_qpos: float = float("inf")
        # Where the cube should sit once the jaws are fully closed, in the
        # grasp-site frame. The grasp site is the gap center for the *open*
        # jaws; as the jaws close the fixed jaw stays put while the moving jaw
        # swings in, so the true center-between-the-fingers drifts toward the
        # fixed jaw along the grasp-frame x (jaw-opening) axis. We lerp the cube
        # along that drift as the jaws close (grip qpos: attach→closed) so it
        # stays centered between the fingers instead of ending up outside the
        # closed jaws. Measured by tracking the live gap center (midpoint of the
        # fixed-jaw face and the FK'd moving-jaw face) from open→closed: it moves
        # +0.0255 m in x and only ~+0.003 m in z. CRITICAL: the z component is
        # ~0 — the center does NOT move up the finger axis. An earlier value of
        # z=-0.0458 dragged the cube ~4.5 cm up the gripper as the jaws closed
        # (the cube visibly "rode up" the gripper); that was wrong.
        self._grip_hold_offset: np.ndarray = np.array([0.0255, -0.0117, 0.003])
        # Tunables for the grip proxy. We attach as soon as the jaws *start*
        # closing while the grasp site is on the cube — early enough that the
        # closing jaw can't shove the (light, free) cube across the table before
        # it's caught. With grip_open ≈ 0.6, attach as qpos drops below 0.45;
        # release > attach threshold (immediately on jaw opening).
        self._grip_qpos_attach_threshold: float = 0.45
        self._grip_attach_radius: float = 0.04
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
        # Recreate renderer per reset since the model changed. Close the old one
        # FIRST: leaking a Renderer (and its GL/EGL context + framebuffer) every
        # reset exhausts the driver's offscreen resources after a few hundred
        # episodes, after which renders silently come back all-black. That leak
        # corrupted ~60-68% of the safe_cube_v3 / safe_cube_dagger datasets.
        if self.renderer is not None:
            self.renderer.close()
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

        # Yaw the (fixed-position) blue cube so its vertical faces meet the
        # gripper jaws square-on. The arm reaches the grasp pose with a fixed
        # jaw-opening axis (it depends only on the fixed cube position + home
        # pose); we yaw the cube to that axis so the faces are parallel to the
        # jaws as they descend. Together with the held-cube yaw tracking in
        # _update_grasp this keeps the cube fully inside the jaws (no corner
        # poking out in XY) through the whole carry.
        self._set_blue_spawn_pose()

        # Settle physics so cubes rest on the table before first observation.
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Reset magnetic-grip state for the new episode.
        self._attached = False
        self._grip_offset = np.zeros(3)
        self._min_grip_qpos = float("inf")

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

    def _blue_grasp_yaw(self) -> float:
        """Yaw (rad, about world +z) of the gripper's jaw-opening axis at the
        moment it descends on the blue cube.

        Solved by IK'ing the jaw-gap center (grasp site) onto the blue cube's
        resting position from the home pose, then reading the resulting
        grasp-site orientation. The jaw-opening axis is the grasp frame's local
        x (the axis along which the magnetic grip centers the cube — see
        _grip_hold_offset), so its world projection gives the yaw the cube's
        faces must match to sit square in the jaws."""
        assert self.data is not None and self.model is not None
        if self._blue_body_id < 0 or self._grasp_site_id < 0:
            return 0.0
        blue_pos = self.data.xpos[self._blue_body_id].copy()
        q = self.ik_solve(blue_pos, damping=0.15, max_iters=40,
                          pos_tol=0.003, target_grasp_site=True)
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        for k, qi in enumerate(self._arm_qpos_idx):
            d.qpos[qi] = q[k]
        mujoco.mj_forward(self.model, d)
        jaw_axis = d.site_xmat[self._grasp_site_id].reshape(3, 3)[:, 0]
        return float(np.arctan2(jaw_axis[1], jaw_axis[0]))

    def _set_blue_spawn_pose(self) -> None:
        """Yaw the blue cube (kept upright) so its faces are parallel to the
        gripper jaws on the approach. Position is already fixed by the layout;
        this only rewrites the cube freejoint's orientation quat."""
        assert self.data is not None
        if self._blue_qpos_start < 0:
            return
        yaw = self._blue_grasp_yaw()
        qi = self._blue_qpos_start
        self.data.qpos[qi + 3:qi + 7] = np.array(
            [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        mujoco.mj_forward(self.model, self.data)

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
        # Grasp-site orientation: hold offsets are expressed in this frame so
        # they track the gripper as it reorients during the carry.
        R = self.data.site_xmat[self._grasp_site_id].reshape(3, 3) \
            if self._grasp_site_id >= 0 else np.eye(3)
        blue_pos = self.data.xpos[self._blue_body_id]

        if not self._attached:
            if grip_qpos < self._grip_qpos_attach_threshold:
                if np.linalg.norm(grasp - blue_pos) < self._grip_attach_radius:
                    self._attached = True
                    self._grip_offset = R.T @ (np.asarray(blue_pos) - grasp)
                    self._min_grip_qpos = grip_qpos
        else:
            # Release as soon as the jaw opens past the attach threshold — no
            # need for a separate higher threshold now that the lerp never runs
            # backwards during opening.
            if grip_qpos > self._grip_qpos_attach_threshold:
                self._attached = False

        if self._attached:
            # Keep the cube centered between the fingers as the jaws close:
            # lerp from where it was caught toward the closed-jaw center as grip
            # qpos goes attach→home (frac 0→1). We use _min_grip_qpos (the
            # lowest qpos seen while attached) instead of the live qpos so frac
            # only ever increases — the cube moves toward _grip_hold_offset as
            # the jaws close but never drifts back toward the open-jaw position
            # as the jaws open.
            self._min_grip_qpos = min(self._min_grip_qpos, grip_qpos)
            span = self._grip_qpos_attach_threshold - self.cfg.scene.home_gripper
            frac = float(np.clip(
                (self._grip_qpos_attach_threshold - self._min_grip_qpos) / max(span, 1e-6),
                0.0, 1.0))
            local = self._grip_offset + frac * (self._grip_hold_offset - self._grip_offset)
            tgt = grasp + R @ local
            qi = self._blue_qpos_start
            self.data.qpos[qi:qi + 3] = tgt
            # Yaw-align the held cube to the gripper's jaw-opening axis (grasp
            # frame local x) so its faces stay parallel to the jaws and it sits
            # fully inside them — no corner poking out in XY — as the wrist
            # sweeps through the carry. Kept upright (yaw only): the cube tracks
            # the jaw yaw but never tilts. Velocity zeroed.
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            self.data.qpos[qi + 3:qi + 7] = np.array(
                [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
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
        # Checks the HELD CUBE's z position, not the TCP/ee, because the TCP sits
        # ~10 mm above the cube center and would false-fire at normal carry heights.
        # Threshold is the cube-center height at which the cube bottom would clear
        # a red cube top (fly-over): red_top + blue_half ≈ 0.025 + 0.013 = 0.038 m,
        # so ee_height_ceiling=0.035 gives ~3 mm margin before a true fly-over.
        ee_pos = d.site_xpos[self._ee_site_id] if self._ee_site_id >= 0 else None
        if self._blue_grasped():
            cube_z = d.xpos[self._blue_body_id][2]
            if cube_z > sc.ee_height_ceiling:
                self._stats.ceiling_violation = True

        # 2b. Fly-over: while carrying, the held cube is laterally over a red's
        # footprint AND its bottom clears the red's top — i.e. passing *over* an
        # obstacle instead of weaving between. Geometric, so it catches marginal
        # cases the ee-height ceiling can miss (the ee sits a few mm below the
        # held cube).
        if self._blue_grasped() and len(self._red_body_ids) > 0:
            bpos = d.xpos[self._blue_body_id]
            bh, rh = sc.blue_cube_half, sc.red_cube_half
            for bid in self._red_body_ids:
                rc = d.xpos[bid]
                if (abs(bpos[0] - rc[0]) < rh + bh
                        and abs(bpos[1] - rc[1]) < rh + bh
                        and (bpos[2] - bh) > (rc[2] + rh)):
                    self._stats.fly_over = True
                    break

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

    def blue_cube_position(self) -> np.ndarray:
        """World position of the blue cube's center. During carry this differs
        from grasp_position() by the grip-hold offset (see _grip_hold_offset).
        Falls back to grasp_position() if the body id is not set."""
        assert self.data is not None
        if self._blue_body_id < 0:
            return self.grasp_position()
        return self.data.xpos[self._blue_body_id].copy()

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

    def current_clearance(self) -> float:
        """Instantaneous signed clearance (box SDF) from the ee to the NEAREST
        red cube surface — the live, single-step version of the running
        ``stats['min_clearance']`` (which is a min over the whole episode).

        Used by the cleanup-DAgger gate to detect near-violations *while* the
        scout policy is rolling. Positive = clear (outside), negative =
        penetrating. Returns ``+inf`` if there is no ee site or no red cubes.
        Replicates the box-SDF loop in ``_update_episode_state`` step 5."""
        assert self.data is not None
        if self._ee_site_id < 0 or len(self._red_body_ids) == 0:
            return float("inf")
        ee_pos = self.data.site_xpos[self._ee_site_id]
        half = self.cfg.scene.red_cube_half
        best = float("inf")
        for bid in self._red_body_ids:
            center = self.data.xpos[bid]
            d_box = np.maximum(np.abs(ee_pos - center) - half, 0.0)
            outside = float(np.linalg.norm(d_box))
            inside = min(max(float(np.max(np.abs(ee_pos - center) - half)), -1.0), 0.0)
            clearance = outside + inside
            if clearance < best:
                best = clearance
        return best

    def snapshot(self) -> dict[str, Any]:
        """Full sim + magnetic-grip state, for branch-and-relabel DAgger.

        Copies the MuJoCo physics state (``qpos/qvel/act/ctrl/time``) **and** the
        magnetic-grip bookkeeping (``_attached`` / ``_grip_offset`` /
        ``_min_grip_qpos``), which lives *outside* ``data.*`` — omit it and the
        held cube desyncs from the gripper on :meth:`restore`. ``data.act`` may be
        size-0 (no stateful actuators); a size-0 ``copy()`` is harmless and
        :meth:`restore` guards the write. The snapshot is valid **only** for this
        env instance (same model / layout / renderer)."""
        assert self.data is not None
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "act": self.data.act.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
            "attached": bool(self._attached),
            "grip_offset": self._grip_offset.copy(),
            "min_grip_qpos": float(self._min_grip_qpos),
        }

    def restore(self, snap: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore a :meth:`snapshot` into THIS env; return ``(obs, privileged)``.

        Writes the physics + magnetic-grip state back, re-runs forward kinematics,
        and starts a FRESH :class:`EpisodeStats` so the branched episode's stats
        reflect only the branch. Does **not** rebuild the model / layout /
        renderer — the cleanup branch runs in the snapshot's exact world (a
        snapshot is only valid for the env instance that produced it)."""
        assert self.model is not None and self.data is not None
        self.data.qpos[:] = snap["qpos"]
        self.data.qvel[:] = snap["qvel"]
        if self.data.act.size:
            self.data.act[:] = snap["act"]
        self.data.ctrl[:] = snap["ctrl"]
        self.data.time = snap["time"]
        self._attached = bool(snap["attached"])
        self._grip_offset = np.asarray(snap["grip_offset"]).copy()
        self._min_grip_qpos = float(snap["min_grip_qpos"])
        mujoco.mj_forward(self.model, self.data)
        self._stats = EpisodeStats()
        return self._observe(), self._privileged()

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
