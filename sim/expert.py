"""Scripted expert that consumes privileged state and produces joint commands.

Strategy
--------
A simple finite-state machine over Cartesian waypoints:

    APPROACH  -> ee above the blue cube (pre-grasp height)
    DESCEND   -> ee at the blue cube (grasp height)
    CLOSE     -> hold pose, close gripper for K steps
    LIFT      -> raise to carry height (below ee_height_ceiling)
    CARRY     -> repeatedly nudge XY toward the goal, with potential-field
                 repulsion from red cubes; z held at carry height
    DESCEND2  -> over goal, lower to release height
    OPEN      -> hold pose, open gripper
    DONE

Each tick:
  1. choose a Cartesian target from the current state,
  2. run damped least-squares IK against the env's current ee pose + Jacobian,
  3. emit (joint_targets, gripper_cmd) as the action.

Because the env exposes the live Jacobian and current joint positions, the
expert never needs to know the URDF internals — it just talks Jacobians and
joint deltas. This means the same expert works for any SO-style arm whose
joint names you wire into SceneConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .configs import ExpertConfig
from .env import SafeCubeEnv
from .scene import plan_carry_path


class _Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    CARRY = auto()
    DESCEND2 = auto()
    OPEN = auto()
    DONE = auto()


@dataclass
class ScriptedExpert:
    env: SafeCubeEnv
    cfg: ExpertConfig

    def __post_init__(self) -> None:
        self._phase = _Phase.APPROACH
        self._phase_step = 0
        self._grip = self.cfg.grip_open
        self._wp_idx = 0
        self._jaws_closed_at: int = -1   # _phase_step when CLOSE phase saw closure

    def reset(self) -> None:
        self._phase = _Phase.APPROACH
        self._phase_step = 0
        self._grip = self.cfg.grip_open
        self._wp_idx = 0
        self._jaws_closed_at = -1

    def sync_wp_to_cube(self) -> None:
        """Advance _wp_idx past waypoints the cube has already moved beyond.

        Call this when the expert resumes CARRY after a policy chunk. While the
        policy controlled the arm, _wp_idx was frozen; if the policy carried the
        cube forward past several BFS waypoints the expert will otherwise target
        stale (behind-the-cube) waypoints and drive the arm backward.

        The heuristic: a waypoint is "passed" if it is farther from the goal than
        the cube currently is (plus a small tolerance to avoid skipping ahead of
        the cube on noise). We only advance, never retreat.
        """
        if self._phase is not _Phase.CARRY:
            return
        wps = self.env.layout.carry_waypoints if self.env.layout else []
        if not wps:
            return
        goal_xy = self.env.layout.goal_xy
        cube_xy = self.env.blue_cube_position()[:2]
        dist_cube_goal = float(np.linalg.norm(cube_xy - goal_xy))
        while (self._wp_idx < len(wps) - 1 and
               np.linalg.norm(wps[self._wp_idx] - goal_xy) > dist_cube_goal + 0.02):
            self._wp_idx += 1

    def replan_carry_from_current(self, reds: np.ndarray) -> None:
        """Plan a FRESH carry corridor by BFS from the cube's CURRENT position to
        the goal and reset ``_wp_idx`` to 0 on it.

        Called once at the start of a branch-and-relabel ("cleanup") episode
        (``run_expert_episode(restore_state=...)``): the cube has been restored to
        an on-policy mid-carry state, so the original spawn→goal ``carry_waypoints``
        sit *behind* it — tracking them from ``_wp_idx=0`` would drive the arm
        backward toward the spawn. Re-rooting the corridor at the cube guarantees
        the expert weaves *forward* from wherever the policy left it, regardless of
        whether the cube happens to be near or far from the old corridor (so it
        does not depend on ``_maybe_replan_carry``'s off-path threshold firing).
        No-op if no clearance-respecting path exists from here (keep the old path;
        such a branch usually fails the task and is discarded anyway)."""
        if self.env.layout is None:
            return
        cube_xy = self.env.blue_cube_position()[:2]
        new_path = plan_carry_path(
            self.env.cfg.scene, np.asarray(reds, dtype=np.float64),
            cube_xy, self.env.layout.goal_xy,
        )
        if new_path:
            self.env.layout.carry_waypoints = new_path
            self._wp_idx = 0

    def _maybe_replan_carry(self, cube_xy: np.ndarray, reds: np.ndarray) -> None:
        """Re-route the carry corridor if the held cube has drifted off it.

        ``sync_wp_to_cube`` only fixes *along-path* progress (the waypoint
        index); it cannot help when the policy has pushed the cube *laterally*
        off the corridor, where the original spawn→goal waypoints no longer
        describe a safe route from the cube's position (following them can drive
        the arm back across the field or straight through a red).

        When the cube ("the arm's state", measured at the cube center the
        corridor is cleared for) is farther than ``replan_offpath_threshold``
        from the CLOSEST current waypoint, re-run the BFS planner from the cube's
        CURRENT XY to the goal and adopt that fresh corridor (``_wp_idx`` reset
        to 0). Uses the live red positions so the route respects any cubes the
        rollout has nudged. No-op when disabled, before any path exists, when the
        cube is still on-corridor, or when no clearance-respecting path exists
        from here (keep the old path rather than steer through an obstacle).

        Fires only in DAgger: the pure expert tracks waypoints within
        ``waypoint_reach_tol`` (≪ threshold), so the closest waypoint is always
        near and this never triggers during demonstration collection.
        """
        thr = self.cfg.replan_offpath_threshold
        if thr <= 0 or self.env.layout is None:
            return
        wps = self.env.layout.carry_waypoints
        if not wps:
            return
        closest = float(np.linalg.norm(np.asarray(wps) - cube_xy, axis=1).min())
        if closest <= thr:
            return
        new_path = plan_carry_path(
            self.env.cfg.scene, reds, cube_xy, self.env.layout.goal_xy,
        )
        if new_path:
            self.env.layout.carry_waypoints = new_path
            self._wp_idx = 0

    # ----- public --------------------------------------------------------

    def act(self, privileged: dict) -> np.ndarray:
        """Return a single action vector (joint targets + gripper)."""
        blue = np.asarray(privileged["blue_cube_pos"], dtype=np.float64)
        goal = np.asarray(privileged["goal_pos"], dtype=np.float64)
        reds = np.asarray(privileged["cube_positions"], dtype=np.float64)

        # Grasp-state resync. If the cube is already in hand but the FSM is still
        # in a pre-grasp phase, the policy grasped it (DAgger mixing) before the
        # expert's own phase logic advanced. Left alone, APPROACH keeps steering
        # toward a point ABOVE the cube (pre_grasp_height), lifting the *held*
        # cube over the ceiling. Snap to LIFT so we instead carry it DOWN to
        # carry_z. In normal expert-driven rollouts `grasped` only turns True
        # during CLOSE, so this never fires there -> no change to demo behavior.
        if self._phase in (_Phase.APPROACH, _Phase.DESCEND) and self.env._blue_grasped():
            self._phase = _Phase.LIFT
            self._phase_step = 0
            self._wp_idx = 0
            self._jaws_closed_at = -1
        # ref is the grasp site (jaw-gap center). Pre-grasp phases target it
        # directly; CARRY/DESCEND2 back-correct the IK target so the actual
        # cube center (not the TCP) follows the cleared corridor.
        ref = self.env.grasp_position()

        target, self._grip, advance = self._target_for_phase(ref, blue, goal, reds)
        joint_cmd = self._ik_step(target, target_grasp_site=True)
        self._phase_step += 1
        if advance:
            self._advance_phase()
        return np.concatenate([joint_cmd, [self._grip]])

    def done(self) -> bool:
        return self._phase == _Phase.DONE

    # ----- phase logic ---------------------------------------------------

    def _target_for_phase(
        self,
        ref: np.ndarray, blue: np.ndarray, goal: np.ndarray, reds: np.ndarray,
    ) -> tuple[np.ndarray, float, bool]:
        """Return (target, gripper_cmd, advance).

        `target` is the absolute Cartesian position the IK should put the
        grasp site at. Pre-grasp phases pass waypoints directly; CARRY and
        DESCEND2 subtract the cube→grasp offset so the cube center tracks
        the goal, not the TCP. `advance` triggers on actual progress.
        """
        c = self.cfg
        reach_tol = 0.02   # 2 cm tolerance for most phases

        if self._phase is _Phase.APPROACH:
            tgt = np.array([blue[0], blue[1], blue[2] + c.pre_grasp_height])
            return tgt, c.grip_open, np.linalg.norm(ref - tgt) < reach_tol

        if self._phase is _Phase.DESCEND:
            tgt = np.array([blue[0], blue[1], c.descend_grasp_z])
            return tgt, c.grip_open, np.linalg.norm(ref - tgt) < c.descend_reach_tol

        if self._phase is _Phase.CLOSE:
            # Hold pose, command the gripper closed, and wait until the jaws
            # have *physically* closed (qpos < threshold) plus a configurable
            # post-grasp hold (so the grip settles before the lift starts).
            grip_qpos = self.env.gripper_qpos()
            jaws_closed = grip_qpos < c.grasp_close_qpos_threshold
            if jaws_closed and self._jaws_closed_at < 0:
                self._jaws_closed_at = self._phase_step
            held_long_enough = (
                self._jaws_closed_at >= 0
                and self._phase_step - self._jaws_closed_at >= c.post_grasp_hold_steps
            )
            advance = (self._phase_step >= c.grasp_settle_steps and held_long_enough)
            return ref, c.grip_closed, advance

        if self._phase is _Phase.LIFT:
            # Lift in place (hold XY, raise the cube to the absolute carry height).
            tgt = np.array([ref[0], ref[1], c.carry_z])
            return tgt, c.grip_closed, abs(ref[2] - tgt[2]) < reach_tol

        if self._phase is _Phase.CARRY:
            # Drive the held cube along the BFS waypoints. The BFS corridor is
            # cleared for the CUBE CENTER, so we track and target the cube itself
            # (not the grasp site). The cube center sits at grasp_site + offset
            # where offset = _grip_hold_offset rotated into world frame; correct
            # the IK target so the cube lands on the waypoint, not the grasp site.
            cube_pos = self.env.blue_cube_position()
            cube_xy = cube_pos[:2]
            # If the policy (DAgger) has dragged the cube off the planned
            # corridor, re-route from where it actually is BEFORE picking a
            # waypoint, so the relabel routes safely from the current state
            # instead of steering back toward the stale spawn→goal corridor.
            self._maybe_replan_carry(cube_xy, reds)
            wps = self.env.layout.carry_waypoints if self.env.layout else []
            if self._wp_idx >= len(wps):
                tgt = np.array([goal[0], goal[1], c.carry_z])
                return tgt, c.grip_closed, True
            wp = wps[self._wp_idx]
            if np.linalg.norm(cube_xy - wp) < c.waypoint_reach_tol:
                self._wp_idx = min(self._wp_idx + 1, len(wps) - 1)
                wp = wps[self._wp_idx]
            # Shift the grasp-site target by the current cube→grasp offset so
            # the cube center tracks the waypoint rather than the TCP.
            offset = cube_pos - ref  # world-frame: grasp_site → cube_center
            tgt = np.array([wp[0] - offset[0], wp[1] - offset[1], c.carry_z - offset[2]])
            # CARRY advances once the cube is at the final waypoint (= goal_xy).
            done = (self._wp_idx == len(wps) - 1
                    and np.linalg.norm(cube_xy - goal[:2]) < 0.03)
            return tgt, c.grip_closed, done

        if self._phase is _Phase.DESCEND2:
            # Hover the cube over the goal CENTER. Correct for the cube→grasp offset
            # so the cube center (not the TCP) lands on the goal.
            cube_pos = self.env.blue_cube_position()
            offset = cube_pos - ref
            tgt = np.array([goal[0] - offset[0], goal[1] - offset[1], c.pre_release_z - offset[2]])
            done = np.linalg.norm(cube_pos - np.array([goal[0], goal[1], c.pre_release_z])) < c.descend2_reach_tol
            return tgt, c.grip_closed, done

        if self._phase is _Phase.OPEN:
            # Wait for the jaws to *physically* open past the release
            # threshold (mirrors the close logic) and keep the cube released.
            grip_qpos = self.env.gripper_qpos()
            advance = (self._phase_step >= c.release_settle_steps
                       and grip_qpos > c.grasp_open_qpos_threshold)
            return ref, c.grip_open, advance

        return ref, self._grip, False

    def _advance_phase(self) -> None:
        order = [
            _Phase.APPROACH, _Phase.DESCEND, _Phase.CLOSE, _Phase.LIFT,
            _Phase.CARRY, _Phase.DESCEND2, _Phase.OPEN, _Phase.DONE,
        ]
        i = order.index(self._phase)
        self._phase = order[min(i + 1, len(order) - 1)]
        self._phase_step = 0

    # ----- IK ------------------------------------------------------------

    @staticmethod
    def _step_toward(p: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        d = target - p
        n = np.linalg.norm(d)
        if n < 1e-9:
            return target
        return p + d * min(1.0, max_step / n)

    def _ik_step(self, target: np.ndarray, *, target_grasp_site: bool = True) -> np.ndarray:
        """Multi-iteration damped LS IK delegated to the env (which evaluates
        FK on a data copy). Solves so the grasp site (jaw-gap center) reaches
        the target."""
        return self.env.ik_solve(
            target,
            damping=max(self.cfg.ik_damping, 0.15),
            max_iters=self.cfg.ik_max_iters,
            pos_tol=self.cfg.ik_pos_tol,
            target_grasp_site=target_grasp_site,
        )
