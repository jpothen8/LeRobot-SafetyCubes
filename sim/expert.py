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

    # ----- public --------------------------------------------------------

    def act(self, privileged: dict) -> np.ndarray:
        """Return a single action vector (joint targets + gripper)."""
        blue = np.asarray(privileged["blue_cube_pos"], dtype=np.float64)
        goal = np.asarray(privileged["goal_pos"], dtype=np.float64)
        reds = np.asarray(privileged["cube_positions"], dtype=np.float64)
        ee = self.env.ee_position()

        target, self._grip, advance = self._target_for_phase(ee, blue, goal, reds)
        joint_cmd = self._ik_step(target)
        self._phase_step += 1
        if advance:
            self._advance_phase()
        return np.concatenate([joint_cmd, [self._grip]])

    def done(self) -> bool:
        return self._phase == _Phase.DONE

    # ----- phase logic ---------------------------------------------------

    def _target_for_phase(
        self,
        ee: np.ndarray, blue: np.ndarray, goal: np.ndarray, reds: np.ndarray,
    ) -> tuple[np.ndarray, float, bool]:
        """Return (ee_target, gripper_cmd, advance).

        Targets are *absolute* Cartesian goals for the current phase — the IK
        solves for the joints that achieve them and the position controller
        drives there over many env steps. `advance` triggers on actual progress
        of the live ee (not on the moving-waypoint trick).
        """
        c = self.cfg
        reach_tol = 0.02   # 2 cm tolerance for most phases

        if self._phase is _Phase.APPROACH:
            tgt = np.array([blue[0], blue[1], blue[2] + c.pre_grasp_height])
            return tgt, c.grip_open, np.linalg.norm(ee - tgt) < reach_tol

        if self._phase is _Phase.DESCEND:
            # Target a small margin ABOVE the cube — the user wants the
            # gripper at the cube edge with a little tolerance, not driving
            # a jaw into the cube center. Combined with a tight reach_tol
            # so the controller actually settles at this offset.
            tgt = np.array([blue[0], blue[1], blue[2] + c.descend_clearance])
            return tgt, c.grip_open, np.linalg.norm(ee - tgt) < c.descend_reach_tol

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
            return ee, c.grip_closed, advance

        if self._phase is _Phase.LIFT:
            # Lift in place (hold XY, raise to the absolute carry height).
            tgt = np.array([ee[0], ee[1], c.carry_z])
            return tgt, c.grip_closed, abs(ee[2] - tgt[2]) < reach_tol

        if self._phase is _Phase.CARRY:
            # Follow the BFS waypoints precomputed by the layout planner.
            # Advance to the next waypoint when within `waypoint_reach_tol`.
            wps = self.env.layout.carry_waypoints if self.env.layout else []
            if self._wp_idx >= len(wps):
                # Defensive fallback (shouldn't normally happen).
                tgt = np.array([goal[0], goal[1], c.carry_z])
                return tgt, c.grip_closed, True
            wp = wps[self._wp_idx]
            if np.linalg.norm(ee[:2] - wp) < c.waypoint_reach_tol:
                self._wp_idx = min(self._wp_idx + 1, len(wps) - 1)
                wp = wps[self._wp_idx]
            tgt = np.array([wp[0], wp[1], c.carry_z])
            # CARRY advances once we're at the *final* waypoint (= goal_xy).
            done = (self._wp_idx == len(wps) - 1
                    and np.linalg.norm(ee[:2] - goal[:2]) < 0.03)
            return tgt, c.grip_closed, done

        if self._phase is _Phase.DESCEND2:
            # Hover over the goal CENTER and only release when actually
            # centered (tight XY tolerance). The cube is glued under the ee
            # via the magnetic grip, so wherever the ee is at the moment of
            # release determines where the cube lands.
            tgt = np.array([goal[0], goal[1], c.pre_release_z])
            return tgt, c.grip_closed, np.linalg.norm(ee - tgt) < c.descend2_reach_tol

        if self._phase is _Phase.OPEN:
            # Wait for the jaws to *physically* open past the release
            # threshold (mirrors the close logic) and keep the cube released.
            grip_qpos = self.env.gripper_qpos()
            advance = (self._phase_step >= c.release_settle_steps
                       and grip_qpos > c.grasp_open_qpos_threshold)
            return ee, c.grip_open, advance

        return ee, self._grip, False

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

    def _ik_step(self, ee_target: np.ndarray) -> np.ndarray:
        """Multi-iteration damped LS IK delegated to the env (which evaluates
        FK on a data copy)."""
        return self.env.ik_solve(
            ee_target,
            damping=max(self.cfg.ik_damping, 0.15),
            max_iters=self.cfg.ik_max_iters,
            pos_tol=self.cfg.ik_pos_tol,
        )
