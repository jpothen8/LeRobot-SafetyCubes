"""Round-trip tests for SafeCubeEnv.snapshot()/restore() — the core primitive
behind branch-and-relabel ("cleanup") DAgger (CLAUDE.md §4).

The high-value check: restoring a *mid-carry* snapshot must reproduce the held
blue cube's position, which depends on the magnetic-grip state (``_attached`` /
``_grip_offset`` / ``_min_grip_qpos``) being captured and written back — that
state lives OUTSIDE ``data.*``, so forgetting it desyncs the cube silently and
poisons every cleanup demo.

Needs a MuJoCo GL context (EGL on the box). Skipped if one can't be created:

    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/pytest \
        sim/tests/test_snapshot_restore.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.configs import EnvConfig, ExpertConfig, SceneConfig
from sim.env import SafeCubeEnv
from sim.expert import ScriptedExpert


def _carry_env():
    """Build an env and roll the scripted expert until the cube is grasped and
    being carried. Returns (env, expert) or skips if no GL / no grasp."""
    scene = SceneConfig(n_red_cubes=8)
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=500, seed=0,
                                terminate_on_red_contact=False))
    try:
        obs, info = env.reset(seed=0)
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not init MuJoCo env (need a GL context): {e}")

    expert = ScriptedExpert(env=env, cfg=ExpertConfig())
    expert.reset()
    # Step until grasped and carrying (off the table), or give up.
    for _ in range(300):
        action = expert.act(info)
        obs, _, terminated, truncated, info = env.step(action)
        if info["grasped"] and float(info["blue_cube_pos"][2]) > 0.02:
            return env, expert, obs, info
        if terminated or truncated:
            break
    env.close()
    pytest.skip("expert never reached a grasped carry state for seed 0")


def test_snapshot_restore_roundtrip_midcarry():
    env, expert, _obs, info = _carry_env()
    try:
        assert env._attached, "fixture should hand back a mid-carry (grasped) state"
        assert env.current_clearance() != float("inf")

        snap = env.snapshot()
        # The magnetic-grip bookkeeping (lives OUTSIDE data.*) must be captured.
        assert snap["attached"] is True
        assert "grip_offset" in snap and "min_grip_qpos" in snap

        # Canonical reference = a restore. (Live xpos can lag qpos by one grip
        # nudge because _update_grasp writes the held-cube qpos without a forward;
        # restore's mj_forward reconciles them, so a restore is the ground truth.)
        _, priv_ref = env.restore(snap)
        qpos_ref = env.data.qpos.copy()
        blue_ref = np.asarray(priv_ref["blue_cube_pos"], dtype=np.float64).copy()
        clear_ref = env.current_clearance()
        assert env._stats.steps == 0  # fresh stats after restore

        # Diverge HARD: command the gripper open so the cube is released and
        # _attached flips to False. If restore ignored the grip state, the held
        # cube would be lost — this is exactly the silent-corruption bug.
        grip_open = ExpertConfig().grip_open
        for _ in range(25):
            hold = np.concatenate([env.joint_positions(), [grip_open]])
            _obs, _, term, trunc, info = env.step(hold)
            if term or trunc:
                break
        assert not env._attached, "divergence should have released the cube"

        # Restore must bring back the grasp AND the held-cube position exactly.
        _, priv = env.restore(snap)
        assert env._attached is True
        assert priv["grasped"] is True
        np.testing.assert_allclose(env.data.qpos, qpos_ref, atol=0, rtol=0)
        np.testing.assert_allclose(priv["blue_cube_pos"], blue_ref, atol=1e-9)
        assert abs(env.current_clearance() - clear_ref) < 1e-9
    finally:
        env.close()


def test_restore_is_repeatable():
    """Two restores of the same snapshot yield identical observations — so a
    branch is deterministic given its anchor (independent of scout divergence)."""
    env, expert, _obs, info = _carry_env()
    try:
        snap = env.snapshot()
        obs_a, _ = env.restore(snap)
        # Perturb, then restore the same snapshot again.
        for _ in range(5):
            obs, _, term, trunc, info = env.step(expert.act(info))
            if term or trunc:
                break
        obs_b, _ = env.restore(snap)
        np.testing.assert_allclose(obs_a["state"], obs_b["state"], atol=0, rtol=0)
        np.testing.assert_array_equal(obs_a["image"], obs_b["image"])
    finally:
        env.close()
