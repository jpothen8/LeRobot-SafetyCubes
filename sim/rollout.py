"""Shared expert-episode runner.

ONE code path computes the expert label at every visited state, so every
collector — the demonstration set (``sim.scripts.collect_demos``) and the
branch-and-relabel "cleanup" DAgger collector
(``sim.scripts.collect_dagger_cleanup``) — cannot drift apart: the cleanup
expert does exactly what the demonstration expert does.

Two knobs vary the episode:

* ``choose_executed`` — *which* action is **executed**. ``None`` → execute the
  expert (this is both a plain demonstration and a cleanup branch: every
  recorded frame is a coherent expert action). The deprecated α-mixing DAgger
  (``sim.dagger.collect_dagger_round``) passed an expert/policy mix here.
* ``restore_state`` — *where the episode starts*. ``None`` → ``env.reset(seed)``.
  A ``snapshot()`` dict → ``env.restore(...)`` so the pure expert runs to
  completion from an on-policy hard state (cleanup DAgger's branch step).

The action that is **recorded** is always the expert's, at the state actually
visited — a demonstration for pure-expert collection, a relabel otherwise.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .configs import ExpertConfig
from .env import SafeCubeEnv
from .expert import ScriptedExpert
from .recorder import EpisodeRecorder

# choose_executed(expert_action, obs, info) -> (action, actor_label)
# actor_label: 0.0 = expert, 1.0 = policy
ChooseExecuted = Callable[[np.ndarray, dict, dict], tuple[np.ndarray, float]]


def run_expert_episode(
    *,
    env: SafeCubeEnv,
    expert: ScriptedExpert,
    rec: EpisodeRecorder,
    task: str,
    seed: int,
    grip_open: float,
    choose_executed: ChooseExecuted | None = None,
    restore_state: dict | None = None,
) -> dict:
    """Roll one episode, recording the expert action at every visited state.

    Args:
        choose_executed: picks the action to *execute*. ``None`` → execute the
            expert (demonstration collection). For DAgger, pass a callable that
            returns the expert action w.p. ``alpha`` else the policy action.
        restore_state: a ``SafeCubeEnv.snapshot()`` dict. When given, the episode
            starts from ``env.restore(restore_state)`` instead of
            ``env.reset(seed)`` (``seed`` is ignored) — the branch-and-relabel
            ("cleanup") DAgger entry point: run the pure scripted expert to
            completion from an on-policy hard state. The rest of the loop is
            unchanged, so a branched episode is recorded exactly like a demo.

    Returns the episode ``stats`` dict. The caller decides save vs. discard
    (``rec.save_episode()`` / ``rec.discard_episode()``) so the same loop serves
    both the keep-everything and the successes-only policies.
    """
    if restore_state is not None:
        obs, info = env.restore(restore_state)
    else:
        obs, info = env.reset(seed=seed)
    expert.reset()
    if restore_state is not None:
        # Branch (cleanup DAgger): re-root the carry corridor by BFS from the
        # RESTORED cube position so the expert weaves *forward* from the on-policy
        # state — never back toward the original spawn→goal waypoints behind it.
        expert.replan_carry_from_current(info["cube_positions"])
    rec.begin_episode(task=task)

    terminated = truncated = False
    # After the expert finishes, hold pose (gripper OPEN) for a short settle
    # window so the success dwell can register and the magnetic grip does not
    # re-grab the released cube. Identical budget in demos and DAgger.
    settle_budget = max(int(env.cfg.fps), env.cfg.success_dwell_steps + 5)
    settle_left = settle_budget
    while not (terminated or truncated):
        if expert.done():
            if settle_left <= 0:
                break
            settle_left -= 1
            expert_action = np.concatenate([env.joint_positions(), [grip_open]])
        else:
            expert_action = expert.act(info)

        if choose_executed is None:
            executed, actor = expert_action, 0.0
        else:
            executed, actor = choose_executed(expert_action, obs, info)
        # DAgger relabel / demonstration: always record the EXPERT action.
        rec.add(obs, expert_action, info, actor=actor)
        obs, _, terminated, truncated, info = env.step(executed)

    return info["stats"]


def default_grip_open() -> float:
    return ExpertConfig().grip_open
