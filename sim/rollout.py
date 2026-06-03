"""Shared expert-episode runner.

ONE code path computes the expert label at every visited state, so the
demonstration collector (``sim.scripts.collect_demos``) and the DAgger
relabeler (``sim.dagger.collect_dagger_round``) cannot drift apart: the
"safety"/DAgger expert does exactly what the demonstration expert does.

The only thing that differs between the two callers is *which* action is
**executed** in the env (pure expert for demos; an expert/policy mix for
DAgger). The action that is **recorded** is always the expert's, at the state
the agent actually visited — that is the DAgger relabel and, for pure-expert
collection, simply the demonstration.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .configs import ExpertConfig
from .env import SafeCubeEnv
from .expert import ScriptedExpert
from .recorder import EpisodeRecorder

# choose_executed(expert_action, obs, info) -> action actually stepped in the env.
ChooseExecuted = Callable[[np.ndarray, dict, dict], np.ndarray]


def run_expert_episode(
    *,
    env: SafeCubeEnv,
    expert: ScriptedExpert,
    rec: EpisodeRecorder,
    task: str,
    seed: int,
    grip_open: float,
    choose_executed: ChooseExecuted | None = None,
) -> dict:
    """Roll one episode, recording the expert action at every visited state.

    Args:
        choose_executed: picks the action to *execute*. ``None`` → execute the
            expert (demonstration collection). For DAgger, pass a callable that
            returns the expert action w.p. ``alpha`` else the policy action.

    Returns the episode ``stats`` dict. The caller decides save vs. discard
    (``rec.save_episode()`` / ``rec.discard_episode()``) so the same loop serves
    both the keep-everything and the successes-only policies.
    """
    obs, info = env.reset(seed=seed)
    expert.reset()
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

        executed = (
            expert_action if choose_executed is None
            else choose_executed(expert_action, obs, info)
        )
        # DAgger relabel / demonstration: always record the EXPERT action.
        rec.add(obs, expert_action, info)
        obs, _, terminated, truncated, info = env.step(executed)

    return info["stats"]


def default_grip_open() -> float:
    return ExpertConfig().grip_open
