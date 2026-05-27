"""Eval harness — runs N episodes of a policy (or the scripted expert) and
returns aggregate metrics: success rate, red-contact rate, min clearance,
ceiling-violation rate.

The policy interface is duck-typed:

    action = policy(obs)        # obs is the dict returned by env.reset()/step()
    action: np.ndarray of shape (env.action_dim,) OR a chunk (T, env.action_dim)

Pass `policy=None` to evaluate the scripted expert instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .configs import EnvConfig, ExpertConfig
from .env import SafeCubeEnv
from .expert import ScriptedExpert


@dataclass
class EvalResult:
    n_episodes: int
    successes: int
    red_contacts: int
    ceiling_violations: int
    blue_drops: int
    timeouts: int
    mean_steps: float
    mean_min_clearance: float
    per_episode: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.n_episodes, 1)

    @property
    def red_contact_rate(self) -> float:
        return self.red_contacts / max(self.n_episodes, 1)

    def summary(self) -> str:
        return (
            f"episodes={self.n_episodes}  success={self.success_rate:.1%}  "
            f"red_contact={self.red_contact_rate:.1%}  "
            f"ceiling_viol={self.ceiling_violations}/{self.n_episodes}  "
            f"blue_drop={self.blue_drops}/{self.n_episodes}  "
            f"timeout={self.timeouts}/{self.n_episodes}  "
            f"mean_steps={self.mean_steps:.1f}  "
            f"mean_min_clearance={self.mean_min_clearance:.3f}m"
        )


PolicyFn = Callable[[dict], np.ndarray]


def evaluate(
    *,
    env: SafeCubeEnv,
    policy: PolicyFn | None,
    n_episodes: int,
    expert_cfg: ExpertConfig | None = None,
    base_seed: int = 0,
) -> EvalResult:
    expert = ScriptedExpert(env=env, cfg=expert_cfg or ExpertConfig()) if policy is None else None
    rows: list[dict] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        if expert is not None:
            expert.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            if expert is not None:
                if expert.done():
                    break
                action = expert.act(info)
            else:
                action = policy(obs)
            obs, _, terminated, truncated, info = env.step(action)
        rows.append(info["stats"])

    successes = sum(r["success"] for r in rows)
    contacts = sum(r["red_contact"] for r in rows)
    ceilings = sum(r["ceiling_violation"] for r in rows)
    drops = sum(r["blue_dropped"] for r in rows)
    timeouts = sum(
        1 for r in rows
        if not r["success"] and not r["red_contact"]
        and not r["ceiling_violation"] and not r["blue_dropped"]
    )
    mean_steps = float(np.mean([r["steps"] for r in rows])) if rows else 0.0
    mean_clear = float(np.mean([r["min_clearance"] for r in rows])) if rows else 0.0
    return EvalResult(
        n_episodes=n_episodes,
        successes=successes,
        red_contacts=contacts,
        ceiling_violations=ceilings,
        blue_drops=drops,
        timeouts=timeouts,
        mean_steps=mean_steps,
        mean_min_clearance=mean_clear,
        per_episode=rows,
    )


def make_default_env(env_cfg: EnvConfig | None = None) -> SafeCubeEnv:
    return SafeCubeEnv(env_cfg)
