"""Train entrypoint for the constraint-aware π0 policy.

This is a thin shim around LeRobot's stock training loop. Importing
``sim.safe_pi0_policy`` registers the ``safe_pi0`` config with draccus, after
which the standard ``--policy.type=safe_pi0`` resolves through the policy
factory's dynamic fallback. Everything else (optimizer, dataloader, scheduler,
checkpointing, fine-tuning from ``lerobot/pi0_base``) is stock.

Usage (BC warm start from the pretrained π0 base):

    uv run python -m sim.scripts.train_safe_pi0 \
        --policy.type=safe_pi0 \
        --policy.pretrained_path=lerobot/pi0_base \
        --policy.safety_weight=1.0 \
        --policy.repo_id=local/safe-pi0 \
        --dataset.repo_id=local/safe-cube-mixed \
        --dataset.root=data/safe_cube_mixed \
        --batch_size=8 \
        --steps=20000 \
        --policy.gradient_checkpointing=true

Tip: the safety term makes the loss landscape non-stationary, so early-stop at
peak rollout success rather than at min loss (``project_summary.md`` §7.2).
"""

from __future__ import annotations

import sim.safe_pi0_policy  # noqa: F401  -- side effect: registers `safe_pi0`
from lerobot.scripts.lerobot_train import train

if __name__ == "__main__":
    train()
