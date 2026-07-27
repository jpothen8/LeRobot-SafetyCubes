"""Train entrypoint for the constraint-aware Diffusion Policy.

Thin shim around LeRobot's stock training loop, exactly like
``sim/scripts/train_safe_pi0.py``. Importing ``sim.safe_diffusion_policy``
registers the ``safe_diffusion`` config with draccus, after which
``--policy.type=safe_diffusion`` resolves through the policy factory's dynamic
fallback and all the standard ``--policy.* / --dataset.* / --batch_size /
--steps`` args apply.

Requires the ``diffusers`` package (LeRobot's ``diffusion`` extra)::

    uv pip install --python .venv/bin/python "diffusers>=0.27.2,<0.36.0"

Usage (BC warm start — trains from scratch, there is no pretrained base)::

    tmux new -s diff_bc      # ALWAYS in tmux — see CLAUDE.md §2
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD .venv/bin/python \\
        -m sim.scripts.train_safe_diffusion \\
        --policy.type=safe_diffusion --policy.push_to_hub=false \\
        --policy.safety_weight=1.0 --policy.obstacle_weight=2.0 --policy.ceiling_weight=4.0 \\
        --policy.sdf_margin=0.02 \\
        --dataset.repo_id=local/safe-cube-mixed --dataset.root=data/safe_cube_v7.1 \\
        --output_dir=outputs/safe_diffusion_bc --batch_size=64 --steps=100000 --save_freq=10000 \\
        --wandb.enable=true --wandb.disable_artifact=true --wandb.project=safe-cube \\
        2>&1 | tee outputs/diffusion_bc_train.log

Differences from the π0 recipe that will bite you if you copy it verbatim:

* **No ``--policy.pretrained_path``.** Diffusion Policy trains from scratch
  (only the ResNet18 backbone is ImageNet-initialized, via
  ``--policy.pretrained_backbone_weights``). There is no ``lerobot/pi0_base``
  analogue, so expect to need *more* steps — but each is ~50x cheaper.
* **Steps, not hours.** π0's 15-20 k steps at ~5 s/step is ~21 h; this is a
  ~50 M-param model, so budget many more steps at a small fraction of the
  wall-clock. Early-stop on rollout **success**, not on loss (the safety term is
  non-stationary — same caveat as §2 of CLAUDE.md).
* **Checkpoints are small** (~0.6 GB vs ~22 GB), so a tight ``--save_freq`` is
  cheap here.

Everything downstream is policy-agnostic and works unchanged:
``sim.dagger.PolicyRollout``, ``sim/scripts/benchmark_policy.py``,
``sim/scripts/record_policy_rollout.py``, and the cleanup-DAgger collector.
"""

from __future__ import annotations

import sim.safe_diffusion_policy  # noqa: F401  -- side effect: registers `safe_diffusion`
from lerobot.scripts.lerobot_train import train

if __name__ == "__main__":
    train()
