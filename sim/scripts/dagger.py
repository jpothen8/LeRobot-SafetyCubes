"""DAgger orchestrator: collect → retrain → repeat for the safe-cube task.

Each round rolls episodes with smooth expert→policy mixing
(``alpha = alpha0 ** round``), records expert-labeled frames into one growing
dataset, then fine-tunes the policy by shelling out to the *stock* training
entrypoint (``sim.scripts.train_safe_pi0``). Round 0 is pure expert (a BC warm
start); later rounds load the previous round's checkpoint and mix in policy
actions, which is what drags the training distribution toward deployment states
and naturally produces the failure data the safety term needs.

Example (3 rounds, fine-tuning from the π0 base):

    uv run python -m sim.scripts.dagger \
        --repo-id local/safe-cube-dagger \
        --root data/safe_cube_dagger \
        --output-root outputs/safe_pi0_dagger \
        --rounds 3 --episodes-per-round 40 \
        --base-policy lerobot/pi0_base \
        --safety-weight 1.0 --steps 8000 --batch-size 8

Use ``--no-train`` to only collect (e.g. to build a static mixed set first), or
``--print-only`` to see the constructed training commands without running them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from sim.configs import EnvConfig, ExpertConfig, SceneConfig
from sim.dagger import PolicyRollout, collect_dagger_round
from sim.env import SafeCubeEnv
from sim.expert import ScriptedExpert
from sim.recorder import EpisodeRecorder
from sim.scripts.collect_demos import TASK_DESCRIPTION


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True, help="dataset repo id (grows across rounds)")
    p.add_argument("--root", type=Path, default=None, help="dataset root dir")
    p.add_argument("--output-root", type=Path, required=True, help="per-round train output dirs")
    p.add_argument("--mjcf", type=str, default=SceneConfig().mjcf_path)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes-per-round", type=int, default=40)
    p.add_argument("--alpha0", type=float, default=0.5, help="round alpha = alpha0 ** round")
    p.add_argument("--n-red-cubes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--device", type=str, default=None)
    # Training pass settings (forwarded to the stock entrypoint).
    p.add_argument("--base-policy", type=str, default="lerobot/pi0_base",
                   help="pretrained_path for round 0 (later rounds resume from the prior checkpoint)")
    p.add_argument("--safety-weight", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-train", action="store_true", help="collect only; skip retraining")
    p.add_argument("--print-only", action="store_true", help="print train commands without running")
    return p.parse_args()


def _ckpt_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints" / "last" / "pretrained_model"


def _train_command(args: argparse.Namespace, pretrained_path: str, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "sim.scripts.train_safe_pi0",
        "--policy.type=safe_pi0",
        f"--policy.pretrained_path={pretrained_path}",
        f"--policy.safety_weight={args.safety_weight}",
        f"--dataset.repo_id={args.repo_id}",
        f"--output_dir={output_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--seed={args.seed}",
    ]
    if args.root is not None:
        cmd.append(f"--dataset.root={args.root}")
    if args.gradient_checkpointing:
        cmd.append("--policy.gradient_checkpointing=true")
    if args.device is not None:
        cmd.append(f"--policy.device={args.device}")
    return cmd


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    scene = SceneConfig(mjcf_path=args.mjcf, n_red_cubes=args.n_red_cubes)
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=args.max_steps, seed=args.seed))
    obs, _ = env.reset(seed=args.seed)
    state_dim = obs["state"].shape[0]

    rec = EpisodeRecorder.create(
        repo_id=args.repo_id,
        root=args.root,
        n_red_cubes=args.n_red_cubes,
        image_size=scene.image_size,
        action_dim=env.action_dim,
        state_dim=state_dim,
        fps=env.cfg.fps,
    )
    expert = ScriptedExpert(env=env, cfg=ExpertConfig())
    grip_open = ExpertConfig().grip_open

    prev_ckpt = args.base_policy
    for r in range(args.rounds):
        alpha = args.alpha0 ** r
        # Round 0 is pure expert; later rounds load the previous checkpoint.
        policy = None
        if r > 0:
            policy = PolicyRollout(
                checkpoint=prev_ckpt,
                dataset_repo_id=args.repo_id,
                dataset_root=str(args.root) if args.root else None,
                device=args.device,
            )

        round_seed = args.seed + r * args.episodes_per_round
        stats = collect_dagger_round(
            env=env, expert=expert, rec=rec, task=TASK_DESCRIPTION,
            n_episodes=args.episodes_per_round, alpha=alpha, policy=policy,
            base_seed=round_seed, rng=rng, grip_open=grip_open,
        )
        print(f"[round {r}] {stats}")

        if args.no_train:
            continue

        out_dir = args.output_root / f"round_{r}"
        cmd = _train_command(args, prev_ckpt, out_dir)
        print("  train:", " ".join(cmd))
        if not args.print_only:
            subprocess.run(cmd, check=True)
            prev_ckpt = str(_ckpt_dir(out_dir))

    env.close()
    rec.finalize()
    print("DAgger loop complete. Latest checkpoint:", prev_ckpt)


if __name__ == "__main__":
    main()
