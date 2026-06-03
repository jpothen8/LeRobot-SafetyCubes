"""DAgger primitives for the constraint-aware cube task.

Two reusable pieces:

* :class:`PolicyRollout` — wraps a trained (Safe)PI0 checkpoint behind a simple
  ``act(obs, task) -> np.ndarray`` interface, mirroring LeRobot's eval-time
  inference path (preprocessor → ``select_action`` → postprocessor). Also usable
  as the ``policy`` argument to :func:`sim.evaluate.evaluate`.
* :func:`collect_dagger_round` — rolls episodes with smooth expert/policy
  mixing, **always labels with the expert action** (the DAgger invariant), and
  records expert-labeled frames into an :class:`EpisodeRecorder`. Early policy
  rollouts naturally produce the failure data the safety term needs.

The CLI that strings rounds together (collect → retrain → repeat) lives in
``sim/scripts/dagger.py``; this module holds the logic so it can be imported and
tested without the argparse shell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import sim.safe_pi0_policy  # noqa: F401  -- registers `safe_pi0` with draccus
from sim.env import SafeCubeEnv
from sim.expert import ScriptedExpert
from sim.recorder import EpisodeRecorder
from sim.rollout import run_expert_episode


@dataclass
class RoundStats:
    episodes: int = 0
    saved: int = 0
    successes: int = 0
    red_contacts: int = 0
    alpha: float = 0.0

    def __str__(self) -> str:
        return (
            f"alpha={self.alpha:.3f} episodes={self.episodes} saved={self.saved} "
            f"success={self.successes}/{self.episodes} red_contact={self.red_contacts}/{self.episodes}"
        )


class PolicyRollout:
    """Run a trained checkpoint closed-loop on a single (non-vectorized) env.

    Args:
        checkpoint: path to a saved ``pretrained_model`` dir (LeRobot layout).
        dataset_repo_id / dataset_root: a dataset whose metadata supplies the
            feature shapes and normalization stats. Use the same dataset the
            policy was trained on.
        device: torch device string; defaults to the policy config's device.
        image_key: main (agentview) dataset image feature key.
        wrist_image_key: wrist-cam dataset image feature key.
    """

    def __init__(
        self,
        *,
        checkpoint: str,
        dataset_repo_id: str,
        dataset_root: str | None = None,
        device: str | None = None,
        image_key: str = "observation.images.agentview",
        wrist_image_key: str = "observation.images.wrist",
    ) -> None:
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies import make_policy, make_pre_post_processors

        ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        if device is not None:
            cfg.device = device

        self.policy = make_policy(cfg, ds_meta=ds_meta)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, pretrained_path=checkpoint, dataset_stats=ds_meta.stats
        )
        self.policy.eval()
        self.device = cfg.device
        self.image_key = image_key
        self.wrist_image_key = wrist_image_key

    def reset(self) -> None:
        self.policy.reset()

    def _format(self, obs: dict, task: str) -> dict:
        def to_chw(arr) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(arr))         # (H, W, 3) uint8
            return t.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # (1, 3, H, W) in [0, 1]

        state = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)).unsqueeze(0)
        batch = {
            self.image_key: to_chw(obs["image"]).to(self.device),
            "observation.state": state.to(self.device),
            "task": [task],
        }
        # Wrist cam is a second policy camera; harmless if the loaded policy was
        # trained single-cam (_preprocess_images only reads its config's keys).
        if "wrist_image" in obs:
            batch[self.wrist_image_key] = to_chw(obs["wrist_image"]).to(self.device)
        return batch

    def act(self, obs: dict, task: str) -> np.ndarray:
        """Return a single raw-joint action ``(action_dim,)`` for ``obs``.

        Re-infers a fresh chunk each call (queue reset) so the action always
        reflects the current state — the right granularity for DAgger relabeling.
        """
        self.policy.reset()
        batch = self.preprocessor(self._format(obs, task))
        with torch.inference_mode():
            action = self.policy.select_action(batch)   # (1, action_dim), normalized
        action = self.postprocessor(action)              # un-normalized raw joints
        return action.squeeze(0).to("cpu").numpy()

    def act_queued(self, obs: dict, task: str) -> np.ndarray:
        """Like :meth:`act` but WITHOUT the per-step queue reset, so
        ``select_action`` executes the predicted action chunk before re-planning
        — the exact closed-loop path used at deployment/eval. Call
        :meth:`reset` once per episode. Use this to *generate* DAgger rollout
        states so they match the deployment distribution.
        """
        batch = self.preprocessor(self._format(obs, task))
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.squeeze(0).to("cpu").numpy()


def collect_dagger_round(
    *,
    env: SafeCubeEnv,
    expert: ScriptedExpert,
    rec: EpisodeRecorder,
    task: str,
    n_episodes: int,
    alpha: float,
    policy: PolicyRollout | None,
    base_seed: int,
    rng: np.random.Generator,
    grip_open: float,
    successes_only: bool = False,
) -> RoundStats:
    """One DAgger collection round.

    At each step the *executed* action is the expert's with probability
    ``alpha`` and the policy's otherwise (``alpha=1`` → pure expert, which is the
    right setting for round 0 when there is no policy yet). The *recorded* label
    is always the expert action, so the dataset stays an expert-supervised set on
    the policy's own state distribution.

    Args:
        alpha: expert-mixing probability for this round (e.g. ``alpha0 ** round``).
        policy: trained policy wrapper, or ``None`` for pure-expert collection.
        grip_open: gripper command to hold after the expert finishes (prevents
            the magnetic grip from re-attaching the released cube — see CLAUDE.md).
    """
    stats = RoundStats(alpha=alpha)
    pure_expert = policy is None or alpha >= 1.0

    def choose_executed(expert_action: np.ndarray, obs: dict, info: dict) -> np.ndarray:
        # Execute the expert w.p. alpha, else the policy rolled the SAME (queued)
        # way it is deployed, so DAgger visits the deployment distribution. The
        # label recorded inside run_expert_episode is always the expert action.
        if rng.random() < alpha:
            return expert_action
        return policy.act_queued(obs, task)

    for ep in range(n_episodes):
        if policy is not None:
            policy.reset()  # reset the action queue once per episode (queued rollout)
        # SAME expert path as sim.scripts.collect_demos -> identical labels.
        s = run_expert_episode(
            env=env, expert=expert, rec=rec, task=task,
            seed=base_seed + ep, grip_open=grip_open,
            choose_executed=None if pure_expert else choose_executed,
        )
        stats.episodes += 1
        stats.successes += int(s["success"])
        stats.red_contacts += int(s["red_contact"])
        if successes_only and not s["success"]:
            rec.discard_episode()
        else:
            rec.save_episode()
            stats.saved += 1
    return stats
