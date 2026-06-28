"""Constraint-aware π0 policy: ``L = L_flow_matching + λ · L_safety``.

This is the policy half of the project described in ``project_summary.md``.
It subclasses LeRobot's :class:`PI0Policy` and overrides :meth:`forward` so a
single velocity prediction feeds *both* the standard flow-matching loss and a
differentiable safety term that penalises action chunks whose predicted
end-effector trajectory enters a red obstacle cube.

Why a subclass and not a runtime filter
----------------------------------------
The safety preference is baked into the *weights* via the loss gradient, not
enforced by an online projection at deployment. At inference this behaves
exactly like a vanilla π0: ``image -> action`` with no SDF, no cube positions,
no filter. See ``project_summary.md`` §2 / §9.

The flow-matching convention (read this before touching the math)
-----------------------------------------------------------------
``project_summary.md`` §5.1 assumed the rectified-flow convention
``τ=1 → data, τ=0 → noise``. **LeRobot's π0 uses the opposite convention.**
From ``modeling_pi0.py`` (``PI0Pytorch.forward``)::

    x_t = t * noise + (1 - t) * actions      # t=0 -> data,  t=1 -> noise
    u_t = noise - actions                    # regression target for v_θ

So the network learns ``v_θ ≈ u_t = noise - actions``. Solving the interpolant
for the clean action chunk gives the **endpoint estimate** used by the safety
term::

    actions = x_t - t * v_θ                  # exact when v_θ == u_t

This estimate is most reliable near the *data* side (small ``t``) and noisy
near the *noise* side (large ``t``). LeRobot samples ``t`` from a Beta(1.5, 1)
that concentrates toward 1, so the safety loss is **weighted by ``(1 - t)``**
to down-weight unreliable high-``t`` estimates. (This is the mirror image of the
"weight by τ / high-τ band" advice in the summary, which was written for the
opposite convention — the convention mismatch is the documented footgun.)

Geometry primitives (``box_sdf`` / ``sdf_clearance`` / ``safety_loss`` / FK)
live in ``sim/safety_geometry.py`` — pure torch, no policy dependency — so they
can be unit-tested in isolation against hand-placed cubes before a long
fine-tune.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


# --- Python 3.14 + draccus argparse compatibility shim ----------------------
# CPython 3.14 made argparse strict about ``type=`` being callable, but draccus
# (≤ 0.11.x) still passes the raw ``X | None`` ``UnionType`` for Optional fields
# (e.g. ``PI0Config.device``), which crashes parsing with
# ``TypeError: str | None is not callable``. We collapse union types to their
# first non-None member here before argparse sees them. Patching argparse's
# ``_ActionsContainer.add_argument`` (the common base) catches both direct
# parser calls and argument-group calls — draccus uses both. Active anywhere
# ``safe_pi0`` is loaded (training and ``PolicyRollout`` both import this).
if sys.version_info >= (3, 14):  # pragma: no cover - environment-specific
    import argparse as _argparse
    import typing as _typing

    _orig_add_argument = _argparse._ActionsContainer.add_argument

    def _add_argument_union_safe(self, *args, **kwargs):
        t = kwargs.get("type")
        if t is not None and not callable(t):
            members = [a for a in _typing.get_args(t) if a is not type(None)]
            kwargs["type"] = members[0] if members else str
        return _orig_add_argument(self, *args, **kwargs)

    _argparse._ActionsContainer.add_argument = _add_argument_union_safe


from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy, make_att_2d_masks
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from sim.safety_geometry import FKChain, height_ceiling_loss, safety_loss, sdf_clearance

# Privileged batch keys written by sim.recorder.EpisodeRecorder. The policy
# never feeds these to the network; only the safety loss reads them, and they
# are absent at deployment.
PRIV_CUBE_POSITIONS = "privileged.cube_positions"
PRIV_CUBE_HALF_EXTENTS = "privileged.cube_half_extents"
PRIV_EE_POS = "privileged.ee_pos"
PRIV_GRASPED = "privileged.grasped"


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@PreTrainedConfig.register_subclass("safe_pi0")
@dataclass
class SafePI0Config(PI0Config):
    """π0 config plus the constraint-aware safety knobs.

    Inherits every π0 field (chunk_size, normalization, optimizer, ...). Defaults
    follow ``project_summary.md`` §8. ``λ`` (``safety_weight``) is the central
    knob — sweep it first.
    """

    chunk_size: int = 50
    n_action_steps: int = 50

    # Safety loss weighting and shape. ``safety_weight`` (λ) is the OVERALL safety
    # multiplier; ``obstacle_weight`` / ``ceiling_weight`` set the balance between
    # the lateral-clearance and stay-low terms. Effective coeffs in the total loss:
    #   collision = λ·obstacle_weight,  ceiling = λ·ceiling_weight,  flow = 1.0.
    # Default = collision 1.0 (1·1), ceiling 4.0 (1·4).
    safety_weight: float = 1.0            # λ — overall task vs. safety trade-off
    obstacle_weight: float = 1.0          # weight of the lateral-clearance term within L_safety
    sdf_alpha: float = 50.0               # sigmoid sharpness (1/m)
    sdf_margin: float = 0.02              # desired clearance buffer (m)
    ee_radius: float = 0.03               # gripper bounding-sphere radius (m)
    # Weight the safety term by (1 - t) so unreliable high-t endpoint estimates
    # count less. Set False to weight all flow-times equally.
    safety_time_weighting: bool = True

    # Obstacle avoidance geometry. With lateral_clearance=True the obstacle SDF
    # is measured in XY only (cubes as infinite vertical prisms), so lifting the
    # ee *over* a red cube no longer counts as avoidance — the policy must weave
    # *around* in the plane. Height is constrained separately by the ceiling term.
    lateral_clearance: bool = True

    # Stay-low ("don't fly over the obstacles") penalty. A smooth, differentiable
    # term that spikes when the predicted ee height exceeds `ee_height_ceiling`.
    # Mirrors the env's SceneConfig.ee_height_ceiling stay-low rule and, like the
    # env, is gated to the carry phase (privileged.grasped) so the initial
    # reach-down from the home pose (ee starts ~85 mm up) isn't penalised.
    ee_height_ceiling: float = 0.035      # m — keep in sync with SceneConfig
    # softplus sharpness (1/m). ~250 puts the knee a few mm wide: ≈0 at the
    # expert carry height (~23 mm cube center) and a sharp spike past the 35 mm ceiling.
    ceiling_alpha: float = 250.0
    ceiling_weight: float = 4.0           # weight of the height term within L_safety
    ceiling_grasped_only: bool = True     # only penalise height during carry (grasped)

    # Forward-kinematics source. Must match the arm the dataset was recorded on.
    urdf_path: str = "sim/assets/so101/so101_new_calib.urdf"
    ee_link_name: str = "gripper_frame_link"
    arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
    )


# ----------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------
class SafePI0Policy(PI0Policy):
    """π0 with collision avoidance baked into the training loss."""

    config_class = SafePI0Config
    name = "safe_pi0"

    def __init__(self, config: SafePI0Config, **kwargs):
        # `dataset_stats` and `dataset_meta` are passed by make_policy; PI0Policy
        # ignores them. We need the action mean/std to un-normalize the endpoint
        # estimate back to raw joint targets before FK.
        dataset_stats = kwargs.get("dataset_stats")
        super().__init__(config, **kwargs)
        self.config: SafePI0Config = config

        self._n_arm = len(config.arm_joint_names)
        self._fk = FKChain(config.urdf_path, config.ee_link_name, config.arm_joint_names)

        action_dim = config.output_features[ACTION].shape[0]
        mean = torch.zeros(action_dim)
        std = torch.ones(action_dim)
        if dataset_stats is not None and ACTION in dataset_stats:
            st = dataset_stats[ACTION]
            if "mean" in st and "std" in st:
                mean = torch.as_tensor(st["mean"], dtype=torch.float32).reshape(-1)[:action_dim]
                std = torch.as_tensor(st["std"], dtype=torch.float32).reshape(-1)[:action_dim]
            else:
                logging.warning(
                    "SafePI0Policy: dataset_stats[ACTION] has no mean/std; FK un-normalization "
                    "will be an identity. Safety loss geometry will be wrong."
                )
        else:
            logging.warning(
                "SafePI0Policy: no dataset_stats[ACTION] provided. The safety loss needs the "
                "action normalization stats to un-normalize the endpoint estimate before FK."
            )
        # Non-persistent so they don't collide with the loaded π0 checkpoint's
        # state_dict (strict=True load stays clean — we add no parameters).
        self.register_buffer("action_mean", mean, persistent=False)
        self.register_buffer("action_std", std, persistent=False)

    # ----- velocity field (single VLM encode feeds both losses) -------------
    def _flow_velocity(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        x_t: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Predict the flow velocity ``v_θ(x_t, t, obs)``.

        Mirrors the body of ``PI0Pytorch.forward`` but returns ``v`` instead of
        the MSE loss, so the same forward pass feeds the task loss *and* the
        endpoint estimate (no second encode of the images through the 3B VLM).
        """
        model = self.model
        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad, suffix_att, adarms_cond = model.embed_suffix(state, x_t, time)

        q_dtype = (
            model.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
        )
        if q_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = model.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = model._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)   # (B, chunk, max_action_dim)

    # ----- safety term -------------------------------------------------------
    def _fk_chunk(self, action_chunk_norm: Tensor) -> Tensor:
        """Normalized action chunk -> world-frame ee trajectory.

        Args:
            action_chunk_norm: ``(B, T, action_dim)`` clean-action endpoint
                estimate in *normalized* space.

        Returns:
            ``(B, T, 3)`` ee positions in the world frame.
        """
        raw = action_chunk_norm * self.action_std + self.action_mean   # un-normalize
        arm_q = raw[..., : self._n_arm]                                # drop gripper dim
        return self._fk.fk(arm_q)

    def _safety_loss(self, ee_traj: Tensor, batch: dict[str, Tensor], time: Tensor) -> Tensor:
        if PRIV_CUBE_POSITIONS not in batch or PRIV_CUBE_HALF_EXTENTS not in batch:
            raise KeyError(
                f"Safety loss requires privileged keys '{PRIV_CUBE_POSITIONS}' and "
                f"'{PRIV_CUBE_HALF_EXTENTS}' in the batch, but they are missing. Confirm the "
                "data processor passes them through (they should not be normalized or dropped)."
            )
        device = ee_traj.device
        cube_pos = batch[PRIV_CUBE_POSITIONS].to(device, torch.float32)        # (B, M*3)
        cube_half = batch[PRIV_CUBE_HALF_EXTENTS].to(device, torch.float32)    # (B, M*3)
        b = cube_pos.shape[0]
        centers = cube_pos.reshape(b, -1, 3)
        halves = cube_half.reshape(b, -1, 3)

        clearance = sdf_clearance(
            ee_traj, centers, halves, self.config.ee_radius, lateral=self.config.lateral_clearance
        )  # (B, T)
        weight = (1.0 - time).clamp(0.0, 1.0) if self.config.safety_time_weighting else None
        return safety_loss(
            clearance, alpha=self.config.sdf_alpha, margin=self.config.sdf_margin, weight=weight
        )

    def _ceiling_loss(self, ee_traj: Tensor, batch: dict[str, Tensor], time: Tensor) -> Tensor:
        """Smooth, grasp-gated penalty for the ee rising above the height ceiling.

        Weighted by ``(1 - t)`` (down-weight noisy high-t endpoint estimates) and,
        when ``ceiling_grasped_only``, by ``privileged.grasped`` so only carry-phase
        frames are penalised — matching the env, which enforces the ceiling only
        once the blue cube is grasped.
        """
        cfg = self.config
        ee_z = ee_traj[..., 2]                                        # (B, T)
        b = ee_z.shape[0]
        if cfg.safety_time_weighting:
            weight = (1.0 - time).clamp(0.0, 1.0).to(ee_z.device)     # (B,)
        else:
            weight = torch.ones(b, device=ee_z.device, dtype=ee_z.dtype)
        if cfg.ceiling_grasped_only:
            grasped = batch.get(PRIV_GRASPED)
            if grasped is not None:
                weight = weight * grasped.to(ee_z.device, ee_z.dtype).reshape(b)
            else:
                logging.warning(
                    "SafePI0Policy: ceiling_grasped_only=True but '%s' is missing from the batch; "
                    "applying the height-ceiling penalty to all frames.", PRIV_GRASPED
                )
        return height_ceiling_loss(
            ee_z, cfg.ee_height_ceiling, alpha=cfg.ceiling_alpha, weight=weight
        )

    # ----- training forward --------------------------------------------------
    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)            # (B, max_state_dim)
        actions = self.prepare_action(batch)         # (B, chunk, max_action_dim)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)   # (B,) in (0, 1)

        # LeRobot convention: t=0 -> data, t=1 -> noise.
        t = time[:, None, None]
        x_t = t * noise + (1.0 - t) * actions
        u_t = noise - actions

        v = self._flow_velocity(images, img_masks, lang_tokens, lang_masks, state, x_t, time)

        action_dim = self.config.output_features[ACTION].shape[0]

        # --- task term: identical to π0's flow-matching MSE ---
        flow_losses = F.mse_loss(u_t, v, reduction="none")[:, :, :action_dim]   # (B, chunk, act)
        l_flow = flow_losses.mean()

        # --- safety term: endpoint estimate -> FK -> SDF -> -log p_safe ---
        a_hat = (x_t - t * v)[:, :, :action_dim]     # clean-action estimate (normalized)
        ee_traj = self._fk_chunk(a_hat)              # (B, chunk, 3) world frame
        l_obstacle = self._safety_loss(ee_traj, batch, time)      # XY (lateral) cube avoidance
        l_ceiling = self._ceiling_loss(ee_traj, batch, time)      # stay-low / anti fly-over
        l_safety = self.config.obstacle_weight * l_obstacle + self.config.ceiling_weight * l_ceiling

        loss = l_flow + self.config.safety_weight * l_safety

        loss_dict = {
            "loss": loss.item(),
            "l_flow": l_flow.item(),
            "l_safety": l_safety.item(),
            "l_obstacle": l_obstacle.item(),
            "l_ceiling": l_ceiling.item(),
            "loss_per_dim": flow_losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            per_sample = flow_losses.mean(dim=(1, 2))
            return per_sample, loss_dict
        return loss, loss_dict
