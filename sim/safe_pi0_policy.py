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
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

# Must precede the lerobot imports below: patches argparse for draccus' `X | None`
# types on Python 3.14. Shared with sim/safe_diffusion_policy.py (it is idempotent,
# so importing both policy classes patches only once).
import sim._draccus_compat  # noqa: F401

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy, make_att_2d_masks
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from sim.safety_geometry import (
    DEFAULT_COLLISION_LINKS,
    DEFAULT_COLLISION_OFFSETS,
    DEFAULT_COLLISION_RADII,
    FKChain,
    ceiling_hinge_loss,
    clearance_hinge_loss,
    collision_index,
    collision_sphere_clearance,
    height_ceiling_loss,
    held_cube_clearance,
    safety_loss,
    sdf_clearance,
)

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

    # ---- penalty SHAPE -------------------------------------------------------
    # "softplus" is the original form; "hinge" is the finite-support replacement.
    #
    # The softplus terms are large on the *expert's own actions*, so they compete
    # with imitation rather than constraining it. Measured on safe_cube_agg_v7.1
    # ground-truth expert actions (sim/scripts/calibrate_safety_loss.py):
    #
    #                        softplus     hinge
    #     L_obstacle          0.391       0.00062
    #     L_ceiling           0.585       0.00000
    #     L_safety (1,4)      2.731       0.00063
    #     vs L_flow=0.0087     314x        0.07x
    #
    # Two independent causes, both fixed by "hinge":
    #  1. softplus has infinite support -- at alpha=50, margin=0.02 the clearance
    #     must exceed 0.112 m before the penalty drops below 0.01, unreachable in
    #     a 0.30 m field of 8 cubes. It is a constant repulsive field, not a
    #     violation detector.
    #  2. `height_ceiling_loss` was fed the FK *gripper-frame* z against a
    #     threshold the env applies to the *held cube* (env.py:493-503, whose own
    #     comment warns the TCP "sits ~10 mm above the cube center and would
    #     false-fire at normal carry heights"). Measured offset: exactly 10.0 mm,
    #     so it false-fired on every carry frame -- L_ceiling 0.585 on a policy
    #     with 0/1000 env ceiling violations.
    #
    # Defaults to "softplus" so resuming an existing train_config.json is
    # unchanged; pass --policy.safety_form=hinge to opt in.
    safety_form: str = "softplus"         # "softplus" | "hinge"
    ceiling_buffer: float = 0.005         # hinge band below the ceiling (m)
    # Obstacle margin for the HINGE form only (the softplus form keeps sdf_margin).
    #
    # Deliberately ~4x tighter than sdf_margin=0.02, because the two forms are
    # measuring against different geometry. The softplus form queried one sphere
    # at a frame with no geometry on it, so its margin was buying standoff to
    # cover that error. The hinge form queries a sphere covering that *contains*
    # the arm meshes (see safety_geometry.DEFAULT_COLLISION_*), so a sphere being
    # clear already proves the geom is clear -- the covering's own residual
    # conservatism (measured -3 to -5 mm vs mj_geomDistance) is the standoff, and
    # stacking 20 mm on top of it would penalise the whole approach and retreat.
    # At 0.005 the term is 0 until 5 mm out and rises to 1.0 at the surface.
    hinge_margin: float = 0.005           # obstacle hinge band (m)
    # SEPARATE noise gate for the ceiling term. The two terms have opposite
    # noise sensitivity, so one shared gate cannot serve both:
    #
    #   gate   l_obstacle   l_ceiling     (safe_diffusion lam=0, margin 5 mm)
    #   0.25    0.000000     0.000000     <- obstacle term is INERT: no gradient
    #   0.50    0.000099     0.000167
    #   1.00    0.000541     0.009680     <- ceiling term is 58x noise-inflated
    #
    # The ceiling term is evaluated on x_hat_0 and blows up at high noise
    # (L_ceiling 0.00003 at k=0 vs 38.8 at k=99), so it needs the tight gate. The
    # obstacle term is noise-robust and only becomes a usable training signal
    # once the gate opens. Sharing 0.25 made L_safety exactly 0.0 on every
    # training batch -- lambda had no gradient and the sweep measured nothing.
    ceiling_max_noise_frac: float = 0.25
    # The carried blue cube, as a sphere in the TCP frame. Active only while
    # `grasped`; see safety_geometry.held_cube_clearance for why both the gating
    # and the non-zero offset matter.
    held_cube_offset: tuple[float, float, float] = (-0.00452, -0.01232, 0.00178)
    held_cube_radius: float = 0.0127      # blue cube half-extent (m)
    # Drop the safety term entirely for samples noisier than this fraction of the
    # schedule (flow time t here; t=0 is the data side).
    #
    # The endpoint estimate is only a faithful reconstruction near the data side;
    # at the noise side it is an extrapolation, and the safety term then measures
    # reconstruction error rather than the policy's behaviour. Measured on a
    # trained lambda=0 diffusion reference, per pinned timestep:
    #
    #       k        L_obstacle   L_ceiling      (hinge form)
    #       0           0.00050     0.00003
    #      25           0.00087     0.01692
    #      50           0.00219     0.13462
    #      99           0.01869    38.81836   <- pure noise, not behaviour
    #
    # Linear (1-t) weighting alone does NOT fix this: 0.01 x 38.8 still swamps
    # the 0.0005 the informative samples contribute. 1.0 disables the gate
    # (legacy behaviour); 0.25 keeps the term measuring what it claims to.
    safety_max_noise_frac: float = 1.0

    # TCP height minus held-cube height, measured over 15304 grasped expert
    # frames (median 10.0 mm, p5-p95 4.9-10.8 mm). Converts the FK TCP z into the
    # cube z the env actually checks.
    ee_to_cube_z_offset: float = 0.010
    # Collision spheres for the hinge form; see sim/safety_geometry.py.
    collision_links: list[str] = field(default_factory=lambda: list(DEFAULT_COLLISION_LINKS))
    collision_offsets: list[list[float]] = field(
        default_factory=lambda: [list(o) for o in DEFAULT_COLLISION_OFFSETS]
    )
    collision_radii: list[float] = field(default_factory=lambda: list(DEFAULT_COLLISION_RADII))

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
        self._sphere_links, self._sphere_index = collision_index(config.collision_links)

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
        self.register_buffer(
            "sphere_offsets",
            torch.tensor(config.collision_offsets, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sphere_radii",
            torch.tensor(config.collision_radii, dtype=torch.float32),
            persistent=False,
        )

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
    def _raw_action(self, action_chunk_norm: Tensor) -> Tensor:
        """Normalized action chunk -> raw joint targets, gripper dim included."""
        return action_chunk_norm * self.action_std + self.action_mean

    def _fk_chunk(self, action_chunk_norm: Tensor) -> Tensor:
        """Normalized action chunk -> world-frame ee trajectory.

        Args:
            action_chunk_norm: ``(B, T, action_dim)`` clean-action endpoint
                estimate in *normalized* space.

        Returns:
            ``(B, T, 3)`` ee positions in the world frame.
        """
        raw = self._raw_action(action_chunk_norm)
        arm_q = raw[..., : self._n_arm]                                # drop gripper dim
        return self._fk.fk(arm_q)

    def _cubes(self, batch: dict[str, Tensor], device) -> tuple[Tensor, Tensor]:
        if PRIV_CUBE_POSITIONS not in batch or PRIV_CUBE_HALF_EXTENTS not in batch:
            raise KeyError(
                f"Safety loss requires privileged keys '{PRIV_CUBE_POSITIONS}' and "
                f"'{PRIV_CUBE_HALF_EXTENTS}' in the batch, but they are missing. Confirm the "
                "data processor passes them through (they should not be normalized or dropped)."
            )
        b = batch[PRIV_CUBE_POSITIONS].shape[0]
        centers = batch[PRIV_CUBE_POSITIONS].to(device, torch.float32).reshape(b, -1, 3)
        halves = batch[PRIV_CUBE_HALF_EXTENTS].to(device, torch.float32).reshape(b, -1, 3)
        return centers, halves

    def _hinge_terms(
        self, raw_action: Tensor, ee_traj: Tensor, batch: dict[str, Tensor], time: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Finite-support obstacle + ceiling penalties. See ``safety_form``."""
        cfg = self.config
        centers, halves = self._cubes(batch, ee_traj.device)
        weight = (1.0 - time).clamp(0.0, 1.0) if cfg.safety_time_weighting else None
        if cfg.safety_max_noise_frac < 1.0:
            # t=0 is the data side, so "noisier than" means a LARGER t.
            gate = (time <= cfg.safety_max_noise_frac).to(ee_traj.dtype)
            weight = gate if weight is None else weight * gate

        clearance = collision_sphere_clearance(
            self._fk, raw_action[..., : self._n_arm + 1], centers, halves,
            links=self._sphere_links,
            link_index=self._sphere_index.to(ee_traj.device),
            offsets=self.sphere_offsets,
            radii=self.sphere_radii,
        )
        # The carried cube is its own obstacle body while grasped.
        tcp_mats = self._fk.fk_link_transforms(
            raw_action[..., : self._n_arm + 1], ["gripper_frame_link"]
        )[:, :, 0]
        cube_clear = held_cube_clearance(
            tcp_mats, centers, halves,
            offset=torch.tensor(cfg.held_cube_offset, dtype=clearance.dtype),
            radius=cfg.held_cube_radius,
            grasped=batch.get(PRIV_GRASPED),
        )
        clearance = torch.cat([clearance, cube_clear], dim=-1)
        l_obstacle = clearance_hinge_loss(clearance, margin=cfg.hinge_margin, weight=weight)

        # The env checks the HELD CUBE's height, not the TCP's (env.py:493-503).
        cube_z = ee_traj[..., 2] - cfg.ee_to_cube_z_offset
        cw = weight if weight is not None else torch.ones(
            cube_z.shape[0], device=cube_z.device, dtype=cube_z.dtype
        )
        if cfg.ceiling_grasped_only:
            grasped = batch.get(PRIV_GRASPED)
            if grasped is not None:
                cw = cw * grasped.to(cube_z.device, cube_z.dtype).reshape(cube_z.shape[0])
        if cfg.ceiling_max_noise_frac < 1.0:
            cw = cw * (time <= cfg.ceiling_max_noise_frac).to(cw.dtype).reshape(cw.shape)
        l_ceiling = ceiling_hinge_loss(
            cube_z, cfg.ee_height_ceiling, buffer=cfg.ceiling_buffer, weight=cw
        )
        return l_obstacle, l_ceiling

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

        # --- safety term: endpoint estimate -> FK -> SDF -> penalty ---
        a_hat = (x_t - t * v)[:, :, :action_dim]     # clean-action estimate (normalized)
        ee_traj = self._fk_chunk(a_hat)              # (B, chunk, 3) world frame
        if self.config.safety_form == "hinge":
            l_obstacle, l_ceiling = self._hinge_terms(
                self._raw_action(a_hat), ee_traj, batch, time
            )
        else:
            l_obstacle = self._safety_loss(ee_traj, batch, time)  # XY (lateral) cube avoidance
            l_ceiling = self._ceiling_loss(ee_traj, batch, time)  # stay-low / anti fly-over
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
