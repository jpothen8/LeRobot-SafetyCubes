"""Constraint-aware Diffusion Policy: ``L = L_diffusion + λ · L_safety``.

The **second policy class** of this project, alongside ``sim/safe_pi0_policy.py``.
Same safety loss, same differentiable FK → box-SDF → clearance pipeline
(``sim/safety_geometry.py``), same privileged-key contract — a different
generative backbone underneath.

Why a second class
------------------
``safe_pi0`` is a 3.3B-param VLA: ~5 s/step, ~77 GB at batch 48, ~21 h for a
15 k-step run. That cost is what has kept ``λ`` (``safety_weight``) — "the
central knob, sweep it first" — effectively unswept. Diffusion Policy (Chi et
al., `arXiv:2303.04137`) is a ~50 M-param ResNet18 + 1D temporal U-Net trained
from scratch, so the same run is hours instead of a day. It is the cheap vehicle
for sweeping the safety coefficients, and it adds an architecture axis to the
ablation table: *the safety loss works across generative policy classes*, not
just on π0.

The trade: no internet-pretrained vision-language prior and no language
conditioning. For this task (one fixed instruction, fixed cameras, sim-only,
~2 k scripted-expert demos) that prior is mostly unexercised — but expect a
lower absolute ceiling than π0 and treat cross-class success numbers as
*comparable trends*, not interchangeable.

The diffusion convention (read this before touching the math)
-------------------------------------------------------------
``safe_pi0`` documents a footgun: π0's flow-matching convention is
``t=0 → data, t=1 → noise``, the mirror image of the rectified-flow convention
in ``project_summary.md`` §5.1. DDPM has **no such flip** — the noise level
rises monotonically with the timestep ``k``, so "small k = data side" reads the
way you expect.

diffusers' forward process (``DDPMScheduler.add_noise``)::

    x_k = sqrt(ᾱ_k) · x_0 + sqrt(1 - ᾱ_k) · eps          # k=0 -> data, k=T -> noise

With ``prediction_type="epsilon"`` the U-Net learns ``eps_θ ≈ eps``, so solving
the interpolant for the clean action chunk gives the **endpoint estimate** the
safety term differentiates through::

    x̂_0 = (x_k - sqrt(1 - ᾱ_k) · eps_θ) / sqrt(ᾱ_k)      # exact when eps_θ == eps

This is the DDPM counterpart of ``safe_pi0``'s ``a_hat = x_t - t · v_θ``, and it
plays the same role: recover the predicted *clean* chunk analytically so
gradients reach the network through FK and the SDF. With
``prediction_type="sample"`` the network already outputs ``x̂_0`` and the
estimate is the raw prediction.

Two numerical caveats that ``safe_pi0`` does **not** have (flow matching's
``x_t - t·v`` is well-conditioned everywhere):

1. **The ``1/sqrt(ᾱ_k)`` blow-up.** As ``k → T``, ``ᾱ_k → 0`` and the estimate
   amplifies network error without bound. Mitigated two ways — the
   ``(1 - k/(T-1))`` time weighting (mirroring ``safe_pi0``'s ``(1 - t)``
   down-weighting of unreliable estimates), and ``clamp_endpoint``.
2. **``clamp_endpoint``** clamps ``x̂_0`` to ``±clip_sample_range``. This is not
   an arbitrary guard: ``DiffusionConfig`` normalizes ACTION with ``MIN_MAX``,
   so clean actions live in exactly ``[-1, 1]``, and the DDPM sampler applies
   this same clip at *every* reverse step when ``clip_sample=True``. The safety
   term therefore sees the same bounded estimate inference does. Note a hard
   clamp zeroes the gradient outside the range; that is intended (an
   out-of-range estimate carries no usable geometry) but it means very noisy
   timesteps contribute little — which the time weighting already wanted.

Everything else — the box SDF, the lateral-only clearance, the grasp-gated
height ceiling, the ``privileged.*`` loss-only contract — is shared verbatim
with ``safe_pi0``, so the coefficients (``safety_weight`` / ``obstacle_weight``
/ ``ceiling_weight`` / ``sdf_margin``) mean the same thing in both classes.

**They do NOT transfer numerically.** The formula is shared; the *effective
strength* is not, because the two imitation losses have different absolute scales.
Measured on converged checkpoints at identical geometry, margin and gates::

    safe_diffusion   L_safety / L_diffusion = 0.765   (92% of batches non-zero)
    pi0              L_safety / L_flow      = 0.027   ( 6% of batches non-zero)

pi0's flow-matching loss is ~6.7x larger in absolute terms (0.0096 vs 0.0014), so
the same lambda is ~28x weaker relative to the objective it has to modify. Port a
swept lambda by matching the **ratio** ``L_safety / L_task``, never the raw number:
lambda=0.25 on safe_diffusion corresponds to lambda~3.5 on pi0. See CLAUDE.md 2d.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

import sim._draccus_compat  # noqa: F401  -- side effect: py3.14 argparse shim

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

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

# Privileged batch keys written by sim.recorder.EpisodeRecorder. The policy never
# feeds these to the network; only the safety loss reads them, and they are absent
# at deployment. Identical contract to sim/safe_pi0_policy.py.
PRIV_CUBE_POSITIONS = "privileged.cube_positions"
PRIV_CUBE_HALF_EXTENTS = "privileged.cube_half_extents"
PRIV_EE_POS = "privileged.ee_pos"
PRIV_GRASPED = "privileged.grasped"


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@PreTrainedConfig.register_subclass("safe_diffusion")
@dataclass
class SafeDiffusionConfig(DiffusionConfig):
    """Diffusion Policy config plus the constraint-aware safety knobs.

    The safety fields are a **verbatim mirror** of ``SafePI0Config`` so a sweep
    script can drive either class with the same flags and the numbers mean the
    same thing.

    Naming is load-bearing: LeRobot's ``_get_policy_cls_from_policy_name``
    derives the policy class from the config class name (``SafeDiffusionConfig``
    → ``SafeDiffusionPolicy``) and looks for it in this same module. Renaming
    either one breaks ``--policy.type=safe_diffusion`` resolution.
    """

    # --- horizon / chunking -------------------------------------------------
    # Defaults chosen for PARITY with SafePI0Config (chunk_size = n_action_steps
    # = 50), so a cross-class comparison isolates the backbone instead of
    # confounding it with the control horizon. `horizon` must be a multiple of
    # 2**len(down_dims) = 8, hence 56 rather than 50.
    #
    # NOTE: n_action_steps = 50 means fully open-loop execution of the whole
    # chunk, matching π0. Diffusion Policy's *native* mode is receding-horizon
    # (predict many, execute a few, re-plan) — lowering `n_action_steps` to
    # ~8-16 closes the loop and is the first thing to try against the
    # placement-undershoot failure mode, which is a consequence of committing to
    # a long chunk. It is an inference-time knob; changing it needs no retrain.
    n_obs_steps: int = 1
    horizon: int = 56
    n_action_steps: int = 50
    # horizon - n_action_steps - n_obs_steps + 1
    drop_n_last_frames: int = 6

    # --- safety loss weighting and shape (mirrors SafePI0Config) ------------
    # `safety_weight` (λ) is the OVERALL safety multiplier; `obstacle_weight` /
    # `ceiling_weight` set the balance between the lateral-clearance and stay-low
    # terms. Effective coeffs in the total loss:
    #   collision = λ·obstacle_weight,  ceiling = λ·ceiling_weight,  diffusion = 1.0.
    safety_weight: float = 1.0            # λ — overall task vs. safety trade-off
    obstacle_weight: float = 1.0          # weight of the lateral-clearance term within L_safety
    sdf_alpha: float = 50.0               # sigmoid sharpness (1/m)
    sdf_margin: float = 0.02              # desired clearance buffer (m)
    ee_radius: float = 0.03               # gripper bounding-sphere radius (m)
    # Weight the safety term by (1 - k/(T-1)) so unreliable high-noise endpoint
    # estimates count less. Set False to weight all timesteps equally.
    safety_time_weighting: bool = True
    # Clamp the endpoint estimate to ±clip_sample_range before FK. See the module
    # docstring — this matches what the DDPM sampler does at every reverse step.
    clamp_endpoint: bool = True

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
    ceiling_alpha: float = 250.0          # softplus sharpness (1/m)
    ceiling_weight: float = 4.0           # weight of the height term within L_safety
    ceiling_grasped_only: bool = True     # only penalise height during carry (grasped)

    # ---- penalty SHAPE (identical contract to safe_pi0; see that file's notes) --
    # "softplus" is the original form, "hinge" the finite-support replacement.
    # On expert data the softplus terms measure L_obstacle 0.391 / L_ceiling 0.585
    # -- 314x the converged imitation loss -- so they compete with imitation
    # rather than constraining it; the hinge forms measure 0.00062 / 0.0 (0.07x).
    # Defaults to "softplus" so resuming an existing train_config.json is
    # unchanged; pass --policy.safety_form=hinge to opt in.
    safety_form: str = "softplus"         # "softplus" | "hinge"
    ceiling_buffer: float = 0.005         # hinge band below the ceiling (m)
    # Obstacle margin for the HINGE form only; see safe_pi0_policy.py for why it
    # is ~4x tighter than sdf_margin (the sphere covering contains the arm, so it
    # supplies its own standoff and does not need 20 mm stacked on top).
    hinge_margin: float = 0.005           # obstacle hinge band (m)
    # SEPARATE noise gate for the ceiling term; see safe_pi0_policy.py for the
    # measured table. In short: sharing one gate at 0.25 made the OBSTACLE term
    # exactly 0.0 on every training batch (no gradient, sweep measures nothing),
    # while opening the gate to 1.0 inflates the CEILING term 58x with
    # reconstruction noise. They need different gates.
    ceiling_max_noise_frac: float = 0.25
    # The carried blue cube, as a sphere in the TCP frame. Active only while
    # `grasped`; see safety_geometry.held_cube_clearance for why both the gating
    # and the non-zero offset matter.
    held_cube_offset: tuple[float, float, float] = (-0.00452, -0.01232, 0.00178)
    held_cube_radius: float = 0.0127      # blue cube half-extent (m)
    # Drop the safety term for samples noisier than this fraction of the DDPM
    # schedule. See safe_pi0_policy.py for the measured per-k table; in short,
    # at k=99 the endpoint estimate's 1/sqrt(alpha_bar) blow-up makes L_ceiling
    # 38.8 where the policy's real carry height would give ~0. 1.0 disables the
    # gate (legacy behaviour); 0.25 keeps the term measuring behaviour.
    safety_max_noise_frac: float = 1.0

    ee_to_cube_z_offset: float = 0.010    # measured TCP-to-held-cube z gap
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
class SafeDiffusionPolicy(DiffusionPolicy):
    """Diffusion Policy with collision avoidance baked into the training loss.

    At inference this is a vanilla Diffusion Policy — ``image → action``, no SDF,
    no cube positions, no runtime filter. The constraint lives in the weights.
    """

    config_class = SafeDiffusionConfig
    name = "safe_diffusion"

    def __init__(self, config: SafeDiffusionConfig, **kwargs):
        # `dataset_stats` is passed by make_policy; DiffusionPolicy ignores it
        # (its normalization lives in the processor pipeline). We need the ACTION
        # stats to un-normalize the endpoint estimate back to raw joint targets
        # before FK.
        dataset_stats = kwargs.get("dataset_stats")
        super().__init__(config, **kwargs)
        self.config: SafeDiffusionConfig = config

        self._n_arm = len(config.arm_joint_names)
        self._fk = FKChain(config.urdf_path, config.ee_link_name, config.arm_joint_names)
        self._sphere_links, self._sphere_index = collision_index(config.collision_links)
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

        action_dim = config.output_features[ACTION].shape[0]
        scale, offset = self._action_unnormalization(config, dataset_stats, action_dim)
        # Non-persistent so they don't collide with a checkpoint's state_dict
        # (we add no parameters — a strict load stays clean).
        self.register_buffer("action_scale", scale, persistent=False)
        self.register_buffer("action_offset", offset, persistent=False)

    # ----- un-normalization --------------------------------------------------
    @staticmethod
    def _action_unnormalization(
        config: SafeDiffusionConfig, dataset_stats: dict | None, action_dim: int
    ) -> tuple[Tensor, Tensor]:
        """Affine coefficients mapping a normalized action back to raw joints.

        Returns ``(scale, offset)`` such that ``raw = normalized * scale + offset``.

        Both supported normalization modes are affine, but they are **not** the
        same affine map — and this is the single easiest way to silently corrupt
        the safety geometry. ``safe_pi0`` inherits π0's ``MEAN_STD`` for ACTION;
        ``DiffusionConfig`` defaults to ``MIN_MAX``. Getting this wrong doesn't
        crash — it just feeds FK plausible-looking garbage, so the SDF penalises
        the wrong region of space.

            MEAN_STD:  raw = x·std + mean
            MIN_MAX:   raw = (x + 1)/2·(max - min) + min
                           = x·(max - min)/2 + (max + min)/2
        """
        mode = config.normalization_mapping.get(FeatureType.ACTION.value, NormalizationMode.MIN_MAX)
        identity = (torch.ones(action_dim), torch.zeros(action_dim))

        if dataset_stats is None or ACTION not in dataset_stats:
            logging.warning(
                "SafeDiffusionPolicy: no dataset_stats[ACTION] provided. The safety loss needs the "
                "action normalization stats to un-normalize the endpoint estimate before FK; "
                "falling back to identity, which makes the safety geometry WRONG."
            )
            return identity

        stats = dataset_stats[ACTION]

        def vec(name: str) -> Tensor:
            return torch.as_tensor(stats[name], dtype=torch.float32).reshape(-1)[:action_dim]

        if mode == NormalizationMode.MEAN_STD:
            if "mean" not in stats or "std" not in stats:
                logging.warning(
                    "SafeDiffusionPolicy: MEAN_STD action normalization but dataset_stats[ACTION] "
                    "has no mean/std; FK un-normalization will be an identity. Safety geometry "
                    "will be wrong."
                )
                return identity
            return vec("std"), vec("mean")

        if mode == NormalizationMode.MIN_MAX:
            if "min" not in stats or "max" not in stats:
                logging.warning(
                    "SafeDiffusionPolicy: MIN_MAX action normalization but dataset_stats[ACTION] "
                    "has no min/max; FK un-normalization will be an identity. Safety geometry "
                    "will be wrong."
                )
                return identity
            lo, hi = vec("min"), vec("max")
            return (hi - lo) / 2.0, (hi + lo) / 2.0

        raise ValueError(
            f"SafeDiffusionPolicy supports MEAN_STD or MIN_MAX action normalization for the safety "
            f"loss' FK un-normalization, but normalization_mapping[ACTION] is {mode}."
        )

    # ----- endpoint estimate -------------------------------------------------
    def _endpoint_estimate(self, x_k: Tensor, pred: Tensor, timesteps: Tensor) -> Tensor:
        """Recover the predicted clean action chunk ``x̂_0`` from a noisy sample.

        Args:
            x_k: ``(B, T, action_dim)`` noised trajectory at timestep ``k``.
            pred: ``(B, T, action_dim)`` U-Net output (``eps`` or ``sample``).
            timesteps: ``(B,)`` long tensor of diffusion timesteps.

        Returns:
            ``(B, T, action_dim)`` clean-action estimate in *normalized* space.
        """
        if self.config.prediction_type == "sample":
            x0 = pred
        else:  # "epsilon" — validated by DiffusionConfig.__post_init__
            ac = self.diffusion.noise_scheduler.alphas_cumprod.to(
                device=x_k.device, dtype=x_k.dtype
            )
            a_bar = ac[timesteps][:, None, None]                       # (B, 1, 1)
            x0 = (x_k - (1.0 - a_bar).sqrt() * pred) / a_bar.sqrt()

        if self.config.clamp_endpoint:
            r = self.config.clip_sample_range
            x0 = x0.clamp(-r, r)
        return x0

    def _time_weight(self, timesteps: Tensor, dtype: torch.dtype) -> Tensor | None:
        """``(B,)`` weight in [0, 1]: 1 at the data side (k=0), 0 at pure noise.

        The DDPM counterpart of ``safe_pi0``'s ``(1 - t)`` — and here the
        convention needs no flip, since noise rises monotonically with ``k``.
        """
        cfg = self.config
        t_max = max(cfg.num_train_timesteps - 1, 1)
        frac = timesteps.to(dtype) / t_max
        w = None if not cfg.safety_time_weighting else 1.0 - frac
        if cfg.safety_max_noise_frac < 1.0:
            gate = (frac <= cfg.safety_max_noise_frac).to(dtype)
            w = gate if w is None else w * gate
        return w

    # ----- safety terms (shared geometry with safe_pi0) ---------------------
    def _raw_action(self, action_chunk_norm: Tensor) -> Tensor:
        """Normalized action chunk -> raw joint targets, gripper dim included."""
        return action_chunk_norm * self.action_scale + self.action_offset

    def _fk_chunk(self, action_chunk_norm: Tensor) -> Tensor:
        """Normalized action chunk -> world-frame ee trajectory ``(B, T, 3)``."""
        raw = self._raw_action(action_chunk_norm)
        arm_q = raw[..., : self._n_arm]                                # drop gripper dim
        return self._fk.fk(arm_q)

    def _hinge_terms(
        self, raw_action: Tensor, ee_traj: Tensor, batch: dict[str, Tensor],
        weight: Tensor | None, timesteps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Finite-support obstacle + ceiling penalties. See ``safety_form``."""
        cfg = self.config
        b = ee_traj.shape[0]
        centers = batch[PRIV_CUBE_POSITIONS].to(ee_traj.device, torch.float32).reshape(b, -1, 3)
        halves = batch[PRIV_CUBE_HALF_EXTENTS].to(ee_traj.device, torch.float32).reshape(b, -1, 3)

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
        w = weight if weight is not None else torch.ones(
            b, device=cube_z.device, dtype=cube_z.dtype
        )
        if cfg.ceiling_grasped_only:
            grasped = batch.get(PRIV_GRASPED)
            if grasped is not None:
                w = w * grasped.to(cube_z.device, cube_z.dtype).reshape(b)
        if cfg.ceiling_max_noise_frac < 1.0:
            frac = timesteps.to(w.dtype) / max(1, cfg.num_train_timesteps - 1)
            w = w * (frac <= cfg.ceiling_max_noise_frac).to(w.dtype)
        l_ceiling = ceiling_hinge_loss(
            cube_z, cfg.ee_height_ceiling, buffer=cfg.ceiling_buffer, weight=w
        )
        return l_obstacle, l_ceiling

    def _safety_loss(self, ee_traj: Tensor, batch: dict[str, Tensor], weight: Tensor | None) -> Tensor:
        if PRIV_CUBE_POSITIONS not in batch or PRIV_CUBE_HALF_EXTENTS not in batch:
            raise KeyError(
                f"Safety loss requires privileged keys '{PRIV_CUBE_POSITIONS}' and "
                f"'{PRIV_CUBE_HALF_EXTENTS}' in the batch, but they are missing. Confirm the "
                "data processor passes them through (they should not be normalized or dropped)."
            )
        device = ee_traj.device
        cube_pos = batch[PRIV_CUBE_POSITIONS].to(device, torch.float32)
        cube_half = batch[PRIV_CUBE_HALF_EXTENTS].to(device, torch.float32)
        b = ee_traj.shape[0]
        # Tolerates (B, M*3) and (B, 1, M*3) — the privileged keys carry no
        # delta_timestamps, so they arrive un-stacked regardless of n_obs_steps.
        centers = cube_pos.reshape(b, -1, 3)
        halves = cube_half.reshape(b, -1, 3)

        clearance = sdf_clearance(
            ee_traj, centers, halves, self.config.ee_radius, lateral=self.config.lateral_clearance
        )  # (B, T)
        return safety_loss(
            clearance, alpha=self.config.sdf_alpha, margin=self.config.sdf_margin, weight=weight
        )

    def _ceiling_loss(self, ee_traj: Tensor, batch: dict[str, Tensor], weight: Tensor | None) -> Tensor:
        """Smooth, grasp-gated penalty for the ee rising above the height ceiling."""
        cfg = self.config
        ee_z = ee_traj[..., 2]                                        # (B, T)
        b = ee_z.shape[0]
        w = weight if weight is not None else torch.ones(b, device=ee_z.device, dtype=ee_z.dtype)
        if cfg.ceiling_grasped_only:
            grasped = batch.get(PRIV_GRASPED)
            if grasped is not None:
                w = w * grasped.to(ee_z.device, ee_z.dtype).reshape(b)
            else:
                logging.warning(
                    "SafeDiffusionPolicy: ceiling_grasped_only=True but '%s' is missing from the "
                    "batch; applying the height-ceiling penalty to all frames.", PRIV_GRASPED
                )
        return height_ceiling_loss(
            ee_z, cfg.ee_height_ceiling, alpha=cfg.ceiling_alpha, weight=w
        )

    # ----- training forward --------------------------------------------------
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """``L = L_diffusion + λ·(obstacle_weight·L_obstacle + ceiling_weight·L_ceiling)``.

        Mirrors ``DiffusionModel.compute_loss`` rather than calling it, because
        the safety term needs the intermediates (``noisy_trajectory``, ``pred``,
        ``timesteps``) that method discards — the same reason ``safe_pi0``
        reimplements π0's ``_flow_velocity``. One U-Net forward feeds both terms.
        """
        cfg = self.config

        # Image stacking, verbatim from DiffusionPolicy.forward.
        if cfg.image_features:
            batch = dict(batch)   # shallow copy so adding a key doesn't mutate the caller's
            for key in cfg.image_features:
                if cfg.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in cfg.image_features], dim=-4)

        model = self.diffusion
        assert set(batch).issuperset({OBS_STATE, ACTION})
        trajectory = batch[ACTION]                                     # (B, horizon, action_dim)

        global_cond = model._prepare_global_conditioning(batch)        # (B, global_cond_dim)

        eps = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)
        timesteps = torch.randint(
            low=0,
            high=model.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = model.noise_scheduler.add_noise(trajectory, eps, timesteps)

        pred = model.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # --- task term: identical to stock Diffusion Policy ---
        target = eps if cfg.prediction_type == "epsilon" else trajectory
        diffusion_losses = F.mse_loss(pred, target, reduction="none")  # (B, horizon, action_dim)
        if cfg.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{cfg.do_mask_loss_for_padding=}."
                )
            mask = (~batch["action_is_pad"]).unsqueeze(-1)
            num_valid = mask.sum() * diffusion_losses.shape[-1]
            l_diffusion = (diffusion_losses * mask).sum() / num_valid.clamp_min(1)
        else:
            l_diffusion = diffusion_losses.mean()

        # --- safety term: endpoint estimate -> FK -> SDF -> -log p_safe ---
        action_dim = cfg.output_features[ACTION].shape[0]
        a_hat = self._endpoint_estimate(noisy_trajectory, pred, timesteps)[..., :action_dim]
        ee_traj = self._fk_chunk(a_hat)                                # (B, horizon, 3) world frame
        weight = self._time_weight(timesteps, ee_traj.dtype)
        if cfg.safety_form == "hinge":
            l_obstacle, l_ceiling = self._hinge_terms(
                self._raw_action(a_hat), ee_traj, batch, weight, timesteps
            )
        else:
            l_obstacle = self._safety_loss(ee_traj, batch, weight)     # XY (lateral) cube avoidance
            l_ceiling = self._ceiling_loss(ee_traj, batch, weight)     # stay-low / anti fly-over
        l_safety = cfg.obstacle_weight * l_obstacle + cfg.ceiling_weight * l_ceiling

        loss = l_diffusion + cfg.safety_weight * l_safety

        # `l_safety` / `l_obstacle` / `l_ceiling` are directly comparable to the
        # safe_pi0 run's panels (same geometry, same units). `l_diffusion` is this
        # class' counterpart of safe_pi0's `l_flow` — a different objective, so
        # its magnitude is NOT comparable across classes.
        loss_dict = {
            "loss": loss.item(),
            "l_diffusion": l_diffusion.item(),
            "l_safety": l_safety.item(),
            "l_obstacle": l_obstacle.item(),
            "l_ceiling": l_ceiling.item(),
            "loss_per_dim": diffusion_losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }
        return loss, loss_dict
