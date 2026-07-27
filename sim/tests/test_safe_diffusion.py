"""Unit tests for the constraint-aware Diffusion Policy's math and wiring.

These cover the two things that can silently corrupt the safety loss without
ever raising — the DDPM **endpoint estimate** and the ACTION **un-normalization**
— plus the naming contract LeRobot's factory relies on to resolve
``--policy.type=safe_diffusion``. None of them build the ~50 M-param network, so
the file runs in seconds on CPU:

    uv run pytest sim/tests/test_safe_diffusion.py -q

Why these three:

* The endpoint estimate inverts diffusers' forward process. A wrong ``sqrt``
  placement still produces finite, plausible-looking numbers — FK then penalises
  the wrong region of space and training quietly optimises nothing useful. The
  test checks the inversion against the *real* ``DDPMScheduler.add_noise``.
* ``safe_pi0`` normalizes ACTION with MEAN_STD, ``safe_diffusion`` with MIN_MAX.
  Applying the wrong affine map is likewise silent.
* The factory derives ``SafeDiffusionPolicy`` from ``SafeDiffusionConfig`` by
  string surgery, so a rename breaks CLI resolution at launch time only.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch

_HAS_DIFFUSERS = importlib.util.find_spec("diffusers") is not None
pytestmark = pytest.mark.skipif(_HAS_DIFFUSERS is False, reason="requires the `diffusers` extra")

from lerobot.configs import NormalizationMode  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

from sim.safe_diffusion_policy import SafeDiffusionConfig, SafeDiffusionPolicy  # noqa: E402


def _fake_policy(config: SafeDiffusionConfig, scheduler=None) -> SimpleNamespace:
    """Minimal stand-in exposing just what the tested methods touch.

    Lets us exercise the math unbound, without instantiating the U-Net, the
    ResNet encoders, or a dataset.
    """
    return SimpleNamespace(config=config, diffusion=SimpleNamespace(noise_scheduler=scheduler))


def _scheduler(config: SafeDiffusionConfig):
    from lerobot.policies.diffusion.modeling_diffusion import _make_noise_scheduler

    return _make_noise_scheduler(
        config.noise_scheduler_type,
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        clip_sample=config.clip_sample,
        clip_sample_range=config.clip_sample_range,
        prediction_type=config.prediction_type,
    )


# ---------------------------------------------------------------- endpoint --
def test_endpoint_estimate_inverts_real_add_noise():
    """x̂_0 must recover x_0 exactly when the network predicts the true epsilon."""
    torch.manual_seed(0)
    cfg = SafeDiffusionConfig(clamp_endpoint=False)
    sched = _scheduler(cfg)
    me = _fake_policy(cfg, sched)

    x0 = torch.randn(4, cfg.horizon, 6, dtype=torch.float64)
    eps = torch.randn_like(x0)
    # Span the schedule, including both endpoints.
    timesteps = torch.tensor([0, 10, 50, cfg.num_train_timesteps - 1]).long()

    x_k = sched.add_noise(x0, eps, timesteps)
    est = SafeDiffusionPolicy._endpoint_estimate(me, x_k, eps, timesteps)

    # The 1/sqrt(alpha_bar) factor amplifies float error at high k, so compare
    # in float64 with a tolerance that scales with the schedule.
    assert torch.allclose(est, x0, atol=1e-6, rtol=1e-5)


def test_endpoint_estimate_sample_prediction_is_identity():
    cfg = SafeDiffusionConfig(prediction_type="sample", clamp_endpoint=False)
    me = _fake_policy(cfg, _scheduler(cfg))
    pred = torch.randn(2, cfg.horizon, 6)
    x_k = torch.randn_like(pred)
    est = SafeDiffusionPolicy._endpoint_estimate(me, x_k, pred, torch.tensor([0, 7]).long())
    assert torch.equal(est, pred)


def test_endpoint_clamp_matches_sampler_range():
    """clamp_endpoint bounds x̂_0 to the same range the DDPM sampler clips to."""
    cfg = SafeDiffusionConfig(prediction_type="sample", clamp_endpoint=True, clip_sample_range=1.0)
    me = _fake_policy(cfg, _scheduler(cfg))
    pred = torch.tensor([[[-5.0, 0.25, 5.0]]])
    est = SafeDiffusionPolicy._endpoint_estimate(me, torch.zeros_like(pred), pred, torch.tensor([0]))
    assert torch.allclose(est, torch.tensor([[[-1.0, 0.25, 1.0]]]))


def test_time_weight_is_one_at_data_side_zero_at_noise_side():
    """Mirrors safe_pi0's (1 - t): full weight on reliable low-noise estimates."""
    cfg = SafeDiffusionConfig()
    me = _fake_policy(cfg)
    k = torch.tensor([0, cfg.num_train_timesteps - 1]).long()
    w = SafeDiffusionPolicy._time_weight(me, k, torch.float32)
    assert w[0].item() == pytest.approx(1.0)
    assert w[1].item() == pytest.approx(0.0)

    assert SafeDiffusionPolicy._time_weight(_fake_policy(
        SafeDiffusionConfig(safety_time_weighting=False)), k, torch.float32) is None


# --------------------------------------------------------- un-normalization --
def _stats(action_dim: int = 6) -> dict:
    torch.manual_seed(1)
    lo = torch.randn(action_dim) - 2.0
    hi = lo + torch.rand(action_dim) * 3.0 + 0.5    # strictly > lo
    return {ACTION: {
        "min": lo, "max": hi,
        "mean": torch.randn(action_dim), "std": torch.rand(action_dim) + 0.5,
    }}


def test_min_max_unnormalization_matches_lerobot_inverse():
    """raw = x·scale + offset must equal LeRobot's (x+1)/2·(max-min)+min."""
    cfg = SafeDiffusionConfig()
    assert cfg.normalization_mapping["ACTION"] == NormalizationMode.MIN_MAX, (
        "DiffusionConfig is expected to normalize ACTION with MIN_MAX; if upstream "
        "changed this, the safety loss' FK un-normalization must follow."
    )
    stats = _stats()
    scale, offset = SafeDiffusionPolicy._action_unnormalization(cfg, stats, action_dim=6)

    x = torch.rand(3, 4, 6) * 2.0 - 1.0                     # normalized, in [-1, 1]
    lo, hi = stats[ACTION]["min"], stats[ACTION]["max"]
    expected = (x + 1) / 2 * (hi - lo) + lo
    assert torch.allclose(x * scale + offset, expected, atol=1e-6)


def test_mean_std_unnormalization_matches_lerobot_inverse():
    """The safe_pi0-style path, for when ACTION normalization is overridden."""
    cfg = SafeDiffusionConfig(
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    stats = _stats()
    scale, offset = SafeDiffusionPolicy._action_unnormalization(cfg, stats, action_dim=6)

    x = torch.randn(3, 4, 6)
    expected = x * stats[ACTION]["std"] + stats[ACTION]["mean"]
    assert torch.allclose(x * scale + offset, expected, atol=1e-6)


def test_unnormalization_falls_back_to_identity_without_stats():
    cfg = SafeDiffusionConfig()
    scale, offset = SafeDiffusionPolicy._action_unnormalization(cfg, None, action_dim=6)
    assert torch.allclose(scale, torch.ones(6))
    assert torch.allclose(offset, torch.zeros(6))


# ------------------------------------------------------------------ wiring --
def test_config_registers_and_resolves_to_the_policy_class():
    """LeRobot resolves the policy class from the CONFIG class name by string
    surgery (SafeDiffusionConfig -> SafeDiffusionPolicy, same module). Renaming
    either half breaks `--policy.type=safe_diffusion` only at launch time."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import _get_policy_cls_from_policy_name

    assert "safe_diffusion" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("safe_diffusion") is SafeDiffusionConfig
    assert _get_policy_cls_from_policy_name("safe_diffusion") is SafeDiffusionPolicy


def test_horizon_is_unet_compatible_and_covers_the_action_steps():
    cfg = SafeDiffusionConfig()
    assert cfg.horizon % 2 ** len(cfg.down_dims) == 0        # enforced by __post_init__
    assert cfg.n_action_steps <= cfg.horizon - cfg.n_obs_steps + 1
    # n_obs_steps=1 is load-bearing for cleanup DAgger: a branched episode's first
    # frame has no real history, so the (state -> chunk) relabel is only valid for
    # a purely state-conditioned policy. See CLAUDE.md §4.
    assert cfg.n_obs_steps == 1


def test_safety_knobs_mirror_safe_pi0_defaults():
    """The coefficients must mean the same thing in both classes, so a λ sweep
    run on the cheap policy transfers to π0."""
    from sim.safe_pi0_policy import SafePI0Config

    diff, pi0 = SafeDiffusionConfig(), SafePI0Config()
    for knob in (
        "safety_weight", "obstacle_weight", "sdf_alpha", "sdf_margin", "ee_radius",
        "safety_time_weighting", "lateral_clearance", "ee_height_ceiling",
        "ceiling_alpha", "ceiling_weight", "ceiling_grasped_only",
        "urdf_path", "ee_link_name", "arm_joint_names",
    ):
        assert getattr(diff, knob) == getattr(pi0, knob), f"{knob} diverged from safe_pi0"
