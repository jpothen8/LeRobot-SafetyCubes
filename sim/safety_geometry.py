"""Differentiable geometry for the constraint-aware safety loss.

Pure torch — no LeRobot / MuJoCo imports — so it can be unit-tested in isolation
against hand-placed cubes and a known arm config (``project_summary.md`` §8
"Test geometry before the big run"). The policy in ``safe_pi0_policy.py`` imports
these.

Contents:
    box_sdf            signed distance, points → axis-aligned boxes (3D or XY-only)
    sdf_clearance      min box SDF minus the ee bounding-sphere radius
    safety_loss        -log σ(α·(clearance − margin)), optionally (1−t)-weighted
    height_ceiling_loss smooth one-sided penalty that spikes when ee z exceeds a ceiling
    FKChain            differentiable forward kinematics via pytorch_kinematics
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def box_sdf(points: Tensor, centers: Tensor, halves: Tensor, lateral: bool = False) -> Tensor:
    """Signed distance from points to axis-aligned boxes.

    Positive outside (safe), negative inside (penetration). The classic
    Inigo-Quilez box SDF, batched.

    Args:
        points:  ``(B, P, 3)`` query points (e.g. ee positions along a chunk).
        centers: ``(B, M, 3)`` box centers (red-cube world positions).
        halves:  ``(B, M, 3)`` box half-extents.
        lateral: if True, measure distance in the XY plane only — i.e. treat each
            cube as an *infinite vertical prism*. Then lifting the ee straight up
            over a cube no longer increases clearance, so "fly over the top" stops
            counting as obstacle avoidance and the only way out is to go *around*
            in XY. Height is instead constrained by :func:`height_ceiling_loss`.

    Returns:
        ``(B, P, M)`` signed distance of every point to every box.
    """
    p = points.unsqueeze(2)        # (B, P, 1, 3)
    c = centers.unsqueeze(1)       # (B, 1, M, 3)
    h = halves.unsqueeze(1)        # (B, 1, M, 3)
    d = (p - c).abs() - h          # (B, P, M, 3)
    if lateral:
        d = d[..., :2]             # X, Y only → infinite vertical prism
    outside = torch.linalg.vector_norm(torch.clamp(d, min=0.0), dim=-1)  # (B, P, M)
    inside = torch.clamp(d.amax(dim=-1), max=0.0)                        # (B, P, M)
    return outside + inside


def sdf_clearance(
    points: Tensor, centers: Tensor, halves: Tensor, ee_radius: float, lateral: bool = False
) -> Tensor:
    """Min over boxes of the box SDF, minus the ee bounding-sphere radius.

    Returns ``(B, P)`` signed clearance; < 0 means the gripper sphere overlaps a
    cube. The held blue cube is absorbed into ``ee_radius`` (``project_summary``
    §4), so it is not treated as an obstacle. ``lateral`` is forwarded to
    :func:`box_sdf` (XY-only avoidance).
    """
    sdf = box_sdf(points, centers, halves, lateral=lateral)   # (B, P, M)
    return sdf.amin(dim=-1) - ee_radius                       # (B, P)


def safety_loss(
    clearance: Tensor,
    *,
    alpha: float,
    margin: float,
    weight: Tensor | None = None,
) -> Tensor:
    """Negative-log safety likelihood, optionally per-sample weighted.

    ``p_safe = σ(α·(clearance − margin))`` so the per-point loss is
    ``-log p_safe = softplus(-α·(clearance − margin))`` (stable form).

    Args:
        clearance: ``(B, P)`` from :func:`sdf_clearance`.
        alpha: sigmoid sharpness (1/m); ~50 per the summary.
        margin: desired clearance buffer (m); ~0.02.
        weight: optional ``(B,)`` per-sample weight (e.g. ``1 − t``). When given,
            returns a weighted mean over the batch; else a plain mean.

    Returns:
        Scalar loss.
    """
    z = alpha * (clearance - margin)              # (B, P)
    per_point = F.softplus(-z)                    # (B, P)  == -log σ(z)
    per_sample = per_point.mean(dim=1)            # (B,)
    if weight is None:
        return per_sample.mean()
    weight = weight.to(per_sample.dtype)
    return (weight * per_sample).sum() / (weight.sum() + 1e-8)


def height_ceiling_loss(
    ee_z: Tensor,
    ceiling: float,
    *,
    alpha: float,
    weight: Tensor | None = None,
) -> Tensor:
    """Smooth one-sided penalty that spikes when ee height exceeds ``ceiling``.

    ``softplus(α·(ee_z − ceiling))``: ≈ 0 well below the ceiling, then rises
    sharply (≈ linearly with slope α) above it. Fully differentiable, so it can
    be added straight into the flow-matching loss. Mirrors the env's
    ``SceneConfig.ee_height_ceiling`` stay-low rule (red-cube tops sit at ~25 mm;
    going above ~60 mm means the arm is lifting *over* the obstacles instead of
    weaving between them). Larger ``alpha`` ⇒ a sharper spike at the threshold.

    Args:
        ee_z:    ``(B, T)`` end-effector height along the predicted chunk.
        ceiling: height (m) above which the penalty turns on.
        alpha:   spike sharpness (1/m).
        weight:  optional ``(B,)`` per-sample weight (e.g. ``(1 − t)`` times a
            grasp/carry-phase gate). When given, returns a weighted mean over the
            batch; else a plain mean. A zero-weight batch returns ~0 safely.

    Returns:
        Scalar loss.
    """
    per_point = F.softplus(alpha * (ee_z - ceiling))   # (B, T)
    per_sample = per_point.mean(dim=1)                 # (B,)
    if weight is None:
        return per_sample.mean()
    weight = weight.to(per_sample.dtype)
    return (weight * per_sample).sum() / (weight.sum() + 1e-8)


class FKChain:
    """Lazy wrapper around ``pytorch_kinematics`` serial-chain FK.

    Loads a URDF once and maps the policy's arm-joint ordering onto the chain's
    actuated-joint ordering. FK targets ``end_link`` (the SO-101
    ``gripper_frame_link``, where the sim places ``ee_site``), so the returned
    position equals the sim's privileged ``ee_pos`` — provided the robot base
    sits at the world origin (``scene.py`` loads the URDF without an offset, so
    the URDF root frame *is* the world frame).
    """

    def __init__(self, urdf_path: str, end_link: str, arm_joint_names: list[str]) -> None:
        self.urdf_path = urdf_path
        self.end_link = end_link
        self.arm_joint_names = list(arm_joint_names)
        self._chain = None
        self._perm: list[int] | None = None
        self._device = None
        self._dtype = None

    def _ensure(self, device, dtype) -> None:
        if self._chain is not None and self._device == device and self._dtype == dtype:
            return
        try:
            import pytorch_kinematics as pk
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "Differentiable FK needs `pytorch_kinematics`. Install it into the project "
                "venv: `uv pip install pytorch_kinematics`."
            ) from e

        path = Path(self.urdf_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"URDF for FK not found at {path}.")

        chain = pk.build_serial_chain_from_urdf(path.read_bytes(), self.end_link)
        chain = chain.to(dtype=dtype, device=device)

        chain_names = chain.get_joint_parameter_names()
        missing = set(chain_names) - set(self.arm_joint_names)
        if missing:
            raise ValueError(
                f"FK chain expects actuated joints {chain_names} but arm_joint_names "
                f"{self.arm_joint_names} is missing {sorted(missing)}."
            )
        self._perm = [self.arm_joint_names.index(n) for n in chain_names]
        self._chain = chain
        self._device = device
        self._dtype = dtype

    def fk(self, arm_q: Tensor) -> Tensor:
        """Forward kinematics for a batch of arm-joint chunks.

        Args:
            arm_q: ``(B, T, J)`` arm-joint targets in ``arm_joint_names`` order.

        Returns:
            ``(B, T, 3)`` end-effector positions in the world frame.
        """
        self._ensure(arm_q.device, arm_q.dtype)
        b, t, _ = arm_q.shape
        q = arm_q.reshape(b * t, -1)[:, self._perm]            # reorder to chain order
        mat = self._chain.forward_kinematics(q).get_matrix()    # (B*T, 4, 4)
        return mat[:, :3, 3].reshape(b, t, 3)
