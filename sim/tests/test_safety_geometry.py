"""Unit tests for the safety-loss geometry (torch-only — no LeRobot/MuJoCo).

Run before any long fine-tune so a sign error in the SDF or FK isn't discovered
40 epochs into a 3B-param run (``project_summary.md`` §8):

    uv run pytest sim/tests/test_safety_geometry.py -q
"""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from sim.safety_geometry import (
    FKChain,
    box_sdf,
    height_ceiling_loss,
    safety_loss,
    sdf_clearance,
)

_HAS_PK = importlib.util.find_spec("pytorch_kinematics") is not None


def test_box_sdf_outside_inside_surface():
    # One unit-half cube at the origin.
    centers = torch.tensor([[[0.0, 0.0, 0.0]]])      # (1, 1, 3)
    halves = torch.tensor([[[1.0, 1.0, 1.0]]])       # (1, 1, 3)
    points = torch.tensor([[
        [3.0, 0.0, 0.0],     # 2.0 outside along +x
        [0.0, 0.0, 0.0],     # center -> -1.0 (deepest inside)
        [1.0, 0.0, 0.0],     # exactly on the +x face -> 0.0
    ]])                                              # (1, 3, 3)
    sdf = box_sdf(points, centers, halves)[0, :, 0]  # (3,)
    assert torch.allclose(sdf, torch.tensor([2.0, -1.0, 0.0]), atol=1e-6)


def test_box_sdf_diagonal_outside_is_euclidean():
    # A point off a corner: distance is the Euclidean norm of the per-axis excess.
    centers = torch.zeros(1, 1, 3)
    halves = torch.ones(1, 1, 3)
    points = torch.tensor([[[2.0, 2.0, 1.0]]])       # excess (1, 1, 0) -> sqrt(2)
    sdf = box_sdf(points, centers, halves)[0, 0, 0]
    assert sdf.item() == pytest.approx(math.sqrt(2.0), abs=1e-6)


def test_sdf_clearance_takes_nearest_cube_and_subtracts_radius():
    centers = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])  # (1, 2, 3)
    halves = torch.full((1, 2, 3), 0.5)
    points = torch.tensor([[[2.0, 0.0, 0.0]]])       # nearer to cube 0: 2.0-0.5 = 1.5
    cle = sdf_clearance(points, centers, halves, ee_radius=0.3)[0, 0]
    assert cle.item() == pytest.approx(1.5 - 0.3, abs=1e-6)


def test_safety_loss_monotonic_in_clearance():
    # Less clearance -> larger -log p_safe.
    near = torch.tensor([[0.0]])
    far = torch.tensor([[1.0]])
    l_near = safety_loss(near, alpha=50.0, margin=0.02)
    l_far = safety_loss(far, alpha=50.0, margin=0.02)
    assert l_near > l_far
    # Deep inside (negative clearance) is heavily penalised; far away is ~0.
    assert safety_loss(torch.tensor([[-0.1]]), alpha=50.0, margin=0.02) > 1.0
    assert safety_loss(torch.tensor([[1.0]]), alpha=50.0, margin=0.02) < 1e-3


def test_safety_loss_weighting_downweights_high_t():
    # weight = (1 - t): a high-t (noisy) sample should contribute less.
    clearance = torch.tensor([[0.0], [0.0]])          # both equally unsafe
    w_lowt = torch.tensor([1.0, 0.0])                 # sample 0 fully counts, sample 1 ignored
    unweighted = safety_loss(clearance, alpha=50.0, margin=0.02)
    weighted = safety_loss(clearance, alpha=50.0, margin=0.02, weight=w_lowt)
    # Equal per-sample loss, so the weighted mean equals the unweighted here,
    # but a zero-weight sample must not blow up the result.
    assert torch.isfinite(weighted)
    assert weighted == pytest.approx(unweighted.item(), abs=1e-6)


def test_safety_loss_gradient_flows_to_points():
    centers = torch.zeros(1, 1, 3)
    halves = torch.full((1, 1, 3), 0.5)
    points = torch.tensor([[[0.6, 0.0, 0.0]]], requires_grad=True)  # just outside +x face
    cle = sdf_clearance(points, centers, halves, ee_radius=0.05)
    loss = safety_loss(cle, alpha=50.0, margin=0.02)
    loss.backward()
    # Pushing the point further along +x increases clearance -> lowers loss,
    # so the gradient wrt x must be negative (descent moves the ee away).
    assert points.grad is not None
    assert points.grad[0, 0, 0] < 0


# ----- lateral (XY-only) obstacle clearance ---------------------------------
def test_box_sdf_lateral_ignores_height():
    # Cube at origin, half 1. A point straight above (same XY) is OUTSIDE in 3D
    # (going over the top) but INSIDE the infinite vertical prism laterally.
    centers = torch.zeros(1, 1, 3)
    halves = torch.ones(1, 1, 3)
    above = torch.tensor([[[0.0, 0.0, 5.0]]])
    sdf_3d = box_sdf(above, centers, halves)[0, 0, 0]
    sdf_xy = box_sdf(above, centers, halves, lateral=True)[0, 0, 0]
    assert sdf_3d.item() == pytest.approx(4.0, abs=1e-6)      # 5 - 1 above the top
    assert sdf_xy.item() == pytest.approx(-1.0, abs=1e-6)     # inside the XY footprint
    assert sdf_xy < sdf_3d

    # Beside the cube: lateral distance equals the XY excess, height ignored.
    beside = torch.tensor([[[3.0, 0.0, 5.0]]])               # XY excess (2, -1) -> 2.0
    assert box_sdf(beside, centers, halves, lateral=True)[0, 0, 0].item() == pytest.approx(
        2.0, abs=1e-6
    )


def test_sdf_clearance_lateral_penalises_flying_over():
    # Directly over a cube: 3D clearance is safely positive, lateral is negative
    # (so "fly over the top" no longer escapes the obstacle penalty).
    centers = torch.zeros(1, 1, 3)
    halves = torch.full((1, 1, 3), 0.5)
    over = torch.tensor([[[0.0, 0.0, 2.0]]])
    cl_3d = sdf_clearance(over, centers, halves, ee_radius=0.05)[0, 0]
    cl_xy = sdf_clearance(over, centers, halves, ee_radius=0.05, lateral=True)[0, 0]
    assert cl_3d > 0
    assert cl_xy < 0


# ----- height-ceiling (stay-low) penalty ------------------------------------
def test_height_ceiling_loss_spikes_above_ceiling():
    ceiling, alpha = 0.06, 250.0              # default config sharpness
    carry = torch.tensor([[0.04, 0.04]])      # expert carry height -> ~0 penalty
    at = torch.tensor([[0.06, 0.06]])         # right at the ceiling
    over = torch.tensor([[0.10, 0.12]])       # flying over the top -> should spike
    l_carry = height_ceiling_loss(carry, ceiling, alpha=alpha)
    l_at = height_ceiling_loss(at, ceiling, alpha=alpha)
    l_over = height_ceiling_loss(over, ceiling, alpha=alpha)
    assert l_carry < 1e-2                      # ~0 at the expert carry height (40 mm)
    assert l_over > l_at > l_carry             # monotonic, spikes above
    assert l_over.item() > 1.0                 # a real spike, not a nudge


def test_height_ceiling_loss_gradient_pushes_down():
    ceiling = 0.06
    ee_z = torch.tensor([[0.10]], requires_grad=True)   # above the ceiling
    height_ceiling_loss(ee_z, ceiling, alpha=100.0).backward()
    # Loss grows with height, so descent (negative grad step) lowers z.
    assert ee_z.grad is not None and ee_z.grad[0, 0] > 0


def test_height_ceiling_loss_grasp_gate_zeroes_out():
    # weight=0 (not grasped / not carrying) -> no penalty even if sky-high.
    ee_z = torch.tensor([[1.0], [1.0]])       # both far above ceiling
    w_off = torch.tensor([0.0, 0.0])
    l = height_ceiling_loss(ee_z, 0.06, alpha=100.0, weight=w_off)
    assert torch.isfinite(l) and l.item() == pytest.approx(0.0, abs=1e-6)


# ----- FK tests (skip if the dependency / URDF is unavailable) --------------
_URDF = "sim/assets/so101/so101_new_calib.urdf"
_ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@pytest.mark.skipif(not _HAS_PK, reason="pytorch_kinematics not installed")
def test_fk_shapes_and_differentiable():
    chain = FKChain(_URDF, "gripper_frame_link", _ARM)
    arm_q = torch.zeros(2, 5, 5, requires_grad=True)   # (B=2, T=5, J=5)
    ee = chain.fk(arm_q)
    assert ee.shape == (2, 5, 3)
    ee.sum().backward()
    assert arm_q.grad is not None
    assert torch.isfinite(arm_q.grad).all()
