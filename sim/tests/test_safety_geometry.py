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
    DEFAULT_COLLISION_LINKS,
    DEFAULT_COLLISION_OFFSETS,
    DEFAULT_COLLISION_RADII,
    FKChain,
    box_sdf,
    ceiling_hinge_loss,
    clearance_hinge_loss,
    collision_index,
    collision_sphere_clearance,
    height_ceiling_loss,
    held_cube_clearance,
    hinge,
    multi_point_clearance,
    safety_loss,
    sdf_clearance,
    sphere_centers,
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


# ----- hinge form: the finite-support replacement ---------------------------
# The property that matters is the one the softplus form lacks: EXACTLY zero
# once the constraint is satisfied. A penalty that is nonzero on the expert's
# own actions competes with imitation instead of constraining it (measured on
# expert data: L_safety was 314x the converged flow loss; the hinge form is
# 0.07x). These tests pin that down so it cannot silently regress.
def test_hinge_is_exactly_zero_when_satisfied():
    d = torch.tensor([-10.0, -1.0, -1e-6, 0.0])
    assert torch.equal(hinge(d), torch.zeros(4))


def test_hinge_is_one_at_the_threshold_and_linear_beyond():
    assert hinge(torch.tensor([1.0])).item() == pytest.approx(1.0)
    assert hinge(torch.tensor([0.5])).item() == pytest.approx(0.25)
    # Linear tail bounds the gradient, so one deep outlier cannot dominate a batch.
    assert hinge(torch.tensor([3.0])).item() == pytest.approx(5.0)


def test_hinge_is_c1_at_both_knots():
    for knot in (0.0, 1.0):
        lo = torch.tensor([knot - 1e-4], requires_grad=True)
        hi = torch.tensor([knot + 1e-4], requires_grad=True)
        hinge(lo).backward()
        hinge(hi).backward()
        assert lo.grad.item() == pytest.approx(hi.grad.item(), abs=1e-3)


def test_hinge_gradient_is_bounded():
    d = torch.tensor([1.0, 5.0, 100.0, 1e6], requires_grad=True)
    hinge(d).sum().backward()
    assert torch.all(d.grad <= 2.0 + 1e-6)


def test_clearance_hinge_zero_outside_margin_one_at_contact():
    margin = 0.02
    clear = torch.tensor([[[margin, margin + 0.1, 1.0]]])
    assert clearance_hinge_loss(clear, margin=margin).item() == pytest.approx(0.0)
    assert clearance_hinge_loss(torch.zeros(1, 1, 1), margin=margin).item() == pytest.approx(1.0)


def test_ceiling_hinge_silent_across_the_expert_carry_band():
    """Expert carries the cube at ~22 mm (p95 27 mm); the ceiling is 35 mm.

    The old softplus form scored 0.585 here because it was handed the TCP z,
    which sits 10 mm higher (env.py:493-503 checks the cube). The hinge form on
    the cube must be silent right up to the buffer band.
    """
    # min / mean / p95 of the measured held-cube height, and the band edge itself.
    band = ceiling_hinge_loss(
        torch.tensor([[0.0082, 0.0222, 0.0268, 0.030]]), 0.035, buffer=0.005
    )
    assert band.item() == pytest.approx(0.0)
    at_ceiling = ceiling_hinge_loss(torch.tensor([[0.035]]), 0.035, buffer=0.005)
    assert at_ceiling.item() == pytest.approx(1.0)
    # The expert's tallest carry frame (30.3 mm) is 0.3 mm into the band, so it
    # earns a token penalty rather than nothing -- the band is doing its job.
    tallest = ceiling_hinge_loss(torch.tensor([[0.0303]]), 0.035, buffer=0.005)
    assert 0.0 < tallest.item() < 0.01


def test_multi_point_clearance_matches_single_point_sdf():
    pts = torch.tensor([[[[0.10, 0.0, 0.0], [0.0, 0.25, 0.0]]]])   # (1, 1, 2, 3)
    centers = torch.tensor([[[0.0, 0.0, 0.0]]])
    halves = torch.full((1, 1, 3), 0.05)
    radii = torch.tensor([0.01, 0.02])
    got = multi_point_clearance(pts, centers, halves, radii)
    assert got.shape == (1, 1, 2)
    assert got[0, 0, 0].item() == pytest.approx(0.10 - 0.05 - 0.01)
    assert got[0, 0, 1].item() == pytest.approx(0.25 - 0.05 - 0.02)


def test_sphere_centers_applies_rotation_then_translation():
    mats = torch.eye(4).reshape(1, 1, 1, 4, 4).clone()
    mats[0, 0, 0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    # 90 deg about z: a +x offset must land on +y.
    mats[0, 0, 0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    got = sphere_centers(mats, torch.tensor([0]), torch.tensor([[0.5, 0.0, 0.0]]))
    assert torch.allclose(got[0, 0, 0], torch.tensor([1.0, 2.5, 3.0]), atol=1e-6)


def test_collision_index_dedups_and_maps_back():
    links, idx = collision_index(["a", "b", "a", "c", "b"])
    assert links == ["a", "b", "c"]
    assert [links[i] for i in idx.tolist()] == ["a", "b", "a", "c", "b"]


def test_default_sphere_set_is_self_consistent():
    assert len(DEFAULT_COLLISION_LINKS) == len(DEFAULT_COLLISION_OFFSETS)
    assert len(DEFAULT_COLLISION_LINKS) == len(DEFAULT_COLLISION_RADII)
    assert all(len(o) == 3 for o in DEFAULT_COLLISION_OFFSETS)
    assert all(r > 0 for r in DEFAULT_COLLISION_RADII)
    # Only the GRIPPER ENDS, and only bodies carrying real collision geoms.
    # `wrist_link` is excluded on purpose: every one of its spheres sits >=97 mm
    # from the TCP and it is the binding body on 2.7% of poses, so it cannot
    # realistically reach a 25 mm cube. `gripper_frame_link` is excluded too --
    # it is a bare frame with no geometry, and the held cube it used to stand in
    # for is handled by `held_cube_clearance` (grasp-gated, correct offset).
    assert set(DEFAULT_COLLISION_LINKS) == {"gripper_link", "moving_jaw_so101_v1_link"}
    # Radii must stay small enough to actually follow the mesh: the covering is
    # what lets hinge_margin be ~5 mm instead of buying standoff. The original
    # bounding-box derivation produced 32.6 mm spheres on a 19.6 mm-thick plate.
    assert max(DEFAULT_COLLISION_RADII) < 0.02


@pytest.mark.skipif(not _HAS_PK, reason="requires pytorch_kinematics")
def test_collision_spheres_are_differentiable_wrt_the_action():
    """The whole point of the loss is that gradients reach the network.

    The multi-sphere path adds FK rotations and a gather; if either detached,
    training would silently optimise nothing.
    """
    fk = FKChain(_URDF, "gripper_frame_link", _ARM)
    links, idx = collision_index(DEFAULT_COLLISION_LINKS)
    q = torch.zeros(2, 3, 6, requires_grad=True)
    centers = torch.tensor([[[0.25, 0.0, 0.02]]]).expand(2, 1, 3).contiguous()
    halves = torch.full((2, 1, 3), 0.0127)

    clear = collision_sphere_clearance(
        fk, q, centers, halves,
        links=links, link_index=idx,
        offsets=torch.tensor(DEFAULT_COLLISION_OFFSETS),
        radii=torch.tensor(DEFAULT_COLLISION_RADII),
    )
    assert clear.shape == (2, 3, len(DEFAULT_COLLISION_LINKS))

    clearance_hinge_loss(clear, margin=0.20).backward()   # wide margin so it fires
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()
    assert q.grad.abs().sum() > 0


def test_held_cube_sphere_is_silent_when_not_grasped():
    """An empty gripper carries no cube, so the sphere must not exist.

    The earlier revision parked a 12.7 mm ball at the TCP unconditionally, which
    swept a phantom obstacle through the field on every reach and retreat.
    """
    mats = torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 3, 1, 1)
    centers = torch.zeros(2, 1, 3)                       # cube at the TCP
    halves = torch.full((2, 1, 3), 0.0127)
    off = torch.zeros(3)

    grasped = torch.tensor([1.0, 0.0])
    clear = held_cube_clearance(mats, centers, halves, offset=off,
                                radius=0.0127, grasped=grasped)
    assert clear.shape == (2, 3, 1)
    assert clear[0].max() < 0                            # grasped -> deep overlap
    assert clear[1].min() > 100                          # not grasped -> silent
    assert clearance_hinge_loss(clear[1:], margin=0.005).item() == 0.0


def test_held_cube_offset_rides_the_tcp_rotation():
    """The offset is in the TCP frame, so it must rotate with the gripper."""
    mats = torch.eye(4).reshape(1, 1, 4, 4).clone()
    mats[0, 0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0],     # +90 deg about z
                                       [1.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0]])
    centers = torch.zeros(1, 1, 3)
    halves = torch.full((1, 1, 3), 0.001)
    off = torch.tensor([0.05, 0.0, 0.0])                 # -> world (0, 0.05, 0)
    clear = held_cube_clearance(mats, centers, halves, offset=off,
                                radius=0.0, grasped=None)
    assert clear.item() == pytest.approx(0.049, abs=1e-3)


def test_obstacle_loss_reduces_over_spheres_by_the_closest_not_the_mean():
    """One sphere in contact is a collision regardless of how clear the rest are.

    Averaging over K divides a real violation by the sphere count; at K=92 that
    made `l_obstacle` read exactly 0.0 on every training batch, so lambda had no
    gradient. The loss must therefore be invariant to padding the query set with
    far-away spheres.
    """
    margin = 0.005
    one_hit = torch.tensor([[[0.0]]])                       # (1, 1, 1) at contact
    padded = torch.cat([one_hit, torch.full((1, 1, 91), 1.0)], dim=-1)
    assert clearance_hinge_loss(padded, margin=margin).item() == pytest.approx(
        clearance_hinge_loss(one_hit, margin=margin).item(), abs=1e-9)
    # ...and equals the single-point value, i.e. a true contact scores 1.0.
    assert clearance_hinge_loss(padded, margin=margin).item() == pytest.approx(1.0, abs=1e-6)


def test_obstacle_loss_still_averages_over_TIME():
    """Time is averaged: a chunk violating on more steps is worse."""
    margin = 0.005
    half = torch.cat([torch.zeros(1, 2, 1), torch.full((1, 2, 1), 1.0)], dim=1)  # 2 of 4 steps
    assert clearance_hinge_loss(half, margin=margin).item() == pytest.approx(0.5, abs=1e-6)


def test_obstacle_loss_accepts_the_2d_single_point_shape():
    """(B, T) input (the legacy single-sphere path) must still work."""
    assert clearance_hinge_loss(torch.zeros(1, 3), margin=0.005).item() == pytest.approx(1.0)
