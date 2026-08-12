"""Differentiable geometry for the constraint-aware safety loss.

Pure torch — no LeRobot / MuJoCo imports — so it can be unit-tested in isolation
against hand-placed cubes and a known arm config (``project_summary.md`` §8
"Test geometry before the big run"). The policy in ``safe_pi0_policy.py`` imports
these.

Contents:
    box_sdf            signed distance, points → axis-aligned boxes (3D or XY-only)
    sdf_clearance      min box SDF minus the ee bounding-sphere radius (single point)
    multi_point_clearance  same, for K spheres per timestep with independent radii

    Two penalty *forms*, selected by the policies' ``safety_form`` config field:
      "softplus" (legacy)          "hinge" (current)
      safety_loss                  clearance_hinge_loss
      height_ceiling_loss          ceiling_hinge_loss

    The softplus forms have infinite support and are large on the expert's own
    actions, so they trade off against imitation everywhere; the hinge forms are
    exactly zero outside the margin. See :func:`hinge` for the measured numbers.
    Both are kept so the before/after is directly comparable.

    hinge              0 / d² / 2d−1 finite-support deficit penalty
    FKChain            differentiable FK via pytorch_kinematics; ``fk`` for the
                       TCP alone, ``fk_links`` for the multi-point query set
"""

from __future__ import annotations

import json
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


def multi_point_clearance(
    points: Tensor, centers: Tensor, halves: Tensor, radii: Tensor, lateral: bool = False
) -> Tensor:
    """Signed clearance for several collision spheres per timestep.

    The multi-point analogue of :func:`sdf_clearance`: instead of one sphere at
    the TCP it takes ``K`` spheres (gripper frame, both jaws, the held cube) with
    independent radii, matching the geoms the env actually collision-checks.

    Args:
        points:  ``(B, T, K, 3)`` world-frame sphere centers.
        centers: ``(B, M, 3)`` red-cube centers.
        halves:  ``(B, M, 3)`` red-cube half-extents.
        radii:   ``(K,)`` sphere radii (m).
        lateral: XY-only distance, forwarded to :func:`box_sdf`.

    Returns:
        ``(B, T, K)`` signed clearance; < 0 means that sphere overlaps a cube.
    """
    b, t, k, _ = points.shape
    sdf = box_sdf(points.reshape(b, t * k, 3), centers, halves, lateral=lateral)  # (B, T*K, M)
    nearest = sdf.amin(dim=-1).reshape(b, t, k)                                   # (B, T, K)
    return nearest - radii.to(nearest.device, nearest.dtype).reshape(1, 1, k)


# The collision query set: a tight sphere covering of the arm's collision meshes,
# regenerated by ``sim/scripts/calibrate_safety_loss.py --derive-spheres`` and
# verified against ``mj_geomDistance``.
#
# Why not one sphere at the TCP, as the softplus form used: the env fires
# `red_contact` on any arm/finger geom or the held blue cube (env.py:481-491),
# while `gripper_frame_link` is a *frame* ~9.8 cm beyond `gripper_link` carrying
# no geometry at all. The bodies that actually collide are covered here.
#
# There is deliberately **no sphere for the held blue cube**. An earlier revision
# put one at the TCP frame origin, which was wrong twice over: it was not gated on
# `grasped`, so it swept a phantom 12.7 mm ball through the field whenever the
# gripper was empty, and the cube does not sit at that origin anyway (measured
# rigid offset [-0.0045, -0.0123, 0.0018] in the TCP frame, 0.7 mm spread). It is
# also the wrong thing to model: the carry corridor is planned to keep the *cube*
# clear in XY, so residual `red_contact` is the wider gripper finger, not the cube
# -- and the held cube's height is already the ceiling term's whole job, via
# `ee_to_cube_z_offset`.
#
# **Containment is the contract.** Each sphere is the bounding sphere of a cluster
# of actual mesh vertices, so the union of spheres *contains* the arm. A sphere
# that is clear of a red cube therefore proves the real geom is clear too, which
# is what lets ``hinge_margin`` sit at ~5 mm instead of buying a standoff: going
# *through* the sphere set is already a conservative stand-in for contact.
#
# Do NOT go back to sizing these from ``geom_size``. For a mesh that field is
# ``max|v|`` about the *mesh origin* (not the bounding-box half-extent), so it is
# inflated by however far the mesh sits off its own origin -- and taking the max
# of the two short axes to size a sphere is >2x too fat for these plate-like
# parts. The two compound: the old 16-sphere set read up to 19 mm over-conservative
# on `wrist_link` from one direction while still reading thin from another, a
# spread no scalar shrink can remove.
_SPHERES = json.loads((Path(__file__).parent / "assets" / "collision_spheres.json").read_text())
DEFAULT_COLLISION_LINKS: list[str] = _SPHERES["links"]
DEFAULT_COLLISION_OFFSETS: list[list[float]] = _SPHERES["offsets"]
DEFAULT_COLLISION_RADII: list[float] = _SPHERES["radii"]


def collision_index(links: list[str]) -> tuple[list[str], Tensor]:
    """Collapse per-sphere link names to a unique list plus an index into it.

    FK is evaluated once per *link*; several spheres ride on the same link, so
    this keeps the FK call small and the gather cheap.
    """
    unique = list(dict.fromkeys(links))
    return unique, torch.tensor([unique.index(n) for n in links], dtype=torch.long)


def collision_sphere_clearance(
    fk: "FKChain",
    q_raw: Tensor,
    centers: Tensor,
    halves: Tensor,
    *,
    links: list[str],
    link_index: Tensor,
    offsets: Tensor,
    radii: Tensor,
) -> Tensor:
    """FK the full arm pose, place the collision spheres, return their clearance.

    Args:
        q_raw: ``(B, T, 6)`` **un-normalized** joint targets — the raw action,
            gripper dim included (the moving jaw's pose depends on it).
        centers/halves: ``(B, M, 3)`` red-cube boxes.
        links: the unique link names, in the order ``link_index`` refers to.

    Returns:
        ``(B, T, K)`` signed clearance per sphere.

    Uses **3D** distance, not the XY-only ``lateral`` mode of the softplus form.
    Lateral distance treats each cube as an infinite vertical prism so that
    flying over one does not read as avoidance — but it also penalises the arm
    for passing high above a cube during the reach and retreat, which the expert
    does constantly (measured: `gripper_link` is laterally inside a red prism on
    4.7% of expert frames). Keeping obstacles in 3D and leaving "don't fly over"
    to the grasp-gated ceiling term separates the two concerns cleanly.
    """
    mats = fk.fk_link_transforms(q_raw, links)
    pts = sphere_centers(mats, link_index, offsets)
    return multi_point_clearance(pts, centers, halves, radii, lateral=False)


def sphere_centers(mats: Tensor, link_index: Tensor, offsets: Tensor) -> Tensor:
    """Place body-frame sphere offsets into the world using FK transforms.

    The arm's collision geoms are long meshes, so the query set is a handful of
    spheres spread along each geom's axis rather than one sphere per link. Each
    sphere is defined once, offline, as ``(link, offset_in_body_frame, radius)``
    by ``sim/scripts/calibrate_safety_loss.py --derive-spheres``; this maps those
    constants through FK every training step.

    Args:
        mats:       ``(B, T, L, 4, 4)`` from :meth:`FKChain.fk_link_transforms`.
        link_index: ``(K,)`` long tensor — which of the ``L`` links each sphere
            rides on.
        offsets:    ``(K, 3)`` offsets in that link's body frame (m).

    Returns:
        ``(B, T, K, 3)`` world-frame sphere centers.
    """
    m = mats.index_select(dim=2, index=link_index.to(mats.device))     # (B, T, K, 4, 4)
    o = offsets.to(mats.device, mats.dtype)                            # (K, 3)
    rotated = torch.einsum("btkij,kj->btki", m[..., :3, :3], o)
    return rotated + m[..., :3, 3]


def hinge(d: Tensor) -> Tensor:
    """Finite-support penalty on a normalized violation deficit.

    ``0`` for ``d <= 0``, ``d²`` on ``(0, 1]``, ``2d - 1`` beyond — C¹ at both
    knots, exactly zero outside the margin, and with a gradient bounded by 2
    (so, in physical units, ``2/margin``).

    This is the shape the ``softplus`` forms lack. ``softplus(-α(c - margin))``
    has *infinite support*: with ``α=50, margin=0.02`` the clearance must exceed
    **0.112 m** before the penalty drops below 0.01, which is unreachable in a
    0.30 m field of 8 cubes. That makes the old term a constant repulsive field
    rather than a violation detector — it is large on the expert's own actions
    (measured ``L_obstacle`` 0.42, ``L_ceiling`` 0.61 vs a converged flow loss of
    0.0087) and therefore fights imitation everywhere instead of only near a
    real violation.
    """
    d = d.clamp(min=0.0)
    return torch.where(d <= 1.0, d * d, 2.0 * d - 1.0)


def _weighted_mean(per_sample: Tensor, weight: Tensor | None) -> Tensor:
    if weight is None:
        return per_sample.mean()
    weight = weight.to(per_sample.dtype)
    return (weight * per_sample).sum() / (weight.sum() + 1e-8)


def clearance_hinge_loss(
    clearance: Tensor, *, margin: float, weight: Tensor | None = None
) -> Tensor:
    """Finite-support obstacle penalty: 0 when clear, 1.0 at contact.

    Args:
        clearance: ``(B, T)`` or ``(B, T, K)`` signed clearance (m).
        margin:    penalty turns on below this clearance (m); also sets the scale,
            so the term is dimensionless and needs no ``alpha``.
        weight:    optional ``(B,)`` per-sample weight (e.g. ``1 - t``).

    Returns:
        Scalar loss. Exactly 0 whenever every point is at least ``margin`` clear.
    """
    # Reduce over SPHERES by the closest one, never by averaging.
    #
    # Safety is a property of the minimum clearance: one sphere buried in a cube
    # is a collision no matter how clear the other 91 are. Averaging over K
    # divides any real violation by the sphere count -- with K=92 a full contact
    # (hinge = 1.0) contributes 1/(T*K) = 1/5152 to the sample loss, which is
    # numerically zero. Measured consequence of getting this wrong: `l_obstacle`
    # read exactly 0.000000 on *every* training batch of a converged policy, so
    # lambda had no gradient at all and the whole sweep was inert.
    #
    # It also makes the loss perversely sensitive to the query-set size -- going
    # from 16 to 91 spheres silently scaled the penalty down ~6x. Under the min
    # reduction, adding spheres can only ever make the estimate tighter.
    if clearance.dim() >= 3:
        clearance = clearance.amin(dim=-1)                # (B, T, K) -> (B, T)
    per_step = hinge((margin - clearance) / margin)
    per_sample = per_step.flatten(start_dim=1).mean(dim=1)
    return _weighted_mean(per_sample, weight)


def held_cube_clearance(
    tcp_mats: Tensor,
    centers: Tensor,
    halves: Tensor,
    *,
    offset: Tensor,
    radius: float,
    grasped: Tensor | None,
) -> Tensor:
    """Clearance of the **carried blue cube** to the red cubes. ``(B, T, 1)``.

    Kept separate from the arm's sphere covering for two reasons the earlier
    single-TCP-sphere version got wrong:

    * **It only exists while grasped.** With an empty gripper there is no cube,
      and a sphere parked at the TCP is a phantom obstacle swept through the
      field. Ungrasped samples are returned as ``+inf`` clearance (silent).
    * **It is not at the TCP frame origin.** The grip is a rigid kinematic lock,
      so the cube sits at a *constant* pose in the TCP frame -- measured
      ``[-0.0045, -0.0123, 0.0018]`` with 0.7 mm spread over 15 k grasped expert
      frames. (Measure that with FK of the *observed* joints; FK of the action
      carries ~24 mm of control lag and reports a spurious 21 mm offset.)

    Args:
        tcp_mats: ``(B, T, 4, 4)`` world transforms of the TCP frame.
        offset:   ``(3,)`` cube centre in the TCP frame.
        grasped:  ``(B,)`` 0/1; ``None`` treats every sample as grasped.
    """
    off = offset.to(tcp_mats.device, tcp_mats.dtype)
    world = torch.einsum("btij,j->bti", tcp_mats[..., :3, :3], off) + tcp_mats[..., :3, 3]
    sdf = box_sdf(world, centers, halves)              # (B, T, M)
    clear = sdf.amin(dim=-1, keepdim=True) - radius    # (B, T, 1)
    if grasped is not None:
        g = grasped.to(clear.device, clear.dtype).reshape(-1, 1, 1)
        clear = clear + (1.0 - g) * 1e3                # ungrasped -> silent
    return clear


def ceiling_hinge_loss(
    cube_z: Tensor, ceiling: float, *, buffer: float, weight: Tensor | None = None
) -> Tensor:
    """Finite-support stay-low penalty: 0 below the band, 1.0 at the ceiling.

    ``cube_z`` must be the **held cube's** height, not the TCP's. The env checks
    the cube (``env.py:493-503``) and its own comment warns that the TCP "sits
    ~10 mm above the cube center and would false-fire at normal carry heights" —
    which is exactly what the old ``height_ceiling_loss`` did by feeding it the
    FK gripper-frame z against the same threshold.

    Rises across ``[ceiling - buffer, ceiling]``, so it is silent over the whole
    expert carry band and only speaks in the last few mm before a violation the
    env would actually record.
    """
    d = (cube_z - (ceiling - buffer)) / buffer
    per_sample = hinge(d).flatten(start_dim=1).mean(dim=1)
    return _weighted_mean(per_sample, weight)


class FKChain:
    """Lazy wrapper around ``pytorch_kinematics`` FK.

    Loads a URDF once and maps the policy's arm-joint ordering onto the chain's
    actuated-joint ordering. :meth:`fk` targets ``end_link`` (the SO-101
    ``gripper_frame_link``, where the sim places ``ee_site``), so the returned
    position equals the sim's privileged ``ee_pos`` — provided the robot base
    sits at the world origin (``scene.py`` loads the URDF without an offset, so
    the URDF root frame *is* the world frame).

    :meth:`fk_links` is the multi-point path used by the hinge safety form. The
    env fires ``red_contact`` on *any* arm/finger geom or the held blue cube
    touching a red cube (``env.py:481-491``), so a single sphere at the TCP is
    the wrong query set — see :func:`clearance_hinge_loss`. It builds a
    **non-serial** chain (one ``forward_kinematics`` call returns every frame)
    and needs the gripper joint too, hence the full 6-DoF action rather than
    :meth:`fk`'s 5 arm joints.
    """

    def __init__(
        self,
        urdf_path: str,
        end_link: str,
        arm_joint_names: list[str],
        *,
        gripper_joint_name: str = "gripper",
    ) -> None:
        self.urdf_path = urdf_path
        self.end_link = end_link
        self.arm_joint_names = list(arm_joint_names)
        self.gripper_joint_name = gripper_joint_name
        self._chain = None
        self._perm: list[int] | None = None
        self._device = None
        self._dtype = None
        # Separate lazy cache for the non-serial (all-frames) chain.
        self._full = None
        self._full_perm: list[int] | None = None
        self._full_device = None
        self._full_dtype = None

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

    # ----- multi-point path (hinge safety form) ------------------------------
    def _ensure_full(self, device, dtype) -> None:
        """Build/cache the non-serial chain whose FK returns *every* frame."""
        if self._full is not None and self._full_device == device and self._full_dtype == dtype:
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

        chain = pk.build_chain_from_urdf(path.read_bytes()).to(dtype=dtype, device=device)

        # The action is `arm_joint_names + [gripper_joint_name]`; the chain wants
        # its own order. Build the permutation once.
        action_order = [*self.arm_joint_names, self.gripper_joint_name]
        chain_names = chain.get_joint_parameter_names()
        missing = set(chain_names) - set(action_order)
        if missing:
            raise ValueError(
                f"Multi-link FK chain expects actuated joints {chain_names} but the action "
                f"ordering {action_order} is missing {sorted(missing)}. The gripper joint is "
                "required because the moving jaw's pose depends on it."
            )
        self._full_perm = [action_order.index(n) for n in chain_names]
        self._full = chain
        self._full_device = device
        self._full_dtype = dtype

    def frame_names(self) -> list[str]:
        """Every frame the URDF defines, in ``fk_links`` name space."""
        self._ensure_full(torch.device("cpu"), torch.float32)
        return list(self._full.forward_kinematics(
            torch.zeros(1, len(self._full.get_joint_parameter_names()))
        ).keys())

    def fk_link_transforms(self, q: Tensor, link_names: list[str]) -> Tensor:
        """Full 4×4 world transforms for several links at once.

        Rotations are needed because the collision spheres sit at *offsets in the
        body frame* (the SO-101 gripper geoms are elongated meshes — one is 14 cm
        long — so a sphere at the link origin is a hopeless fit; see
        :func:`sphere_centers`).

        Args:
            q: ``(B, T, J+1)`` full joint targets ordered
                ``arm_joint_names + [gripper_joint_name]`` — the raw 6-dim action.
            link_names: URDF frames to return, in the requested order.

        Returns:
            ``(B, T, L, 4, 4)`` world transforms, ``L == len(link_names)``.
        """
        self._ensure_full(q.device, q.dtype)
        b, t, _ = q.shape
        flat = q.reshape(b * t, -1)[:, self._full_perm]
        frames = self._full.forward_kinematics(flat)      # dict[str, Transform3d]
        unknown = [n for n in link_names if n not in frames]
        if unknown:
            raise ValueError(
                f"Unknown FK link(s) {unknown}. Available frames: {sorted(frames)}."
            )
        mats = [frames[n].get_matrix() for n in link_names]            # L x (B*T, 4, 4)
        return torch.stack(mats, dim=1).reshape(b, t, len(link_names), 4, 4)

    def fk_links(self, q: Tensor, link_names: list[str]) -> Tensor:
        """World-frame *origins* of several links. See :meth:`fk_link_transforms`.

        Returns:
            ``(B, T, K, 3)`` positions, ``K == len(link_names)``.
        """
        return self.fk_link_transforms(q, link_names)[..., :3, 3]
