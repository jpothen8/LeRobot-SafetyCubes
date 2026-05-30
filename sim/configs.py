"""Dataclass configs for the constraint-aware cube-manipulation sim.

Three configs:
  * SceneConfig  — geometry (arm URDF, cube sizes, layout regions, camera).
  * EnvConfig    — physics + termination + episode length.
  * ExpertConfig — scripted-expert tuning (waypoint speeds, potential-field gains).

All values are deliberate defaults you can override per-run. Distances in meters.

Layout convention (matches the project spec):
    - Camera is BEHIND the arm (small / negative x) looking forward into the
      workspace. From the camera POV, world +y is to the LEFT, -y to the RIGHT.
    - Blue cube spawns on the LEFT side of the arm  (+y).
    - Goal patch is FIXED on the RIGHT side of the arm (-y).
    - Red cubes (6 of them, 1-inch sides) sit in an 8x8 inch square between
      the blue start region and the goal — the policy must weave through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Imperial → metric for the cube/field dimensions called out in the spec.
_INCH = 0.0254
_CUBE_SIDE = 1.0 * _INCH            # 1-inch cubes
_FIELD_SIDE = 12.0 * _INCH           # 8-inch obstacle square


@dataclass
class SceneConfig:
    # Path to the SO-101 robot description (URDF or MJCF — MjSpec auto-detects).
    mjcf_path: str = "sim/assets/so101/so101_new_calib.urdf"

    # ── Cube sizes (1-inch cubes, per spec) ──────────────────────────────
    n_red_cubes: int = 10
    red_cube_half: float = _CUBE_SIDE / 2     # ≈ 0.0127 m
    blue_cube_half: float = _CUBE_SIDE / 2

    # ── Red obstacle field (12x12 inch square, ~0.30 m on a side) ────────
    # Centered in front of the arm, within easy reach of the SO-101 workspace.
    red_field_center: tuple[float, float] = (0.225, 0.0)
    red_field_size: float = _FIELD_SIDE
    # Minimum center-to-center separation between any two red cubes.
    min_cube_separation: float = 0.055
    # Larger safety margin specifically around the blue cube and goal — they
    # need clear approach/release space, so red cubes stay farther away from
    # them than from each other.
    blue_safety_radius: float = 0.085
    goal_safety_radius: float = 0.07

    # ── Blue cube spawn region (LEFT side of arm, outside red field) ─────
    # Sits clearly outside the red field's +y edge (~+0.10), with x shifted
    # 1" closer to the arm to match the field.
    blue_x_range: tuple[float, float] = (0.195, 0.255)
    blue_y_range: tuple[float, float] = (0.17, 0.21)

    # ── Goal patch (FIXED, RIGHT side of arm) ────────────────────────────
    # Sits clearly outside the red field's -y edge (~-0.15), x matches field.
    # Visually rendered as 4 blue tape strips forming a 3x3 inch interior
    # square (see scene.py:_add_goal_tape). Success detection uses an L-inf
    # check against `goal_size/2` so the cube must land inside the square.
    goal_pos: tuple[float, float] = (0.225, -0.20)
    goal_size: float = 3.0 * _INCH    # 3 inch interior side ≈ 0.0762 m

    # ── Path-connectivity check ──────────────────────────────────────────
    # Clearance (m) the ee + held blue cube needs from any red cube center
    # for a candidate layout to be considered "workable."
    path_clearance_radius: float = 0.045
    # Grid resolution for the BFS in sample_layout.
    path_grid_res: float = 0.005
    # Number of resample attempts before giving up.
    max_layout_attempts: int = 100

    # ── Table / ceiling ──────────────────────────────────────────────────
    table_z: float = 0.0
    # Stay-low enforcement: ee z must stay below this once the cube is
    # grasped. Set tight enough that "lifting up and over the cubes" fires a
    # ceiling violation — red cube tops are at ~25 mm, so anything above
    # ~60 mm means the arm is going *over* the obstacles instead of
    # weaving between them.
    ee_height_ceiling: float = 0.060

    # ── Camera + rendering (camera BEHIND arm, looking forward) ──────────
    camera_name: str = "agentview"
    # Camera position: behind the arm base (negative x), elevated enough to
    # see over the home-pose gripper. Looking toward the obstacle field at
    # table height.
    camera_pos: tuple[float, float, float] = (-0.22, 0.0, 0.55)
    camera_lookat: tuple[float, float, float] = (0.23, 0.0, 0.02)
    camera_fovy: float = 50.0
    image_size: tuple[int, int] = (224, 224)  # (H, W)

    # ── Wrist camera (attached to gripper_link, targeting the grasp) ─────
    # No camera in the SO-101 URDF, so we mount our own. Position is in
    # gripper_link's local frame; targetbody mode auto-orients the camera
    # at the moving_jaw so it always shows the grasp area regardless of
    # arm pose.
    wrist_camera_name: str = "wrist_cam"
    # Position is in gripper_link's local frame. Useful axes (home pose):
    #   local +z  = wrist-roll axis (the "wrist barrel" the cam mounts around)
    #   local +x  → world -Y  (operator's right, from the agentview)
    #   local +y  → world +Z  (up / "on top of the wrist")
    # The gripper extends toward -z, so +z is the along-barrel back-offset
    # (smaller = closer to the gripper) and +y is the height above the wrist.
    # Mounted ON TOP of the wrist (radial offset along +y), a little above and
    # pulled in close over the gripper. (Rotated 90° about the wrist-roll axis
    # from the old right-side (0.05, 0, 0.06) mount — this moves the camera,
    # not the wrist joint.)
    wrist_camera_pos: tuple[float, float, float] = (0.0, 0.08, 0.03)
    wrist_camera_fovy: float = 70.0
    # Extra upward tilt of the wrist cam, in degrees, applied on top of the
    # auto-framed orientation (see scene._freeze_wrist_camera_orientation):
    # pitches the optical axis up about the camera's right axis, so it looks
    # further up the gripper's reach. Keeps the view horizontally centered and
    # the gripper vertical; just slides the gripper lower in frame.
    wrist_camera_pitch_up_deg: float = 12.0

    # ── Joint names (must match the URDF) ────────────────────────────────
    arm_joint_names: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    gripper_joint_name: str = "gripper"
    ee_site_name: str = "ee_site"
    # Tool-center-point site at the jaw-gap center (midway between the open
    # finger faces, at the fingertip plane). The URDF's gripper frame sits at
    # the FIXED jaw, so it is offset ~3 cm laterally from where a grasped cube
    # actually goes; the grasp site corrects that so IK + the magnetic grip
    # center the cube between the fingers. Offset is in the gripper-frame body
    # frame, measured from the ee_site (see PR grasp-geometry probe).
    grasp_site_name: str = "grasp_site"
    grasp_site_offset: tuple[float, float, float] = (0.03083, -0.00067, 0.0012)

    # ── Home pose (per joint, radians) ───────────────────────────────────
    # Default folds the arm forward so the gripper hangs over the obstacle
    # field instead of pointing straight up (the URDF zero pose blocks the
    # camera). Order matches arm_joint_names; gripper open.
    home_qpos: tuple[float, ...] = (0.0, -1.2, 1.4, 0.3, -1.5707963267948966)  # wrist_roll +90°
    # Gripper starts CLOSED at reset. The expert opens it during APPROACH,
    # closes it during CLOSE. Matches a real teleop start-of-episode pose.
    home_gripper: float = -0.1

    def resolved_mjcf(self) -> Path:
        return Path(self.mjcf_path).expanduser().resolve()


@dataclass
class EnvConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    fps: int = 30
    sim_substeps: int = 5                  # physics steps per env step
    max_episode_steps: int = 400

    # Termination behavior.
    terminate_on_red_contact: bool = True
    terminate_on_ceiling_violation: bool = True
    success_dwell_steps: int = 5           # frames the blue cube must rest in goal

    # Observation chunking. Most VLA policies act open-loop over a chunk; the env
    # accepts a chunk and steps through it.
    accept_chunked_actions: bool = True

    seed: int | None = None


@dataclass
class ExpertConfig:
    # Cartesian speeds (m / env-step). Used by the potential-field carry phase.
    approach_speed: float = 0.015
    descend_speed: float = 0.008
    carry_speed: float = 0.012

    # Pre-grasp lift above the cube (relative to cube z, used by APPROACH).
    pre_grasp_height: float = 0.07

    # DESCEND drives the jaw-gap center (grasp site) to this ABSOLUTE world z,
    # lowering the open jaws down the SIDES of the cube. Calibrated so the
    # fingertips land at ~0.25x the cube height (≈6.3 mm above the table for a
    # 1-inch cube) — a realistic side grasp rather than perching on the cube
    # top. The grasp site sits ~7 mm above the fingertips at the grasp pose,
    # so 0.0134 → tips ≈ 0.0064 m.
    descend_grasp_z: float = 0.0134
    descend_reach_tol: float = 0.012    # settle close to the target depth

    # CLOSE phase advances only once the gripper qpos drops below this
    # (jaws actually closed). Min in URDF is -0.174.
    grasp_close_qpos_threshold: float = 0.25
    # DESCEND2 (final hover over goal) target tolerance — tighter than the
    # generic 2 cm so the drop lands close to the goal patch center.
    descend2_reach_tol: float = 0.012
    # OPEN phase advances only once the gripper qpos has risen above this
    # (jaws actually open enough to release the cube).
    grasp_open_qpos_threshold: float = 0.45

    # ABSOLUTE world-frame z for the held cube (jaw-gap center) during the
    # lift/carry/release phases. The cube is locked to the grasp site, so these
    # are the cube-center heights directly. Set so the carried cube rides at
    # the same height as the red obstacles (cube center ≈ red-top height),
    # which is what "weave between" actually means — anything higher means the
    # trajectory clears the obstacle field by going over it. The ee_site (fixed
    # jaw) rides above the cube and stays under SceneConfig.ee_height_ceiling.
    carry_z: float = 0.028
    pre_release_z: float = 0.016

    # Waypoint following during CARRY: advance to the next BFS waypoint when
    # the TCP (held cube) is within this XY tolerance of it.
    waypoint_reach_tol: float = 0.025

    # Damped least-squares IK.
    ik_damping: float = 0.05
    ik_max_iters: int = 20
    ik_pos_tol: float = 0.005

    # Gripper command range. SO-101 convention (verified by probing the
    # moving_jaw geom): LOW qpos → jaws closed, HIGH qpos → jaws open.
    # URDF joint limits: [-0.174, 1.745], full ROM ~1.92 rad.
    # grip_open at 0.6 ≈ 36% of ROM open from closed — plenty of clearance
    # for a 1-inch cube to enter the jaws without flailing them wide open.
    grip_open: float = 0.6
    grip_closed: float = -0.1

    grasp_settle_steps: int = 8
    release_settle_steps: int = 5
    # Additional steps to hold the grasp after the jaws physically close,
    # before lifting the cube. ~0.5s at the env's default 30 fps.
    post_grasp_hold_steps: int = 15
