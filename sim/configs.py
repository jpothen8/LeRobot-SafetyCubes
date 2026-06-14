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
    n_red_cubes: int = 8
    red_cube_half: float = _CUBE_SIDE / 2     # ≈ 0.0127 m
    blue_cube_half: float = _CUBE_SIDE / 2

    # ── Red obstacle field (12x12 inch square, ~0.30 m on a side) ────────
    # Centered in front of the arm, within easy reach of the SO-101 workspace.
    red_field_center: tuple[float, float] = (0.225, 0.0)
    red_field_size: float = _FIELD_SIDE
    # Minimum center-to-center separation between any two red cubes — the
    # *placement-diversity* knob, but COUPLED to path_clearance_radius: to weave
    # *between* two cubes (not route around the field) the planner needs clearance
    # on both sides, so clearance must be ≤ sep/2. 8 cubes at 0.11/0.055 weave but
    # pack into predictable "hubs" (peak/mean 6.3). Shrinking to 0.08/0.04 keeps
    # weaving (clearance=sep/2 → 0% route-around) and scatters the field (spread
    # 4.0), BUT the finger clip rate climbs hard as clearance tightens: measured
    # 20%→30%→37%→42% at sep/clearance 0.11/0.055 → 0.10/0.05 → 0.09/0.045 →
    # 0.08/0.04. So 0.04 is NOT free; it costs ~42% discarded episodes. (Fewer
    # cubes at the safe 0.11/0.055 is the cheaper variety lever: 6 cubes → spread
    # 3.7, 100% weave, 0% around, only ~7% clips.)
    min_cube_separation: float = 0.09
    # Larger safety margin specifically around the blue cube and goal — they
    # need clear approach/release space, so red cubes stay farther away from
    # them than from each other. The blue margin is sized for the deep side
    # grasp: the open SO-101 jaws straddle the cube and reach ~5-6 cm out, so
    # reds must stay clear or the descending jaw clips a neighbouring cube.
    blue_safety_radius: float = 0.115
    goal_safety_radius: float = 0.07

    # ── Blue cube spawn (FIXED position, LEFT side of arm) ───────────────
    # Fixed (not randomized): the cube starts at the same spot every episode so
    # the pick is repeatable. The env then YAWS it at reset so its vertical
    # faces meet the descending gripper jaws square-on (see
    # SafeCubeEnv._blue_grasp_yaw / _set_blue_spawn_pose). Sits clearly outside
    # the red field's +y edge (~+0.15). Kept as (degenerate) ranges so the
    # layout sampler and path-bounds code are untouched — both endpoints equal,
    # so rng.uniform just returns the fixed value.
    blue_x_range: tuple[float, float] = (0.21, 0.24)
    blue_y_range: tuple[float, float] = (0.175, 0.205)

    # ── Goal patch (FIXED, RIGHT side of arm) ────────────────────────────
    # Sits clearly outside the red field's -y edge (~-0.15), x matches field.
    # Visually rendered as 4 blue tape strips forming a 3x3 inch interior
    # square (see scene.py:_add_goal_tape). Success detection uses an L-inf
    # check against `goal_size/2` so the cube must land inside the square.
    goal_pos: tuple[float, float] = (0.225, -0.20)
    goal_size: float = 3.0 * _INCH    # 3 inch interior side ≈ 0.0762 m

    # ── Path-connectivity check ──────────────────────────────────────────
    # Clearance (m) the planned corridor keeps from every red cube center. Set to
    # sep/2 so the path threads *between* adjacent cubes (clearance > sep/2 makes
    # the BFS route around the whole field — see min_cube_separation). This is also
    # the expert's tracking-margin budget against finger clips: empirically the
    # clip rate is ~2*clearance-sensitive — 0.055→20%, 0.05→30%, 0.045→37%,
    # 0.04→42%. 0.04 (paired with sep 0.08) weaves with max layout variety but
    # discards ~42% of episodes to clips; it is an experimental setting, not a free
    # lunch. Do NOT raise above sep/2 without accepting route-around.
    path_clearance_radius: float = 0.045
    # Grid resolution for the BFS in sample_layout.
    path_grid_res: float = 0.005
    # Number of resample attempts before giving up.
    max_layout_attempts: int = 100

    # ── Table / ceiling ──────────────────────────────────────────────────
    table_z: float = 0.0
    # Stay-low enforcement: ee z must stay below this once the cube is
    # grasped. Set tight enough that "lifting up and over the cubes" fires a
    # ceiling violation — red cube tops are at ~25 mm and clearing one needs the
    # held cube center ≥ ~38 mm, so a 35 mm limit makes going *over* an obstacle
    # a violation and forces weaving *between* them.
    ee_height_ceiling: float = 0.035

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
    # Expressed in the gripper-frame (ee_site) frame, which is rotated 180° about
    # y from the gripper_link body frame — so x and z are negated relative to a
    # gripper_link-frame measurement of the jaw-gap offset.
    grasp_site_offset: tuple[float, float, float] = (-0.03083, -0.00067, -0.0012)

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
    # Height is a *soft* constraint (penalised by the safety loss + the expert
    # actively carries low), NOT a terminal failure: going over the ceiling
    # should be recovered from (drive the arm back down), not end the episode.
    # This also stops alpha-mixing thrash from killing DAgger rollouts early.
    terminate_on_ceiling_violation: bool = False
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
    # top. The grasp site sits ~4-5 mm above the fingertips at the grasp pose,
    # so 0.0110 → fingertips ≈ 0.0064 m (= 0.25 × the 1-inch cube).
    descend_grasp_z: float = 0.01
    descend_reach_tol: float = 0.005    # settle close to the calibrated depth

    # CLOSE phase advances only once the gripper qpos drops below this
    # (jaws actually closed). Min in URDF is -0.174.
    grasp_close_qpos_threshold: float = 0.25
    # DESCEND2 (final hover over goal) target tolerance — tighter than the
    # generic 2 cm so the drop lands close to the goal patch center.
    descend2_reach_tol: float = 0.0075
    # OPEN phase advances only once the gripper qpos has risen above this
    # (jaws actually open enough to release the cube). Kept ≥ the env's grip
    # release threshold (0.55) so the cube is physically released *before* OPEN
    # ends, not during the post-episode settle.
    grasp_open_qpos_threshold: float = 0.55

    # ABSOLUTE cube-center z during LIFT/CARRY. Every phase steers the grasp
    # site (= held cube), so this is the cube height directly. 30 mm is the
    # tested low-weave sweet spot: low enough that the cube bottom (~17 mm) stays
    # under the 25 mm red tops, so it physically cannot pass over a red and must
    # weave *between* them in XY; high enough that the bulky side-grasp gripper
    # (which reaches ~25 mm below the cube) clears the table. Stays under
    # ee_height_ceiling (0.035). The wide gripper needs roomy corridors — see
    # path_clearance_radius (0.075).
    carry_z: float = 0.025
    # ABSOLUTE cube-center z at release: DESCEND2 steers the grasp site (= cube)
    # down to the goal at this height for a gentle drop into the goal square.
    pre_release_z: float = 0.022

    # Waypoint following during CARRY: advance to the next BFS waypoint when
    # the TCP (held cube) is within this XY tolerance of it.
    waypoint_reach_tol: float = 0.025

    # CARRY-phase off-corridor replan threshold (m, XY). If the held cube drifts
    # farther than this from the CLOSEST waypoint of the current carry path, the
    # expert re-runs the BFS planner from the cube's CURRENT position to the goal
    # and follows that fresh corridor. ~0 in pure-expert demos (the expert tracks
    # waypoints within waypoint_reach_tol, so the closest waypoint is always
    # near); fires in DAgger when the policy has carried the cube off the original
    # spawn→goal corridor — so the relabel is a valid collision-free route from
    # where the policy actually left the cube, not a drive back across the field
    # to a stale corridor. Waypoints are spaced ~1.5 cm (axis-aligned stride) up
    # to ~2.1 cm (8-connected diagonal stride 3·√2·grid_res, plus the goal-snap
    # final segment), so a cube sitting perfectly mid-corridor is at most ~half a
    # stride (~1.1 cm) from its closest waypoint — this 4 cm threshold clears that
    # comfortably and only fires on genuine off-corridor drift. Set ≤ 0 to disable.
    replan_offpath_threshold: float = 0.04

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
