This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).

---

# Project: `sim/` — Constraint-Aware VLA Sim (SO-101 + MuJoCo)

A research-style sub-project lives at **`sim/`** (top of repo, *not* inside `src/lerobot/`). It is the data-collection half of the project described in [`project_summary.md`](./project_summary.md): train a π0 flow-matching VLA to pick a blue cube, weave it low through a field of red cubes, and drop it on a goal patch — with collision avoidance baked into the policy weights, not enforced post-hoc.

The sim is the source of (image, state, action, privileged.*) episodes that the future `SafePI0Policy` and DAgger driver will consume. **None of the policy / training code is in this repo yet.**

## High-level layout (`sim/`)

```
sim/
├── configs.py        SceneConfig · EnvConfig · ExpertConfig (dataclasses)
├── scene.py          MuJoCo scene composition via MjSpec (URDF + procedural)
├── env.py            SafeCubeEnv — reset / step / render, privileged info
├── expert.py         Scripted FSM expert (APPROACH→DESCEND→CLOSE→LIFT→CARRY→DESCEND2→OPEN→DONE)
├── recorder.py       LeRobotDataset writer (policy-visible + privileged.* keys)
├── evaluate.py       Eval harness (success / red_contact / clearance / ceiling)
├── randomize.py      Domain randomization knobs (NOT yet wired into env.reset)
├── scripts/
│   ├── collect_demos.py      CLI → dataset of expert rollouts
│   ├── record_rollout.py     CLI → MP4 (agentview + wrist_cam + composite)
│   └── view_env.py           CLI → interactive MuJoCo viewer (macOS: use mjpython)
└── assets/so101/
    ├── so101_new_calib.urdf  Onshape-derived URDF (checked in)
    └── assets/*.stl          13 STL meshes referenced relatively by the URDF
```

Install: `uv pip install "mujoco>=3.2"` (already in the project venv as `mujoco==3.8.1`).

## What the env exposes

`SafeCubeEnv.reset()` and `.step(action)` return a Gymnasium-flavored tuple. **Observation** (policy-visible): `image` `uint8 (H, W, 3)`, `state` `float32 (6,)` (5 arm joints + gripper). **Privileged info** (loss-only): `cube_positions (N, 3)`, `cube_half_extents (N, 3)`, `blue_cube_pos (3,)`, `goal_pos (3,)`, `ee_pos (3,)`, `grasped (bool)`. **Termination flags** in `info["stats"]`: `red_contact`, `ceiling_violation`, `blue_dropped`, `success`, `dwell`, `min_clearance`, `steps`.

Action is `(n_arm + 1,) = (6,)` — absolute joint position targets (position-controlled actuators), gripper last. Chunked actions `(T, 6)` are accepted by `.step()` for the policy's open-loop chunk inference.

## Scene & layout (current tuning)

- **SO-101 URDF** loaded via `mujoco.MjSpec.from_file(...)` — actuators / cameras / damping are added procedurally on top.
- **Red field**: 12×12 inch (0.305 m) square, centered at `(0.225, 0.0)` on the table. Visualized as a flat black tape border (`zone_edge_*`).
- **10 red cubes**, 1 inch each (`half = 0.0127 m`), reject-sampled with `min_cube_separation = 0.055 m`.
- **Blue cube**: 1 inch, sampled in `x ∈ [0.195, 0.255]`, `y ∈ [0.17, 0.21]` (left side, +y). `blue_safety_radius = 0.085 m` from any red cube.
- **Goal patch**: FIXED at `(0.225, -0.20)` (right side, -y). Drawn as a 3×3 inch blue tape square (`goal_tape_*`). `goal_safety_radius = 0.07 m` from any red cube. Success uses L-inf check `|dx| < goal_size/2 AND |dy| < goal_size/2`.
- **BFS path-connectivity check** in `_find_path` runs at sample time — if no collision-free route from blue→goal exists, the whole layout is rejected and resampled (up to `max_layout_attempts = 100`). The resulting waypoint list is stored on `Layout.carry_waypoints` and consumed directly by the expert's CARRY phase.
- **Cameras**:
  - `agentview`: fixed, behind the arm (`pos = (-0.22, 0, 0.55)`, `lookat = (0.23, 0, 0.02)`, fovy 50°). World +y appears on the LEFT of the rendered image — that's why blue is on +y.
  - `wrist_cam`: attached to `gripper_link`, `pos = (0.05, 0, 0.06)` in local frame, `mode=mjCAMLIGHT_TARGETBODY` targeting `moving_jaw_so101_v1_link` so it auto-reorients with the arm. fovy 70°.
- **Home pose** (critical — URDF zero is "arm pointing straight up"): `home_qpos = (0.0, -1.2, 1.4, 0.3, 0.0)` (pan, lift, elbow, wrist_flex, wrist_roll) + `home_gripper = -0.1` (jaws closed at start). Settle 20 sim steps after seeding qpos+ctrl; `mj_forward` once before that.

## Hard-won MuJoCo / URDF lessons (read before changing the scene)

1. **MuJoCo URDF parser flattens fixed-joint links.** The `base_link` body's geoms end up attached *directly to the world body* (id 0) with no name. `gripper_frame_link` is flattened away entirely. Code that wants to identify "arm parts" must either look up bodies by name (for the surviving links) OR filter unnamed world-body geoms via a hardcoded exclusion list (`_NON_ARM_WORLD_GEOMS` in `scene.py`).

2. **`MjSpec.compile()` silently prunes geoms with `contype=0, conaffinity=0`** (treats them as dead). For visual-only decoration (border tape, etc.) use a unique bit (we use `contype=4, conaffinity=4`) — collisions only fire when bits overlap, so 4∩(1,2) = 0 ⇒ no collisions, but the geom survives compile and renders.

3. **URDFs ship NO joint damping / armature.** Without injecting them, position-controlled SO-101 oscillates wildly. We set `damping=2.0`, `armature=0.01` on every arm joint in `build_scene()`. `MjsJoint.damping` is a length-3 vector (per-axis), so pass `[d, 0, 0]` for a hinge.

4. **Arm self-collision pins joints.** Adjacent URDF link collision geoms slightly overlap → contact reaction torques cancel actuator force → `shoulder_pan` literally cannot move. Fix is a `contype` / `conaffinity` bitmask: arm-link + flattened-base geoms = `(2, 1)`, everything else (floor, table, cubes) = `(1, 1)`. arm-vs-arm masks intersect to 0 = no collision; arm-vs-cube/table = 1 = collides.

5. **`mj_kinematics` is NOT enough to compute the Jacobian.** `mj_jacSite` returns all zeros after `mj_kinematics`. Use `mj_forward` instead — it does the prereq inertia / com pass.

6. **Position controllers can't chase a "moving waypoint."** An expert that computes `target = ee + step` each tick advances the target faster than the controller can track → arm lags forever and never converges. Target *absolute* poses (the actual goal of each phase) instead and let the controller smooth.

7. **The SO-101 gripper physically can't pinch a 1-inch cube.** Minimum jaw spacing at fully closed is ~55 mm vs the 25 mm cube. We solve it with a **magnetic grip** in `env._update_grasp()`: when (a) the gripper *qpos* (NOT ctrl) is below `_grip_qpos_attach_threshold = 0.20` AND (b) the cube is within `_grip_attach_radius = 0.04 m` of the ee_site, the cube is kinematically locked to `ee_pos + (0, 0, -0.012)` each substep. Released when qpos rises above `_grip_qpos_release_threshold = 0.40`. The policy still sees the natural pixel-level cause-and-effect.

8. **Grasp must trigger on qpos, not ctrl.** Triggering on ctrl makes the cube snap to the gripper the instant the CLOSE phase begins, before the jaws have physically moved — looks awful and is unphysical. Trigger on actual `data.qpos[gripper]`.

9. **Once grasped, `blue_z` tracks the gripper.** So `LIFT/CARRY/DESCEND2` targets must be in *absolute* world-frame z, not relative to `blue.z` — otherwise the target ramps with the gripper and the arm flies off (and through the ceiling).

10. **Potential-field navigation has local minima.** Original CARRY used attractive-toward-goal + repulsive-from-reds and routinely deadlocked. Replaced with BFS waypoints from the path-connectivity check — no minima possible.

11. **macOS interactive viewer needs `mjpython`.** `mujoco.viewer.launch_passive` requires the main thread; on Darwin the entry point is `.venv/bin/mjpython`, not `python`. For headless rendering use a separate `mujoco.Renderer` — no thread issue.

12. **Gripper open/closed convention** (verified by probing the moving_jaw geom): **low qpos = closed, high qpos = open**. URDF limits are `[-0.174, 1.745]`. Current config uses `grip_closed = -0.10`, `grip_open = 0.60` (~36% of ROM). Don't open all the way — there's no need and it slows the sequence.

## Expert FSM phases (`sim/expert.py`)

```
APPROACH   pan over the blue cube at z = blue.z + pre_grasp_height (0.07 m)
DESCEND    drop to z = blue.z + descend_clearance (0.004 m); reach_tol 0.018 m
           — contact stops the arm ~8 mm above the cube, gripper at cube edge
CLOSE      ctrl = grip_closed; advance only after jaws actually close
           (qpos < grasp_close_qpos_threshold) AND post_grasp_hold_steps (15 ≈ 0.5s)
LIFT       z = carry_z (0.040 m, absolute, well under the 0.060 m ceiling)
CARRY      iterate through Layout.carry_waypoints in XY at carry_z
DESCEND2   z = pre_release_z (0.028 m); tight XY tolerance (0.012 m) for centering
OPEN       ctrl = grip_open; advance only after jaws actually open
           (qpos > grasp_open_qpos_threshold = 0.45) AND release_settle_steps
DONE       (post-loop hold of grip_open held by the rollout scripts)
```

Each phase targets a Cartesian pose; IK is solved via `env.ik_solve()` (damped LS on an `MjData` copy with `mj_forward` per iteration).

## Tuned constants worth knowing

| Knob | Value | Where | Why |
|---|---|---|---|
| `_KP` (position ctrl) | 75 N·m/rad | `scene.py` | Halved from 150 for smoother motion |
| `_KD` | 4 | `scene.py` | Damped enough to avoid overshoot |
| `_FORCE_LIMIT` | 30 N·m | `scene.py` | Generous for sim (real STS3215 is ~2–3) |
| `_JOINT_DAMPING` | 2.0 | `scene.py` | URDF carries none |
| `_JOINT_ARMATURE` | 0.01 | `scene.py` | Same |
| `carry_z` | 0.040 m | `configs.py` | Cube at red-cube level — forces weaving |
| `ee_height_ceiling` | 0.060 m | `configs.py` | Tight enough that "going over" fires a violation |
| `home_qpos` | (0, -1.2, 1.4, 0.3, 0) | `configs.py` | Arm folded over workspace |
| `home_gripper` | -0.10 | `configs.py` | Jaws closed at episode start |
| `descend_clearance` | 0.004 m | `configs.py` | Combined with reach_tol → gripper edge near cube |
| `descend_reach_tol` | 0.018 m | `configs.py` | Tolerant of contact-stopped descent |
| `descend2_reach_tol` | 0.012 m | `configs.py` | Tighter for final drop centering |
| `post_grasp_hold_steps` | 15 | `configs.py` | ~0.5 s pause after physical closure |
| `wrist_camera_pos` | (0.05, 0, 0.06) | `configs.py` | gripper_link local; up-forward-back |

## CLI usage

```bash
# Smoke-test the scene + expert; writes 3 MP4s (agentview, wrist, composite)
uv run python -m sim.scripts.record_rollout --out videos/demo.mp4 --n-episodes 3 --seed 0

# Collect a demo dataset
uv run python -m sim.scripts.collect_demos --repo-id local/safe-cube-bc \
    --root data/safe_cube_bc --n-episodes 50 --successes-only

# Interactive viewer (macOS — note mjpython, not python)
.venv/bin/mjpython -m sim.scripts.view_env
```

## What's NOT done yet (for the next agent)

- **`SafePI0Policy`** — the LeRobot subclass with `L = L_flow_matching + λ·L_safety` per `project_summary.md` §5. Privileged keys are already in the recorded datasets (`privileged.cube_positions`, `privileged.cube_half_extents`, `privileged.blue_cube_pos`, `privileged.goal_pos`, `privileged.ee_pos`, `privileged.grasped`) — the policy/loss just needs to read them from the batch and ignore them at inference. Per-cube arrays are flattened to 1D; reshape on load.
- **DAgger driver** — sketched in `sim/README.md`. The primitives it needs are already in place (`SafeCubeEnv`, `ScriptedExpert`, `EpisodeRecorder`, `evaluate.evaluate()`).
- **Domain randomization not yet applied.** `sim/randomize.py` defines `apply_dr(model, rng, cfg)` but `env.reset()` doesn't call it. Wire it in after the scene compiles, before the first physics step, when ready.
- **Real-world fine-tune / hardware deployment** — out of scope for the sim.

## Gotchas for future edits

- If you add a new visual-only geom to the world, add its name to `_NON_ARM_WORLD_GEOMS` in `scene.py` so the collision-mask rewrite doesn't flag it as a flattened-base arm geom.
- If you change `grip_open` or `grip_closed`, double-check the four thresholds: `_grip_qpos_attach_threshold` / `_grip_qpos_release_threshold` (env.py), and `grasp_close_qpos_threshold` / `grasp_open_qpos_threshold` (configs.py). They need to sit between the new closed and open values with hysteresis.
- After `expert.done()`, the rollout scripts must send `grip_open` (not `home_gripper`) as the gripper action — otherwise the gripper closes back on the released cube and re-attaches it via the magnetic grip. Already wired in `record_rollout.py` and `collect_demos.py`; replicate the pattern if you write new drivers.
- `success` only fires after `success_dwell_steps = 5` frames of the cube resting in the goal zone *with the gripper released*. Don't exit the env loop the instant `expert.done()` is True — keep stepping for ~1 s with a held-open pose so the dwell window can register.
- The wrist camera is mounted on `gripper_link` (rotates with the gripper's `wrist_roll`). If you ever change the home pose so the wrist is rolled, expect the wrist-cam horizon to tilt accordingly.
- Sample-time path BFS uses `path_clearance_radius = 0.045 m`. If you shrink `min_cube_separation` or pack the field tighter, this BFS will start rejecting layouts and `sample_layout` will raise after 100 attempts.
