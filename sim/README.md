# `sim/` — Constraint-Aware Cube-Manipulation Simulation

MuJoCo-based simulation pipeline for the project described in
[`project_summary.md`](../project_summary.md): train a VLA to pick up a blue
cube, weave it low through a field of red cubes, and place it on a goal patch
— with the avoidance constraint baked into the policy weights.

This directory is intentionally separate from `src/lerobot/envs/` because it is
project-specific research scaffolding, not a generic LeRobot env (yet).

## Layout

```
sim/
├── configs.py      # SceneConfig, EnvConfig, ExpertConfig
├── scene.py        # MJCF composition: SO-101 + cubes + goal + camera
├── env.py          # SafeCubeEnv: reset / step / render
├── randomize.py    # domain randomization knobs
├── expert.py       # scripted expert (Jacobian IK + potential-field weaving)
├── recorder.py     # writes LeRobotDataset episodes with privileged keys
├── evaluate.py     # rollout harness with success / contact / clearance stats
├── scripts/
│   ├── collect_demos.py   # CLI: generate demo dataset
│   └── view_env.py        # CLI: launch MuJoCo viewer on the scene
└── assets/
    └── README.md          # how to fetch the SO-101 MJCF
```

## Install

The base LeRobot env covers most deps. You additionally need MuJoCo (installed by default into the project venv: `mujoco==3.8.1`). If starting fresh:

```bash
uv pip install "mujoco>=3.2"
```

The SO-101 URDF + STL meshes are committed under `sim/assets/so101/`. The scene loader (`scene.py`) uses `MjSpec.from_file` to ingest the URDF, then procedurally adds:

- a table, floor, lights, fixed-pose camera (`agentview`)
- an `ee_site` at the gripper tip
- N red cubes (free-joint, 2.5 cm), the blue cube (free-joint, 2.2 cm), and the goal patch site
- position-controlled actuators for all 6 joints
- joint damping (2.0) and armature (0.01) applied to all arm joints — the URDF carries neither, so position control oscillates without these

## Quick start

```bash
# 1. View the scene with the scripted expert driving it
uv run python -m sim.scripts.view_env

# 2. Record 50 demos for the BC warm start
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-bc \
    --root data/safe_cube_bc \
    --n-episodes 50 --successes-only

# 3. Record a larger noisy set that contains failures (for the safety loss)
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-mixed \
    --root data/safe_cube_mixed \
    --n-episodes 200
```

The resulting datasets have these features:

| Key                            | Shape / type           | Visible to policy? |
|--------------------------------|------------------------|--------------------|
| `observation.images.agentview` | uint8 `(H, W, 3)`      | ✅ yes |
| `observation.state`            | float32 `(n_arm+1,)`   | ✅ yes |
| `action`                       | float32 `(n_arm+1,)`   | (target) |
| `privileged.cube_positions`    | float32 `(N, 3)`       | ❌ loss only |
| `privileged.cube_half_extents` | float32 `(N, 3)`       | ❌ loss only |
| `privileged.blue_cube_pos`     | float32 `(3,)`         | ❌ loss only |
| `privileged.goal_pos`          | float32 `(3,)`         | ❌ loss only |
| `privileged.ee_pos`            | float32 `(3,)`         | ❌ loss only |
| `privileged.grasped`           | float32 `(1,)`         | ❌ loss only |

When wiring `SafePI0Policy` into LeRobot training, point the data processor at
the policy-visible keys only; the loss reads the `privileged.*` keys directly
from the batch.

## Termination flags

`info["stats"]` after each step (and on episode end) contains:

```python
{
    "steps": int,
    "red_contact":         bool,   # arm or blue cube touched a red cube
    "ceiling_violation":   bool,   # ee z > scene.ee_height_ceiling while grasping
    "blue_dropped":        bool,   # blue cube fell off the table
    "success":             bool,   # blue cube rested in goal zone for K frames
    "dwell":               int,    # consecutive frames inside goal zone
    "min_clearance":       float,  # signed min over (ee-to-red) clearances
    "closest_red_idx":     int,
}
```

## End-to-end runbook (collect → BC warm-start → DAgger)

The policy + training code now exists (`sim/safe_pi0_policy.py`, `sim/dagger.py`,
`sim/scripts/`). It was written without a runnable environment, so **it is
untested** — do step 0 before anything else.

### 0. Setup + validate the geometry (cheap, do it first)

```bash
uv sync --locked --extra all          # base + policy deps
uv pip install pytorch_kinematics      # differentiable FK for the safety loss

# A sign error in the SDF/FK is cheap to catch now, expensive 40 epochs into a
# 3B-param fine-tune. These tests are torch-only (no MuJoCo / VLA needed).
uv run pytest sim/tests -q
```

Then spot-check the FK frame against ground truth on one recorded frame: load a
sample, run `FKChain.fk(state[:5])`, and confirm it matches that frame's
`privileged.ee_pos` (the loss assumes FK's base_link frame == the world frame —
true while the URDF base sits at the origin).

### 1. Collect data

The safety term needs **both** success and failure data (failures are its
negative class). Make a clean set for the BC warm start and a larger mixed set
that contains failures:

```bash
# Clean successes — BC warm start
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-bc --root data/safe_cube_bc \
    --n-episodes 100 --successes-only

# Mixed set (keeps red-contact / drop / timeout failures)
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-mixed --root data/safe_cube_mixed \
    --n-episodes 200
```

### 2. BC warm start (fine-tune π0 with the safety loss on)

```bash
uv run python -m sim.scripts.train_safe_pi0 \
    --policy.type=safe_pi0 \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.safety_weight=1.0 \
    --dataset.repo_id=local/safe-cube-mixed --dataset.root=data/safe_cube_mixed \
    --output_dir=outputs/safe_pi0_bc \
    --batch_size=8 --steps=20000 \
    --policy.gradient_checkpointing=true
```

`--policy.safety_weight` (λ) is the central knob — sweep it. Too small → safety
ignored; too large → the policy collapses to pure avoidance and stops doing the
task. The evolving safety term makes the loss landscape non-stationary, so
**early-stop at peak rollout success, not at min loss.** Checkpoint lands at
`outputs/safe_pi0_bc/checkpoints/last/pretrained_model`.

### 3. DAgger (collect on the policy's own state distribution → retrain)

`sim/scripts/dagger.py` runs the full collect→retrain loop. Round 0 is pure
expert; later rounds load the previous checkpoint, mix in policy actions
(`alpha = alpha0 ** round`), and **always label with the expert** — which is what
drags training onto deployment-distribution states and produces fresh failures.

```bash
uv run python -m sim.scripts.dagger \
    --repo-id local/safe-cube-dagger --root data/safe_cube_dagger \
    --output-root outputs/safe_pi0_dagger \
    --rounds 3 --episodes-per-round 40 \
    --base-policy outputs/safe_pi0_bc/checkpoints/last/pretrained_model \
    --safety-weight 1.0 --steps 8000 --batch-size 8
```

Use `--no-train` to only collect, or `--print-only` to see the constructed
training commands without running them.

### 4. Evaluate

`sim.evaluate.evaluate()` reports success / red-contact / clearance / ceiling
rates. Wrap a checkpoint with `sim.dagger.PolicyRollout` and pass its `.act` as
the `policy` argument:

```python
from sim import SafeCubeEnv, EnvConfig
from sim.dagger import PolicyRollout
from sim.evaluate import evaluate

env = SafeCubeEnv(EnvConfig())
roll = PolicyRollout(
    checkpoint="outputs/safe_pi0_dagger/round_2/checkpoints/last/pretrained_model",
    dataset_repo_id="local/safe-cube-dagger", dataset_root="data/safe_cube_dagger",
)
task = "pick up the blue cube ... place it on the goal patch"
res = evaluate(env=env, policy=lambda obs: roll.act(obs, task), n_episodes=50)
print(res.summary())
```

### 5. Deploy

Identical to a vanilla π0: `image → π0 → action`. No SDF, no cube positions, no
filter — the safety preference lives in the weights. (On real hardware, keep the
thin clearance watchdog from `project_summary.md` §7.4 as insurance.)

## Knobs worth tuning first

| Knob | Where | Effect |
|---|---|---|
| `n_red_cubes` | `SceneConfig` | Obstacle density |
| `ee_height_ceiling` | `SceneConfig` | Whether "stay low" is binding |
| `min_cube_separation` | `SceneConfig` | How tight the gaps are |
| `repulse_radius`, `repulse_gain` | `ExpertConfig` | How aggressively the expert weaves |
| `success_dwell_steps` | `EnvConfig` | Strictness of "placed" detection |

## Gotchas / known tuning needs

- **Expert tuning is not finished.** Smoke-tested behavior: APPROACH reaches the
  pre-grasp pose, DESCEND drops to within ~5 cm but doesn't always close the
  XY gap, so the full pick-place pipeline doesn't yet run to completion in one
  shot. Things to tune next:
  - Tighten DESCEND target height (e.g. `blue_z + 0.005` so the gripper isn't
    fighting the table contact).
  - Widen `reach_tol` in `expert.py:_target_for_phase` for the descent phase.
  - Possibly add a wrist-orientation constraint to the IK so the gripper
    points down (currently position-only).
- **Joint damping / armature** are injected by `scene.py` because the URDF
  ships with neither. If you change to a different SO-101 description that
  already carries them, set `_JOINT_DAMPING = 0` and `_JOINT_ARMATURE = 0`.
- **Actuator force limit is unrealistically high** (`_FORCE_LIMIT = 30 N·m`,
  vs ~2–3 N·m on the real STS3215). Sim-only knob: turn it down if you want
  to mimic real motor torque limits during evaluation.
- **One actuator per arm joint** is assumed in `env._cache_indices()`. If you
  swap to a model with coupled actuators (e.g. tendon transmission), update
  the index cache.
- **No real gripper geometry in the privileged SDF.** The policy spec uses a
  small bounding sphere (`ee_radius ≈ 3 cm`) absorbed at the loss layer. The
  `min_clearance` reported here is point-to-box; subtract `ee_radius` for
  apples-to-apples with the safety loss.
- **Grasp proxy is geometric, not force-based.** `_blue_grasped()` returns
  True iff a gripper-side geom is in contact with the blue cube geom. Robust
  enough for the FSM expert; if it flickers, gate it on a sustained contact
  count.
