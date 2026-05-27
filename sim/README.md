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
uv run python -m sim.scripts.view_env --mjcf sim/assets/so101/so_arm101.xml

# 2. Record 50 demos for the BC warm start
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-bc \
    --root data/safe_cube_bc \
    --mjcf sim/assets/so101/so_arm101.xml \
    --n-episodes 50 --successes-only

# 3. Record a larger noisy set that contains failures (for the safety loss)
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-mixed \
    --root data/safe_cube_mixed \
    --mjcf sim/assets/so101/so_arm101.xml \
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

## DAgger loop sketch

The pieces above are the primitives — the DAgger loop is a small driver:

```python
from sim import SafeCubeEnv, EnvConfig, ExpertConfig
from sim.expert import ScriptedExpert
from sim.recorder import EpisodeRecorder

env = SafeCubeEnv(EnvConfig(...))
expert = ScriptedExpert(env=env, cfg=ExpertConfig())

for j in range(N_DAGGER_ROUNDS):
    alpha = ALPHA0 ** j           # mixing weight: expert-heavy -> policy-heavy
    for ep in range(EPISODES_PER_ROUND):
        obs, info = env.reset(seed=...)
        rec.begin_episode(task=...)
        expert.reset()
        while not done:
            expert_action = expert.act(info)
            policy_action = policy.act(obs)       # your π0 inference
            use_expert = rng.random() < alpha
            action = expert_action if use_expert else policy_action
            # Always label with the expert action (DAgger).
            rec.add(obs, expert_action, info)
            obs, _, terminated, truncated, info = env.step(action)
        rec.save_episode()
    train_one_pass(policy, dataset, L_task + lam * L_safety)
```

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
