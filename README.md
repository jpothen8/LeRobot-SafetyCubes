# SafetyCubes — Constraint-Aware VLA Manipulation

Train a π0 flow-matching VLA to pick up a blue cube, weave it **low** through a scattered field of red obstacles, and drop it on a goal patch — **without touching any red cube** — by baking collision avoidance into the policy weights during training.

## The idea

Rather than enforcing safety with a post-hoc runtime filter, the collision constraint becomes a differentiable training term:

```
L = L_flow_matching  +  λ · L_safety
```

`L_safety` flows gradients back through differentiable forward kinematics into the VLA, teaching the policy that obstacles exist. At deployment: image → π_θ → action chunk. No filter, no online solve.

See [project_summary.md](./project_summary.md) for the full design rationale, architecture, and hyperparameter guide.

## Repository layout

```
sim/                       MuJoCo environment + data-collection tools
  configs.py               SceneConfig · EnvConfig · ExpertConfig
  scene.py                 MuJoCo scene (SO-101 URDF + procedural geometry)
  env.py                   SafeCubeEnv — reset / step / render
  expert.py                Scripted FSM expert (8-phase pick-carry-place)
  recorder.py              LeRobotDataset writer
  evaluate.py              Eval harness (success / red_contact / clearance)
  randomize.py             Domain randomization knobs
  safety_geometry.py       Differentiable box SDF + FK chain (PyTorch)
  safe_pi0_policy.py       SafePI0Policy — π0 subclass with safety loss
  dagger.py                DAgger collect-retrain loop
  scripts/
    collect_demos.py           Collect expert rollout dataset
    record_rollout.py          Render expert rollout as MP4
    record_policy_rollout.py   Render trained policy rollout as MP4
    train_safe_pi0.py          Training entry point
    dagger.py                  DAgger CLI orchestrator
    view_env.py                Interactive MuJoCo viewer
  assets/so101/            SO-101 URDF + 13 STL meshes
  tests/
    test_safety_geometry.py    Unit tests for SDF / FK primitives

src/lerobot/               LeRobot training framework (one patch applied)
  processor/converters.py  ← privileged.* key preservation added here
```

## Setup

```bash
uv sync --locked
uv pip install "mujoco>=3.2"
```

## Quick start

```bash
# Smoke-test the scene + expert; writes 3 MP4s
uv run python -m sim.scripts.record_rollout --out videos/demo.mp4 --n-episodes 3 --seed 0

# Collect a demo dataset
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-bc \
    --root data/safe_cube_bc \
    --n-episodes 50 --successes-only

# Unit-test safety geometry (run before any training)
uv run pytest sim/tests -q

# Interactive viewer (macOS: use .venv/bin/mjpython instead of uv run)
uv run python -m sim.scripts.view_env
```

## Training pipeline

### 1. Collect expert demos

```bash
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-bc --root data/safe_cube_bc \
    --n-episodes 200 --successes-only
```

### 2. BC warm-start

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

### 3. DAgger with safety loss

```bash
uv run python -m sim.scripts.dagger \
    --repo-id local/safe-cube-dagger --root data/safe_cube_dagger \
    --output-root outputs/safe_pi0_dagger \
    --rounds 3 --episodes-per-round 40 \
    --base-policy outputs/safe_pi0_bc/checkpoints/last/pretrained_model \
    --safety-weight 1.0 --steps 8000 --batch-size 8
```

The central knob is `--safety-weight` (λ). Too small → safety ignored; too large → task collapses. Sweep first. See [sim/README.md](./sim/README.md#end-to-end-runbook-collect--bc-warm-start--dagger) for the full runbook, flag reference, and known caveats.

## Architecture

```
             ┌──────── privileged (sim only) ──────────────────┐
             │                                                  │
image y ──► π_θ (π0 flow-matching VLA) ──► velocity v_θ         │
             │                    │                             │
             │      endpoint est. â₁ = a_τ + (1−τ)·v_θ         │
             │                    │                             │
             │         differentiable FK ──► ee trajectory      │
             │                    │                             │
cube positions ─────────────────►├──► box SDF ──► clearance     │
             │                    │                             │
             │         L_safety = −log σ(α·(clearance − m))    │
             └─────────────────────────────────────────────────┘

At deployment:  image ──► π_θ ──► action.  (No SDF, no cube positions.)
```

See [project_summary.md §5](./project_summary.md) for the flow-matching endpoint-estimate derivation.

## Safety geometry unit tests

Run these **before any training run** — a sign error in `box_sdf` or `FKChain` is cheap here, expensive 40 epochs into a 3B fine-tune.

```bash
uv run pytest sim/tests/test_safety_geometry.py -v
```

## Reference

Cao, Joa, Borrelli — *A Simple Approach to Constraint-Aware Imitation Learning with Application to Autonomous Racing*, arXiv:2503.07737
