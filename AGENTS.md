# AGENTS.md — orientation for coding agents

This repo is a fork of LeRobot stripped down to one research project:
**SafetyCubes** — training a π0 flow-matching VLA to pick a blue cube and weave
it through a field of red obstacles *without touching them*, with the collision
constraint baked into the policy weights during training (not a runtime filter).

## Where things live

| Path | What |
|---|---|
| `sim/` | The project. MuJoCo env, scripted expert, data recorder, the `safe_pi0` policy, the DAgger loop, and all CLI scripts. |
| `sim/README.md` | **The authoritative end-to-end training runbook.** Start here. |
| `project_summary.md` | Design rationale, the safety-loss derivation, hyperparameter guide. |
| `src/lerobot/` | Upstream LeRobot training framework. One patch: `processor/converters.py` preserves `privileged.*` keys. Don't treat the rest as project code. |
| `README.md` | Short project overview + a condensed runbook (defers to `sim/README.md`). |

## Starting a training job end to end

The full, verified command sequence lives in
[`sim/README.md` → "End-to-end runbook"](./sim/README.md#end-to-end-runbook-collect--bc-warm-start--dagger).
The short version:

```bash
# 0. Setup + validate the safety geometry FIRST (torch-only, no MuJoCo/VLA needed).
#    A sign error in the SDF/FK is cheap to catch now, expensive 40 epochs into a
#    3B-param fine-tune.
uv sync --locked
uv pip install "mujoco>=3.2" pytorch_kinematics
uv run pytest sim/tests -q

# 1. Collect data — the safety term needs BOTH successes and failures.
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-bc --root data/safe_cube_bc \
    --n-episodes 100 --successes-only
uv run python -m sim.scripts.collect_demos \
    --repo-id local/safe-cube-mixed --root data/safe_cube_mixed \
    --n-episodes 200

# 2. BC warm-start: fine-tune π0 from the pretrained base with the safety loss on.
uv run python -m sim.scripts.train_safe_pi0 \
    --policy.type=safe_pi0 \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.safety_weight=1.0 \
    --dataset.repo_id=local/safe-cube-mixed --dataset.root=data/safe_cube_mixed \
    --output_dir=outputs/safe_pi0_bc \
    --batch_size=8 --steps=20000 \
    --policy.gradient_checkpointing=true

# 3. DAgger: collect on the policy's own state distribution, relabel with the
#    expert, retrain. Checkpoint path from step 2 below is the canonical layout.
uv run python -m sim.scripts.dagger \
    --repo-id local/safe-cube-dagger --root data/safe_cube_dagger \
    --output-root outputs/safe_pi0_dagger \
    --rounds 3 --episodes-per-round 40 \
    --base-policy outputs/safe_pi0_bc/checkpoints/last/pretrained_model \
    --safety-weight 1.0 --steps 8000 --batch-size 8
```

### Things an agent will trip on if it doesn't read this

- **`--safety-weight` / `--policy.safety_weight` (λ) is the central knob — sweep
  it.** Too small → safety ignored; too large → the policy collapses to pure
  avoidance and stops doing the task. The evolving safety term makes the loss
  landscape non-stationary, so **early-stop at peak rollout success, not at min
  loss.**
- **`--mjcf` defaults to `sim/assets/so101/so101_new_calib.urdf`** (the only
  committed model). Omit the flag unless you have a different model — don't
  invent a path.
- **`train_safe_pi0` is a thin shim** over LeRobot's stock `lerobot_train`.
  Importing `sim.safe_pi0_policy` registers the `safe_pi0` config with draccus;
  after that, all the standard `--policy.* / --dataset.* / --batch_size / --steps`
  args from `TrainPipelineConfig` apply. `dagger.py` shells out to this same
  entrypoint per round.
- **Checkpoints land at** `<output_dir>/checkpoints/last/pretrained_model` — that
  exact path is what `--base-policy` and `PolicyRollout(checkpoint=...)` expect.
- **`privileged.*` dataset keys are loss-only** — never feed them to the policy.
  The data processor sees only `observation.*`; the safety loss reads
  `privileged.*` straight from the batch.

## Status / caveat

The policy + training code (`sim/safe_pi0_policy.py`, `sim/dagger.py`,
`sim/scripts/*`) was written **without a runnable environment, so it is
untested**. Do step 0 (geometry unit tests) before anything else, and expect to
tune the scripted expert — see "Gotchas / known tuning needs" at the bottom of
`sim/README.md`.

## Reference

Cao, Joa, Borrelli — *A Simple Approach to Constraint-Aware Imitation Learning
with Application to Autonomous Racing*, arXiv:2503.07737.
