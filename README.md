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
  dagger.py                PolicyRollout + collect_dagger_round (α-mixing — DEPRECATED)
  scripts/
    collect_demos.py             Collect expert rollout dataset
    collect_dagger_cleanup.py    Cleanup (branch-and-relabel) DAgger collector — current method
    collect_dagger_parallel.py   Parallel α-mixing DAgger collector — DEPRECATED (regressed 75%→0%)
    record_rollout.py            Render expert rollout as MP4
    record_policy_rollout.py     Render trained policy rollout as MP4
    train_safe_pi0.py            Training entry point
    dagger.py                    α-mixing DAgger CLI orchestrator — DEPRECATED
    view_env.py                  Interactive MuJoCo viewer
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

### 3. DAgger — "cleanup" branch-and-relabel  *(replaces the old α-mixing DAgger)*

> **Status: implemented** in `sim/scripts/collect_dagger_cleanup.py` (+
> `SafeCubeEnv.snapshot/restore/current_clearance` and `run_expert_episode(restore_state=)`).
> The α-mixing DAgger (`sim/scripts/dagger.py`,
> `sim/scripts/collect_dagger_parallel.py`, `sim.dagger.collect_dagger_round`) is
> **deprecated**: it regressed BC v6 from **75 % (6/8) → 0/6** rollout success.
> Use the method below instead.

**Why the old DAgger failed.** π0 predicts a **50-step action chunk executed
open-loop**. The old collector rolled a per-step Bernoulli mix of expert/policy
actions and relabeled every frame with the expert. Two problems, fatal together:

1. It switched actor every `switch_interval` steps (default 10) ≪ the 50-step
   chunk, injecting hard `policy.reset()` / `sync_wp_to_cube()` discontinuities
   *mid-chunk*. Every recorded 50-step training chunk straddled ~5 switches, so
   the flow-matching target was a stitched-together "Frankenstein" trajectory →
   the policy learned jaggedy, incoherent chunks.
2. Deeper: wherever the *executed* action ≠ the *recorded* (expert) action — i.e.
   any policy-driven frame — the recorded chunk is the expert's reactive 1-step
   relabel *along the policy's path*, not a coherent plan the expert would ever
   execute. Even chunk-aligned switching doesn't fix this, and even rare handoffs
   pollute a full chunk-length (50) of training-window offsets each.

Net: DAgger r1 (20k from BC v6) → **0/6 success**, drove into the field with red
contacts and timeouts. A strict regression.

**The fix — cleanup DAgger.** Decouple *generating on-policy states* from
*recording the label*; never mix actors in recorded data.

1. **Scout (not recorded).** Roll the **policy** open-loop (`act_queued`) for the
   whole episode from `env.reset(seed)`. Its only job is to discover where the
   policy gets into trouble. The trajectory is thrown away.
2. **Anchor (gate) — at the chunk-planning boundary, not the danger frame.** A
   **near-violation in the weave phase** *triggers* an anchor: `grasped` **and**
   instantaneous ee→nearest-red clearance < `gate_margin` **and** the cube isn't
   yet at the goal (excludes pre-grasp pickup and the dropoff). But the snapshot we
   **branch from** is the most recent grasped weave-phase **planning boundary**
   (`len(action_queue) == 0` → the policy re-infers a fresh chunk), not the frame
   where the cube is already next to the red. **Why:** π0 executes a 50-step chunk
   open-loop, so a near-violation is the *consequence* of the chunk the policy
   planned at the last queue-refill (≤ chunk-length steps earlier) — and the only
   on-policy state a chunked policy can be corrected at is the one it *planned
   from*. Anchoring at the danger frame would train `(danger → recovery)`, which
   the policy (mid-chunk, not re-planning there) almost never invokes; anchoring at
   the boundary trains `(planning state → coherent safe chunk)`, which actually
   steers it clear. Rising-edge trigger + cooldown + a per-episode anchor cap so one
   episode can't over-sample.
3. **Branch / relabel.** Restore each boundary snapshot and run the **pure scripted
   expert from there to completion**, recorded as one normal episode (the *exact*
   `collect_demos` code path), **except** the expert first re-roots its BFS carry
   corridor at the restored cube position (`replan_carry_from_current`) so it weaves
   *forward* from the on-policy state — never back toward the spawn→goal waypoints
   sitting behind it. Every recorded episode is 100 % expert → coherent chunks,
   smooth, zero handoffs; its first frame is `(on-policy planning state → coherent
   safe expert chunk)`, the correct DAgger relabel for a chunked policy.

**There is no hand-back.** The scout and each cleanup are *separate* rollouts that
share only a starting state. The expert never returns control to the policy — it
just finishes the task. That is precisely what dodges the unsolvable *inline*
hand-off jaggedness (where the expert's pose ≠ where the policy would resume).

**Why mid-episode (partial) demos still train cleanly.** π0 here is
`n_obs_steps=1` — purely **state-conditioned** (single observation → 50-step
chunk), no step counter, no history. A frame from a branched cleanup is
indistinguishable to the network from a BC frame; a partial demo is just valid
`(state → coherent expert chunk)` pairs sampled from *harder* states. That is
literally DAgger's definition (aggregate state-action pairs from the policy's own
visitation distribution, expert-labeled, wherever in a trajectory they fall). The
risky state is only ever an **input**, never a target (the scout is discarded),
so the labels teach **recovery, not risk-seeking**. The only thing that would
break partial demos is a history- or progress-conditioned policy
(`n_obs_steps > 1`) — π0 here is single-obs, so it doesn't apply.

**What to manage at retrain** (not specific to partial demos):
- **Mix ratio** — training weights by *frame* count; cleanup episodes are
  shorter, so collect enough anchors × scouts (or upweight) for them to bite, but
  not so many you cause over-avoidance. Always aggregate with the **full BC set**.
- **Over-avoidance / forgetting** — a correction-heavy aggregate can make the
  policy timid and avoid the field entirely (the timeout mode already seen on some
  BC v6 seeds). Fine-tune from the BC checkpoint at low LR for fewer steps;
  early-stop on rollout **success**, not loss.

**The pieces** (all implemented):
- `sim/env.py`: `snapshot() -> dict` (copies `qpos/qvel/act/ctrl/time` **plus** the
  magnetic-grip state `_attached` / `_grip_offset` / `_min_grip_qpos` — the grip is
  a kinematic lock kept *outside* the physics state, so omitting it desyncs the
  held cube on restore); `restore(snap) -> (obs, privileged)` (writes the state
  back, `mj_forward`, fresh `EpisodeStats`, **same** model/layout/renderer — does
  **not** rebuild the layout); `current_clearance() -> float` (instantaneous
  ee→nearest-red signed clearance — the live version of `stats['min_clearance']`,
  which is a running min). Round-tripped in `sim/tests/test_snapshot_restore.py`.
- `sim/expert.py`: `replan_carry_from_current(reds)` — BFS a fresh carry corridor
  from the cube's *current* position to the goal and reset `_wp_idx=0`, so a branch
  weaves forward from the restored on-policy state instead of chasing the spawn→goal
  waypoints behind it (unconditional — does not rely on `_maybe_replan_carry`).
- `sim/rollout.py`: `run_expert_episode(restore_state=None)`; when given, starts
  from `env.restore(restore_state)` instead of `env.reset(seed)` and calls
  `expert.replan_carry_from_current(...)` once before the loop — the pure-expert
  path is otherwise unchanged.
- `sim/scripts/collect_dagger_cleanup.py`: parallel collector modeled on
  `collect_dagger_parallel.py` (spawn workers, shard → `aggregate_datasets`,
  `--vcodec auto` NVENC, EGL). Per scout (`scout_for_anchors`): a
  `PolicyRollout.act_queued` rollout + the gate, tracking the latest grasped
  weave-phase **planning boundary** (`len(action_queue) == 0`) and returning that
  boundary on a near-violation (not the danger frame, see step 2); then a
  pure-expert branch per anchor via `run_expert_episode(restore_state=snap)`. The
  recorder uses `track_actor=False` (every frame is expert → no `privileged.actor`
  field → it aggregates with the BC set with **no** stripping step). Flags: `--gate-margin`
  (~0.03 m), `--max-anchors` (~3), `--cooldown-steps` (~20), `--dropoff-radius`
  (~0.05 m), optional `--branch-cap` (cap branch length past the anchor; default =
  run to completion), `--successes-only/--no-successes-only` (keep only clean
  successful branches; default on), plus the usual `--checkpoint /
  --dataset-repo-id / --dataset-root / --repo-id / --root / --n-red-cubes /
  --max-steps / --seed / --n-workers`.

The central training knob remains `--policy.safety_weight` (λ): too small → safety
ignored; too large → task collapses. See [CLAUDE.md §4](./CLAUDE.md) for the full
operational runbook and the intended command.

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
