# CLAUDE.md — operational runbook (current state)

This is the **practical, up-to-date** guide for running the SafetyCubes pipeline.
`sim/README.md` and `project_summary.md` cover design rationale and the original
plan — but several of their operational caveats are **stale** (they describe a
pre-runnable state: "untested", "expert doesn't close the XY gap"). The scripted
expert now solves ~95% of layouts; the config values and commands **in this file
supersede** the defaults described there.

---

## What this project is

Train a **π0 flow-matching VLA** to **pick up the blue cube, carry it LOW while
weaving BETWEEN a field of red cubes (never *over* them), and place it on the
green goal** — with the collision-avoidance constraint baked into the policy
*weights* during training (a differentiable safety loss), not a runtime filter.
Policy class: `safe_pi0` (`sim/safe_pi0_policy.py`): `L = L_flow_matching + λ·L_safety`.

Pipeline is **BC-first, then DAgger**. DAgger needs a competent base policy to
roll out — from an untrained π0 it just collects garbage — so you must collect
expert demos → train BC → *then* DAgger.

---

## Design theory — how the constraint gets into the weights

Instead of a post-hoc runtime safety filter, the collision constraint is a
**differentiable training term**. At deployment the policy is a vanilla π0:
`image → π_θ → action chunk` — no SDF, no cube positions, no online solve. The
"don't touch the red cubes / don't fly over them" preference lives entirely in
the fine-tuned weights.

**The loss:** `L = L_flow_matching + λ·L_safety`.
- `L_flow_matching` is π0's stock conditional-flow-matching objective (imitate
  the expert action chunk).
- `L_safety` is a soft clearance penalty evaluated on the policy's *own*
  predicted action, wired so gradients reach the VLA:

```
training:  image ─► π_θ ─► velocity v_θ ─► â₁ = a_τ + (1−τ)·v_θ  ─► differentiable FK ─► ee / held-cube trajectory
                                                                                          │
                        cube positions (privileged, sim-only) ──────► box SDF ──► clearance
                                                                                          │
                                       L_safety = −log σ(α·(clearance − margin))   (+ softplus height term)
deploy:    image ─► π_θ ─► action          (no SDF, no cube positions, no filter)
```

1. π0 (a flow-matching VLA) predicts a denoising velocity `v_θ` at flow time τ.
2. **Endpoint estimate** `â₁ = a_τ + (1−τ)·v_θ` recovers the predicted clean
   action chunk analytically so gradients flow (project_summary.md §5).
3. **Differentiable FK** (`sim/safety_geometry.py`, PyTorch) maps the predicted
   joint targets → end-effector / held-cube trajectory in the world frame.
4. **Box SDF** gives signed clearance from that trajectory to each obstacle.
5. The penalty is ~0 when comfortably clear and rises sharply inside `margin`;
   gradients propagate back through the SDF and FK into the network, so it learns
   obstacles exist *without ever seeing their coordinates at inference*.

**Two safety terms in `safe_pi0` (`sim/safe_pi0_policy.py`):**
- **Lateral clearance** — XY box-SDF distance to each red cube → the "weave
  *between* them" term.
- **Grasp-gated height ceiling** — a softplus penalty when the predicted ee
  height exceeds `ee_height_ceiling`, active only while grasped → the "carry low,
  don't fly *over*" term. **Keep its value synced with `SceneConfig`** (currently
  0.035).

`λ` (`--policy.safety_weight`) trades these against task imitation and is the
central thing to sweep.

**Privileged information** (cube positions, FK targets, `grasped`) is read **only
by the loss**, straight from the batch — never fed to the policy. The one patch
to upstream LeRobot, `src/lerobot/processor/converters.py`, exists solely to
preserve the `privileged.*` keys through to the loss; the data processor still
hands the network only `observation.*`. That separation is what keeps the
deployed policy a plain image→action VLA.

**Why BC-then-DAgger.** BC alone only constrains the *expert's* state
distribution, but the safety term matters most exactly where the policy drifts
off-distribution. DAgger rolls out the current policy, relabels its *own* visited
states with the expert (privileged state → labels stay valid even where the
policy misbehaves), and retrains — dragging the loss and its safety term onto
deployment-distribution states and surfacing fresh near-violations. DAgger needs
a competent base to roll out, so BC comes first.

**Reference:** Cao, Joa, Borrelli — *A Simple Approach to Constraint-Aware
Imitation Learning with Application to Autonomous Racing*, arXiv:2503.07737.

---

## ⚠️ Read this before running anything

### 1. Rendering MUST be headless EGL
`sim/env.py` now does `os.environ.setdefault("MUJOCO_GL", "egl")`, but **always
run sim/render/collection jobs with an explicit `env -u DISPLAY MUJOCO_GL=egl`**:

```bash
env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python -m sim.scripts.<...>
```

**Why this matters (cost us ~60-68% of every old dataset):** with `MUJOCO_GL`
unset and a `DISPLAY` present, MuJoCo binds a display-backed **GLX** context to
the X display (`:1` — the **AnyDesk** virtual display). When the remote session
disconnects/sleeps, that framebuffer goes dark and `renderer.render()` returns
**pure black** (luma 16). It comes back when reconnected, so the corruption is
*interleaved* and silent. EGL renders headlessly on the GPU, independent of any
display — immune to AnyDesk. **Always verify a new dataset/video is non-black**
(decode a few frames; black = luma 16 / max 0, real content = luma ~120-210).

### 2. Interpreter / paths
- Use **`.venv/bin/python`** (cpython 3.14). `python`/`uv` are not reliably on
  PATH in non-interactive shells; `uv run python` works interactively but for
  scripted/background runs call the venv python directly.
- Set **`PYTHONPATH=$PWD`** (repo root) when running scripts so `sim` imports.
- The pretrained π0 base is cached at `~/.cache/huggingface/hub/models--lerobot--pi0_base` (~14 GB). Don't delete it — BC fine-tunes from it.

### 3. GPU / box stability
- **RTX PRO 6000 Blackwell, 96 GB.** The box **reboots under GPU load** unless
  power/fan-capped. The user manages power caps — **don't touch them**. Keep all
  long runs **resumable**.
- Target VRAM **75-85 GB, no swap**.
- Don't sleep-poll background tasks — they emit a notification on completion;
  read their `.output` file then.

---

## Current scene config (the hard-won geometry)

All in `sim/configs.py` unless noted. The task is "weave BETWEEN short (25 mm)
obstacles, stay low, don't go over." These values are a tuned sweet spot —
**change them only with the reasoning below in mind:**

| Param | Value | Why |
|---|---|---|
| `SceneConfig.n_red_cubes` | **8** | The arm only reaches ~x∈[0.10, 0.40]; 8 cubes is a comfortable clutter level in that band. (At the old 0.11 separation this was a hard packing limit; at 0.07 there's slack to add more if desired.) |
| `SceneConfig.min_cube_separation` | **0.07** | **Placement-diversity knob, decoupled from clearance.** At 0.11 (=2×clearance) 8 cubes packed so tight in the 0.30 m field they snapped to a near-fixed 3×3 grid → **8 predictable "hubs"** in the occupancy heatmap. 0.07 gives ~4× the slots → cubes scatter freely (peak cell occupancy ~21→8, layouts stay 100% valid). It's **below 2×clearance**, so the planner threads the wide gaps and routes *around* the occasional sub-0.11 m tight pair — still collision-safe, just a more varied "navigate a cluttered field" task. |
| `SceneConfig.path_clearance_radius` | **0.055** | BFS corridor clearance, sized to the **finger** (not just the cube) — the gripper's real physical need and the **binding safety constraint**. **Do NOT lower** to open tighter gaps (fingers would clip reds). Threading *between* two cubes needs this on each side → a gap ≥ 0.11 m; below that the path rounds the pair. |
| `ExpertConfig.carry_z` | **0.030** | Cube-center carry height. Low enough the cube bottom (~17 mm) can't clear a 25 mm red top → **no fly-over**; high enough the bulky gripper (reaches ~25 mm below the cube) clears the table. (0.015 drags the fingers on the floor; the expert also undershoots to ~24 mm actual.) |
| `SceneConfig.ee_height_ceiling` | **0.035** | Stay-low limit. Catches any cube ≥38 mm (a fly-over) while leaving ~5 mm margin above the carry height. **Mirror this value in `sim/safe_pi0_policy.py` (the safety-loss copy) — keep them in sync.** |

**Fly-over** is a first-class env stat (`info["stats"]["fly_over"]`): geometric —
the held cube is laterally over a red's XY footprint AND its bottom clears the
red's top. It's structurally ~0 because the BFS keeps the *cube* clear in XY; any
residual `red_contact` is the wider gripper *finger*, not the cube — just discard
those clips (the collector already does). Note `min_cube_separation` and
`path_clearance_radius` are now **independent**: separation controls layout
variety, clearance controls gripper safety. Don't re-couple them.

---

## 1. Collect data (scripted expert → BC dataset)

`sim/scripts/collect_demos.py`. Current working command (8-cube weaving, A* expert, clean):

```bash
env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python -m sim.scripts.collect_demos \
  --repo-id local/safe-cube-mixed --root data/safe_cube_v6 \
  --successes-only --n-red-cubes 8 --max-steps 500 --n-episodes 1600 --seed 0 \
  --path-clearance-weight 1.0 \
  > outputs/collect_v6.log 2>&1
```

`--path-clearance-weight 1.0` enables the cost-field A* corridor planner, which bows expert
paths toward the center of free gaps rather than hugging the clearance boundary (BFS default).
λ=1.0 threads tight gaps; λ≥3 tends to route around the field — sweep if paths look timid.
Add `--path-wall-field-sides` alongside a high λ to force through-field weaving.

- **`--successes-only`** drops failures. `collect_demos` *also* unconditionally
  drops any `fly_over` episode (a demo of going *over* an obstacle is poison for
  the stay-low task) — the discard is `fell_over or (successes_only and not success)`.
- ~94% success → ~1500 clean demos from 1600 attempts. ~0.2 ep/s (~2 h). Light
  GPU load (~1.5 GB, EGL render + CPU SVT-AV1 encode) — gentler than training.
- Seeds: episode N uses `seed + N`. So this run consumes seeds 0..1599 → **eval
  must use seeds ≥ 1600** (don't evaluate on training layouts).
- **After collection, verify non-black** (decode ~30 episodes, both cameras,
  check max>0 and mean luma >100). The coverage script below does this.

Dataset features (`privileged.*` are **LOSS-ONLY — never feed them to the
policy**; the data processor sees only `observation.*`):
`observation.images.agentview|wrist (224×224×3 video)`, `observation.state (6)`,
`action (6)`, `privileged.cube_positions (n_red×3)`, `cube_half_extents`,
`blue_cube_pos`, `goal_pos`, `ee_pos`, `grasped`.
Note: `cube_positions` is `n_red×3` → **24-dim for 8 cubes** (was 30 for 10);
reshape `(-1, 3)`, don't hard-code 10.

### Layout-coverage / non-black audit
`/tmp/v5_layout_coverage.py` (writes `videos/v5_layout_coverage.png`): blue-spawn
scatter + red-occupancy heatmap + 30-episode non-black spot-check. Run after
collection. Variety: every episode's blue spawn and red layout are unique
(continuous randomization); goal is fixed at (0.225, −0.20).

---

## 2. BC warm start (fine-tune π0 with the safety loss on)

`sim/scripts/train_safe_pi0.py` is a thin shim over LeRobot's `lerobot_train`;
importing it registers `safe_pi0`, after which all standard `--policy.* /
--dataset.* / --batch_size / --steps` args apply. **Training does not render**
(it decodes the dataset videos via torchcodec), so EGL is irrelevant here — just
set the alloc config.

> ⚠️ **ALWAYS launch training inside a `tmux` session — never as a plain
> background job.** A backgrounded `python` dies on SSH/session disconnect (we
> lost a DAgger run that way); `tmux` survives disconnects, and the per-`save_freq`
> checkpoints + `--resume` survive box reboots. Pattern: `tmux new -s <name>`, then
> run the command piped to `tee <log>` so it's both on-screen and logged.
>
> ⚠️ **ALWAYS enable wandb so the run is viewable:**
> `--wandb.enable=true --wandb.disable_artifact=true --wandb.project=safe-cube`
> (entity `models-university-of-california-berkeley1717`). `disable_artifact=true`
> stops it uploading the ~22 GB checkpoints. **Gotcha:** `--resume=true` makes
> wandb try to resume the *prior* run and errors if that run had wandb off
> (`Couldn't get the previous WandB run ID`) — enable wandb from the **first**
> launch, or restart fresh.

```bash
tmux new -s dagger_r1     # then, INSIDE the tmux session:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD .venv/bin/python \
  -m sim.scripts.train_safe_pi0 \
  --policy.type=safe_pi0 --policy.pretrained_path=lerobot/pi0_base \
  --policy.safety_weight=1.0 --policy.obstacle_weight=2.0 --policy.ceiling_weight=4.0 \
  --policy.sdf_margin=0.02 --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id=local/safe-cube-mixed --dataset.root=data/safe_cube_v5 \
  --output_dir=outputs/safe_pi0_bc --batch_size=48 --steps=20000 --save_freq=2000 \
  --wandb.enable=true --wandb.disable_artifact=true --wandb.project=safe-cube \
  2>&1 | tee outputs/bc_train.log
```

- **REQUIRED: `--policy.push_to_hub=false`** (else `validate()` errors on a
  missing repo_id).
- **Memory envelope:** batch 48 + `gradient_checkpointing=true` +
  `expandable_segments:True` → **~77 GB, ~5 s/step** (dual-cam vision + safety FK)
  → ~21 h for 15 k steps. Keep grad checkpointing ON. Checkpoints are ~22 GB each
  — `save_freq` controls disk burn.
- **Checkpoints land at** `<output_dir>/checkpoints/<step>/` and
  `.../checkpoints/last/pretrained_model` — that exact path is what
  `--base-policy` (DAgger) and `PolicyRollout(checkpoint=...)` expect.
- **Resume** (if the box reboots): `--config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json --resume=true [--steps=N]`.
  The scheduler is rebuilt from `cfg.steps` *before* the step counter is restored;
  CLI args override the saved `train_config.json`. To extend a run, resume with a
  larger `--steps` (cosine window re-stretches).
- **LR schedule:** `cosine_decay_with_warmup`, peak `2.5e-5`, **floor `2.5e-6`
  (10% of peak — NOT zero)**. Auto-scales the decay window when `steps < 30000`.
- **Safety weighting is split into per-term knobs:** `--policy.obstacle_weight`
  (lateral collision, default **2.0**) and `--policy.ceiling_weight` (stay-low,
  default **4.0**), both scaled by the overall `--policy.safety_weight` (λ,
  default 1.0). **Effective coeffs in the total loss: collision = λ·obstacle_weight
  = 2.0, ceiling = λ·ceiling_weight = 4.0, flow = 1.0.** `--policy.sdf_margin=0.02`
  is the clearance buffer. The stronger (2×) collision term is safe on **A*-collected
  data** because the expert carries at max-available clearance, so the penalty fires
  mostly off-distribution rather than on the expert (on the old BFS data it would
  fight the expert's boundary-hugging — see memory). Sweep λ for overall safety
  strength, or the per-term weights for balance. Too small → safety ignored; too
  large → policy collapses to pure avoidance and stops doing the task.
- **Loss is NOT a clean convergence signal** (non-stationary safety term + noisy
  flow-matching). **Early-stop at peak rollout *success*, not at min loss.**

### 2b. `safe_diffusion` — the second policy class (branch `safe-diffusion-policy`)

`sim/safe_diffusion_policy.py` applies the **same safety loss to a Diffusion
Policy** (Chi et al.) backbone instead of π0: `L = L_diffusion + λ·L_safety`,
same `sim/safety_geometry.py` FK → box-SDF → clearance chain, same `privileged.*`
loss-only contract, same coefficient names and defaults.

**Why it exists: iteration speed.** π0 is ~5 s/step at batch 48 (~21 h/run),
which is why λ — "the central knob, sweep it first" — has never actually been
swept. `safe_diffusion` is 274 M params and measured **~0.03 s/step at batch 4**
on this box, with ~1 GB checkpoints instead of 22 GB. It's the cheap vehicle for
sweeping `safety_weight` / `obstacle_weight` / `ceiling_weight` / `sdf_margin`,
and it adds an architecture axis to the ablation table (*the safety loss works
across generative policy classes*). The trade: no internet-pretrained VLA prior,
no language conditioning, trains **from scratch** — expect a lower absolute
ceiling than π0 and read cross-class numbers as trends, not interchangeable.

```bash
uv pip install --python .venv/bin/python "diffusers>=0.27.2,<0.36.0"   # one-time

tmux new -s diff_bc     # then, INSIDE tmux:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD .venv/bin/python \
  -m sim.scripts.train_safe_diffusion \
  --policy.type=safe_diffusion --policy.push_to_hub=false \
  --policy.safety_weight=1.0 --policy.obstacle_weight=2.0 --policy.ceiling_weight=4.0 \
  --policy.sdf_margin=0.02 \
  --dataset.repo_id=local/safe-cube-mixed --dataset.root=data/safe_cube_agg_v7.1 \
  --output_dir=outputs/safe_diffusion_bc --batch_size=64 --steps=100000 --save_freq=10000 \
  --wandb.enable=true --wandb.disable_artifact=true --wandb.project=safe-cube \
  2>&1 | tee outputs/diffusion_bc_train.log
```

- **No `--policy.pretrained_path`** — there is no `pi0_base` analogue; only the
  ResNet18 backbone is ImageNet-initialized. Budget many more *steps* at a tiny
  fraction of the wall clock. Early-stop on rollout **success**, same as §2.
- **Everything downstream is policy-agnostic and works unchanged**: `PolicyRollout`
  registers both classes, so `benchmark_policy.py`, `record_policy_rollout.py` and
  the cleanup-DAgger collector take a `safe_diffusion` checkpoint as-is (verified
  end-to-end).
- **Chunking defaults are set for PARITY with π0** (`n_action_steps=50`, executed
  fully open-loop), so a cross-class comparison isolates the backbone rather than
  confounding it with the control horizon. `horizon=56` (not 50) because the U-Net
  needs a multiple of `2**len(down_dims)=8`. **Lowering `--policy.n_action_steps`
  to ~8–16 turns on receding-horizon control** — DP's native mode, an
  inference-time knob needing no retrain, and the first thing to try against the
  **placement undershoot**, which is a consequence of committing to a long chunk.
- **`n_obs_steps=1` is load-bearing, not a default** — the cleanup-DAgger relabel
  (§4) is only valid for a purely state-conditioned policy, since a branched
  episode's first frame has no real history. Don't raise it without re-reading §4.
- **DDPM endpoint estimate.** π0's `a_hat = x_t − t·v_θ` becomes
  `x̂₀ = (x_k − √(1−ᾱ_k)·ε_θ)/√ᾱ_k`. Note DDPM has **no convention flip** — noise
  rises monotonically with `k`, so unlike π0 (see the footgun in
  `safe_pi0_policy.py`'s docstring) small-k = data side reads the way you expect.
  Two numerical caveats π0 doesn't have: the `1/√ᾱ_k` blow-up at high noise
  (handled by the `(1−k/(T−1))` weighting, mirroring π0's `(1−t)`), and
  `clamp_endpoint`, which clamps `x̂₀` to `±clip_sample_range` exactly as the DDPM
  sampler does at every reverse step.
- **⚠️ ACTION normalization differs between the classes**: `safe_pi0` inherits π0's
  **MEAN_STD**, `safe_diffusion` uses DP's **MIN_MAX**. The safety loss must
  un-normalize with the matching affine map before FK. Getting this wrong does not
  crash — it feeds FK plausible garbage and silently penalises the wrong region of
  space. Covered by `sim/tests/test_safe_diffusion.py` (`uv run pytest
  sim/tests/test_safe_diffusion.py -q`, CPU, seconds), which also pins the
  config↔policy naming contract LeRobot's factory resolves `--policy.type` through.
- Sanity baseline (measured on `safe_cube_agg_v7.1` expert data, batch 32): FK
  vs `privileged.ee_pos` **24 mm mean** (action is a joint *target* vs measured ee
  — control lag, not error), ee z ∈ [0.010, 0.085] m, clearance mean 0.046 m,
  `L_obstacle` 0.42, `L_ceiling` 0.61. Note the expert sits near the knee of the
  ceiling penalty, so at `ceiling_weight=4` the expert data itself carries a
  constant ~2.4 — worth a sweep now that sweeping is cheap.

---

## 3. Evaluate / rollout videos (use held-out seeds!)

- `sim/scripts/record_policy_rollout.py` — render a trained checkpoint rolling out.
  **Its default `--seed 100` OVERLAPS training layouts (seeds 0..1599) and
  flatters the numbers** — pass `--seed 2000` (or any ≥ collection N).
- **ALWAYS pass `--action-chunking`.** π0 is trained with `chunk_size=n_action_steps=30`
  (overridden in `SafePI0Config`; the LeRobot upstream default is 50);
  the gripper close is a deferred mid-chunk event. Without this flag, `act()` resamples
  every step — the policy hovers at the cube and **never closes the jaws** (0% grasp,
  0% success). `--action-chunking` uses `act_queued`, which executes the full 30-step
  plan open-loop and re-infers only when the queue drains.
- `sim.evaluate.evaluate()` reports success / red-contact / clearance / ceiling
  rates; wrap a checkpoint with `sim.dagger.PolicyRollout` and pass `.act_queued`
  with a per-episode `policy.reset()`.
- `sim/scripts/record_rollout.py` — scripted-expert demo (the `videos/demo*` trio).
- `sim/scripts/viz_cleanup_astar.py` — render stored cleanup-DAgger episodes from a
  dataset with the **A* carry path projected onto the agentview frame**. Reads
  `privileged.cube_positions`/`blue_cube_pos`/`goal_pos` from the parquet, re-plans
  the A* path (λ=1.0, smooth-interior) from the episode's first frame, and overlays
  it per-frame on the decoded video (yellow = ahead, olive = done, orange = current
  target). Output is composite 960×480 (agentview + wrist). Cleanup v7.1 used λ=1.0
  with smooth-interior (`path_clearance_interior_weight=11`, base=5).
  ```bash
  env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \
    -m sim.scripts.viz_cleanup_astar \
    --root data/safe_cube_cleanup_v7.1 --out videos/cleanup_v7.1_astar.mp4 \
    --n-episodes 8
  ```
- All of these RENDER → run with `env -u DISPLAY MUJOCO_GL=egl`, and verify
  non-black.

```bash
env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python -m sim.scripts.record_policy_rollout \
  --checkpoint outputs/safe_pi0_bc/checkpoints/last/pretrained_model \
  --dataset-repo-id local/safe-cube-mixed --dataset-root data/safe_cube_v5 \
  --out videos/bc_rollout.mp4 --n-episodes 6 --seed 2000 --action-chunking
```

---

## 4. DAgger — "cleanup" branch-and-relabel (after BC works)

> **Status: IMPLEMENTED.** Collector `sim/scripts/collect_dagger_cleanup.py`;
> env primitives `SafeCubeEnv.snapshot/restore/current_clearance`;
> `run_expert_episode(restore_state=)`; round-trip test
> `sim/tests/test_snapshot_restore.py`. (Not yet *run* at scale — validate a
> retrain before trusting it.) The old α-mixing DAgger
> (`sim/scripts/dagger.py`, `sim/scripts/collect_dagger_parallel.py`,
> `sim.dagger.collect_dagger_round`) is **DEPRECATED — do not use.** It regressed
> BC v6 from **75 % (6/8) → 0/6** rollout success (`outputs/rollout_dagger_r1.log`:
> drove into the field, 3/6 red contacts, timeouts). Root cause and the
> replacement are below. `PolicyRollout` in `sim/dagger.py` is still good and is
> reused by the new method.

**Why the old DAgger regressed.** π0 predicts a **50-step chunk executed
open-loop** (`--action-chunking` / `act_queued`, see §3). The old collector
executed a per-step Bernoulli expert/policy mix and relabeled every frame with the
expert. Fatal combination:
1. It switched actor every `switch_interval` steps (default 10) ≪ chunk 50, with
   hard `policy.reset()` / `sync_wp_to_cube()` discontinuities *mid-chunk*. Every
   recorded 50-step training chunk straddled ~5 switches → a "Frankenstein"
   flow-matching target → the policy learned jaggedy, incoherent chunks.
2. Deeper: wherever *executed ≠ recorded* (any policy-driven frame) the recorded
   chunk is the expert's reactive 1-step relabel along the policy's path, not a
   coherent expert plan. Chunk-aligned switching doesn't fix this; even rare
   handoffs pollute a full chunk-length (50) of training-window offsets each.

**The replacement — cleanup DAgger (branch-and-relabel / on-policy-anchored expert
demos).** Decouple *generating on-policy states* from *recording the label*. Never
mix actors in recorded data:

1. **Scout (NOT recorded).** Roll the **policy** open-loop (`PolicyRollout.act_queued`)
   for the whole episode from `env.reset(seed)`. Only job: find where the policy
   gets in trouble. Discarded.
2. **Anchor (gate) — at the chunk-PLANNING boundary, not the danger frame.** A
   **near-violation in the weave phase** *triggers* an anchor: `grasped` AND
   `env.current_clearance() < gate_margin` AND the cube isn't at the goal yet
   (excludes pickup = pre-grasp, and dropoff = cube within `dropoff_radius` of
   goal). But the snapshot we **branch from** is the most recent grasped weave-phase
   **planning boundary** (`len(action_queue) == 0` → the policy re-infers a fresh
   chunk), **not** the frame where the cube is already next to the red. **Why:** π0
   executes a 50-step chunk open-loop, so a near-violation is the *consequence* of
   the chunk the policy planned at the last queue-refill (≤ chunk-length steps
   earlier); the only on-policy state a chunked policy can be corrected at is the
   one it *planned from*. Snapshotting at the danger frame trains
   `(danger → recovery)`, which the policy — mid-chunk, not re-planning at the
   danger — almost never invokes; snapshotting at the boundary trains
   `(planning state → coherent safe chunk)`, which is what actually steers it clear.
   (Relabeling the policy's *per-step* pre-danger path instead would reintroduce the
   Frankenstein incoherence — the expert's reactive actions along the policy's path
   aren't a coherent plan.) Rising-edge + cooldown + per-episode `max_anchors` cap.
3. **Branch / relabel.** `env.restore(boundary_snap)`, then run the **pure scripted
   expert to completion** from there, recorded as one normal episode — identical to
   `collect_demos` (`run_expert_episode(..., choose_executed=None,
   restore_state=snap)`), **except** the expert first re-roots its BFS carry corridor
   at the restored cube position (`replan_carry_from_current`) so it weaves
   *forward* from the on-policy state, never back toward the spawn→goal waypoints
   sitting behind it. Every recorded episode is 100 % expert → coherent chunks,
   smooth, zero handoffs; its **first frame is `(on-policy planning state →
   coherent safe expert chunk)`** — the correct DAgger relabel for a chunked policy.

**No hand-back** — scout and cleanup are *separate* rollouts sharing only a start
state; the expert never returns control to the policy, it just finishes. That is
what avoids the inline-handoff jaggedness (expert pose ≠ where the policy resumes).

**Mid-episode demos train cleanly** because π0 here is `n_obs_steps=1` — purely
state-conditioned (single obs → 50-chunk), no history / step counter. A branched
cleanup frame is indistinguishable from a BC frame; a partial demo is valid
`(state → coherent expert chunk)` pairs from harder states. Risky states are only
ever inputs, never targets (scout discarded) → labels teach **recovery, not
risk-seeking**. (Would only break if `n_obs_steps > 1` — it isn't.)

**Visualizing.** A recorded cleanup episode starts *at the anchor* — cube already
grasped, mid-field, close to a red — and shows the pure expert weaving out to the
goal. The policy **scout is NOT recorded**, so the dataset has no footage of the
policy driving *into* trouble; `visualize_dataset_episodes.py` will (correctly)
play a clip that opens in a near-violation. It already handles the absent
`privileged.actor` field (`track_actor=False`) gracefully — prints
`actor tracking=no` and shows `?` in the actor banner, no crash. To actually *see*
the policy cause the near-violation you'd add a debug render of the scout that
marks the anchor frame (not part of this spec).

**Retrain hygiene** (aggregate cleanup with the **full BC set**, then
`train_safe_pi0` serially as in §2):
- Training weights by **frame** count; cleanup episodes are short → collect enough
  anchors × scouts (or upweight) to matter, but not so many you induce
  **over-avoidance** (timid field-avoidance / timeouts).
- Fine-tune from the BC checkpoint, **low LR / fewer steps**, early-stop on rollout
  **success** (loss is not a clean signal — see §2).

### Placement-anchor variant (`--anchor-mode place`) — same skeleton, different gate

Same scout → anchor → pure-expert-relabel skeleton as the weave gate above; only
the **trigger** and the **boundary-capture rule** change. It fixes the **open-loop
placement undershoot** — the v7.1 policy drops the cube ~18 mm off-center, biased
toward the arm base (`outputs/quantify_place_bias.log`), because it commits a
50-step descend chunk and never corrects mid-chunk.

**What triggers a branch (the key contrast with weave).** The weave gate fires on a
*measured near-violation* — `grasped AND current_clearance() < gate_margin AND not
at_goal` — a genuine danger event it has to hunt for. The place gate instead fires
on a **distance band of the held cube → goal**
(`collect_dagger_cleanup.py`, `off` at `:191`):

```
off = grasped AND place_tol < d_goal <= place_approach_radius
```

i.e. the grasped cube has entered the goal-approach band (≤ `--place-approach-radius`,
default 0.10 m) but is **not yet centered** (> `--place-tol`, default 0.015 m).
Rising-edge + cooldown, exactly like weave.

**This is NOT a final-error threshold.** There is no "wait for the drop, measure
the miss, branch if it exceeds X". Because the policy *reliably* undershoots, the
rising edge fires the moment the grasped cube first comes within 0.10 m of the goal
— so it anchors **every placement attempt**, not just the bad ones. That's
deliberate: the weave gate has to *find* a rare near-collision (hence the live
clearance measurement), but every placement is undershooting, so the place gate
doesn't need to detect error — it just catches the policy *in the act of placing*
and hands that on-policy approach state to the expert. The correction signal comes
entirely from the **expert relabel** (a centered `DESCEND2` to within 7.5 mm),
never from the gate quantifying how wrong the policy was.

**Boundary capture is unconditional.** Both gates branch from the most-recent
grasped **planning boundary** (`len(action_queue) == 0`), not the trigger frame
(same chunked-policy reasoning as the weave gate — you can only correct a chunked
policy at the state it *planned from*). But the weave gate only *caches* a boundary
while the cube is far from goal (`d_goal > dropoff_radius`); the place gate caches
one at **every** grasped re-plan, NOT distance-gated (`last_place_boundary` at
`:156–163`) — because the chunk that ultimately lands off-center is often planned
from *outside* the 0.10 m band (a 50-step chunk covers a lot of ground), so
distance-gating the capture would routinely leave no boundary to branch from.

`weave` and `place` cover **complementary regions** (far-from-goal vs near-goal),
keep distinct boundary snapshots, and can run together with `--anchor-mode both`.

**The pieces (all built — file/function map):**
- `sim/env.py`:
  - `snapshot() -> dict` — copy `data.qpos/qvel/act/ctrl/time` **and** the
    magnetic-grip state `_attached` / `_grip_offset` / `_min_grip_qpos`. The grip
    is a kinematic lock maintained *outside* the physics state; omit it and the
    held cube desyncs from the gripper on restore. (`data.act` may be size-0 —
    guard the copy/restore.)
  - `restore(snap) -> (obs, privileged)` — write the state back, `mj_forward`,
    `self._stats = EpisodeStats()`, return `_observe()/_privileged()`. **Same**
    model/layout/renderer — does NOT rebuild the layout (the point: the cleanup
    runs in the scout's exact world; snapshots are valid only for that env
    instance).
  - `current_clearance() -> float` — instantaneous ee→nearest-red signed clearance
    (live version of the running `stats['min_clearance']`; replicate the box-SDF
    loop in `_update_episode_state` step 5).
- `sim/expert.py`: `replan_carry_from_current(reds)` — BFS a fresh carry corridor
  from the cube's CURRENT position to the goal and reset `_wp_idx=0` on it.
  Unconditional (does NOT depend on `_maybe_replan_carry`'s off-path threshold), so
  a branch always weaves *forward* from the restored on-policy state instead of
  chasing the spawn→goal waypoints behind it (which would drive the arm backward).
- `sim/rollout.py`: `restore_state=None` on `run_expert_episode`; when set, start
  from `env.restore(restore_state)` instead of `env.reset(seed)` **and** call
  `expert.replan_carry_from_current(info["cube_positions"])` once before the loop
  (rest unchanged — same pure-expert path `collect_demos` uses).
- `sim/scripts/collect_dagger_cleanup.py`: parallel collector built on the
  `collect_dagger_parallel.py` scaffolding (spawn workers, disjoint seed blocks,
  shard → `aggregate_datasets`, `--vcodec auto` NVENC, ~18 GB π0/worker →
  `--n-workers 4` safe, 5+ crowds the 75–85 GB target). Per scout episode the
  module-level `scout_for_anchors()` runs the `PolicyRollout.act_queued` rollout +
  the gate; it tracks the latest grasped weave-phase **planning boundary**
  (`len(policy.policy._action_queue) == 0`) and, on a near-violation, returns that
  boundary snapshot (not the danger frame, see step 2 above), deduped by identity.
  Then a pure-expert branch per anchor via `run_expert_episode(restore_state=snap)`.
  The recorder uses **`track_actor=False`**
  (every frame is expert → no `privileged.actor` field → aggregates with the BC set
  with NO stripping — unlike the deprecated collector, see [[dagger-aggregate-strip-actor]]).
  Branch hygiene mirrors `collect_demos`: drop fly-overs **and** finger-clip red
  contacts always, and (under `--successes-only`, **default on**) non-successes —
  so the cleanup set matches the BC set's clean distribution. The scout env uses
  `terminate_on_red_contact=False` so neither the scout nor a branch dies on a graze.
- Flags: `--gate-margin` (≈0.03 m), `--max-anchors` (≈3), `--cooldown-steps`
  (≈20), `--dropoff-radius` (≈0.05 m), optional `--branch-cap` (limit branch length
  past the anchor; default = run to completion), `--successes-only/--no-successes-only`
  (default on), `--path-clearance-weight` (A* λ, default 1.0), `--path-wall-field-sides`,
  plus the usual `--checkpoint / --dataset-repo-id / --dataset-root /
  --repo-id / --root / --n-red-cubes / --max-steps / --seed / --n-workers / --vcodec`.
- **`--anchor-mode {weave,place,both}`** (default `weave`, branch
  `placement-anchor-dagger`). **`place`** is a second gate that fixes the **open-loop
  placement undershoot** (the v7.1 policy drops the cube ~18 mm off-center, biased
  toward the arm base — see `outputs/quantify_place_bias.log`). It anchors each
  grasped *placement attempt* (cube within `--place-approach-radius`, default 0.10 m,
  of the goal, off-center beyond `--place-tol`, default 0.015 m) at the most recent
  grasped planning boundary — captured at **every** grasped re-plan, NOT
  distance-gated (the placing chunk often originates outside the approach radius) —
  and the pure expert relabels a **centered** drop (`DESCEND2` to within 7.5 mm).
  Yield ≈1.3 clean branches/scout; branches are **short** (≈30–100 frames), and
  `h264_nvenc` can fail to emit a file on the shortest clips → encode with
  **SOFTWARE h264: `--vcodec h264 --gop 4`**. libx264 is robust on short clips,
  and `--gop 4` makes the metadata (h264, g=4, crf=30, preset=None) an **exact
  match to the `h264_nvenc`-collected BC set** so `aggregate_datasets` accepts it.
  **Do NOT use `--vcodec libsvtav1` here** — it survives short clips but yields
  **av1 g=2**, which fails `aggregate_datasets`' video-metadata equality check
  against the h264 BC set (`validate_all_metadata: Same features is expected`).
  Staged scripts: `outputs/run_place_cleanup.sh` (collect 800 branches,
  seeds 4000+, from the v7.1 ckpt) → `outputs/run_train_place.sh` (aggregate
  `safe_cube_agg_v7.1` + `safe_cube_place_cleanup` → `safe_cube_agg_place`, fine-tune
  from the v7.1 cleanup ckpt at lr 1e-5 → `outputs/safe_pi0_place`). Early-stop on
  rollout success **and** re-measured placement bias (`quantify_place_bias.py`).

**Intended command** (held-out seeds ≥ collection N; `--dataset-*` = the BC set the
checkpoint was trained on, for norm stats; `--repo-id/--root` = where the new
cleanup demos are written):

```bash
env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \
  -m sim.scripts.collect_dagger_cleanup \
  --repo-id local/safe-cube-cleanup --root data/safe_cube_cleanup \
  --checkpoint outputs/safe_pi0_bc_v6/checkpoints/last/pretrained_model \
  --dataset-repo-id local/safe-cube-mixed --dataset-root data/safe_cube_v6 \
  --n-red-cubes 8 --max-steps 500 --n-episodes 400 --seed 2000 \
  --gate-margin 0.025 --max-anchors 3 --n-workers 4 \
  --path-clearance-weight 1.0

# then aggregate with the BC set and retrain (serial), as in §2:
#   aggregate_datasets([safe_cube_v6, safe_cube_cleanup]) → data/safe_cube_v7
#   train_safe_pi0, resume from the BC v6 checkpoint, low LR, early-stop on success
```

---

## Repo state (datasets / outputs)

- `data/safe_cube_v5` — **current clean weaving BC set** (8 cubes). The only live
  dataset. (Old `safe_cube_v3` / `safe_cube_v4` / `safe_cube_dagger` and **all
  old `outputs/safe_pi0_*` checkpoints were DELETED** — they were black-frame
  corrupted and/or superseded. Don't trust any pre-EGL-fix artifact.)
- `videos/`: `demo*` trio (current weaving expert demo), `safe_pi0_25k_rollout*`
  (old reference), `v4_layout_coverage.png`, `v5_layout_coverage.png`.
- `data/`, `outputs/`, `videos/` are gitignored ("regeneratable") — nothing in
  them is in git history.

## Conventions
- **Omit the `Co-Authored-By: Claude` trailer** on commits in this repo.
- Concurrent agents may share the working tree — **re-read files before editing**
  and commit with an explicit pathspec.
- Design rationale & the safety-loss derivation live in `project_summary.md`;
  feature/key contracts in `sim/README.md`.
