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

`sim/scripts/collect_demos.py`. Current working command (8-cube weaving, clean):

```bash
env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python -m sim.scripts.collect_demos \
  --repo-id local/safe-cube-mixed --root data/safe_cube_v5 \
  --successes-only --n-red-cubes 8 --max-steps 500 --n-episodes 1600 --seed 0 \
  > outputs/collect_v5.log 2>&1
```

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
  --policy.safety_weight=1.0 --policy.push_to_hub=false \
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
- **`--policy.safety_weight` (λ) is the central knob — sweep it.** Too small →
  safety ignored; too large → policy collapses to pure avoidance and stops doing
  the task.
- **Loss is NOT a clean convergence signal** (non-stationary safety term + noisy
  flow-matching). **Early-stop at peak rollout *success*, not at min loss.**

---

## 3. Evaluate / rollout videos (use held-out seeds!)

- `sim/scripts/record_policy_rollout.py` — render a trained checkpoint rolling out.
  **Its default `--seed 100` OVERLAPS training layouts (seeds 0..1599) and
  flatters the numbers** — pass `--seed 2000` (or any ≥ collection N).
- **ALWAYS pass `--action-chunking`.** π0 is trained with `chunk_size=n_action_steps=50`;
  the gripper close is a deferred mid-chunk event. Without this flag, `act()` resamples
  every step — the policy hovers at the cube and **never closes the jaws** (0% grasp,
  0% success). `--action-chunking` uses `act_queued`, which executes the full 50-step
  plan open-loop and re-infers only when the queue drains.
- `sim.evaluate.evaluate()` reports success / red-contact / clearance / ceiling
  rates; wrap a checkpoint with `sim.dagger.PolicyRollout` and pass `.act_queued`
  with a per-episode `policy.reset()`.
- `sim/scripts/record_rollout.py` — scripted-expert demo (the `videos/demo*` trio).
- All of these RENDER → run with `env -u DISPLAY MUJOCO_GL=egl`, and verify
  non-black.

```bash
env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python -m sim.scripts.record_policy_rollout \
  --checkpoint outputs/safe_pi0_bc/checkpoints/last/pretrained_model \
  --dataset-repo-id local/safe-cube-mixed --dataset-root data/safe_cube_v5 \
  --out videos/bc_rollout.mp4 --n-episodes 6 --seed 2000 --action-chunking
```

---

## 4. DAgger (after BC works)

`sim/scripts/dagger.py` runs the full collect→relabel→retrain loop. Round 0 is
pure expert; later rounds load the previous checkpoint, mix in policy actions
(`alpha = alpha0 ** round`), and **always label with the expert** (privileged
state, so labels are valid even where the policy misbehaves). Shells out to
`train_safe_pi0` per round. Run collection rounds with `env -u DISPLAY
MUJOCO_GL=egl`. See `sim/README.md` (the `sim.scripts.dagger` command) for the
full invocation; point `--base-policy` at the BC checkpoint's
`last/pretrained_model`.

**GPU-accelerated collection (`sim/scripts/collect_dagger_parallel.py`).** A
drop-in for the *collection phase only* — same expert/policy mixing and
always-label-with-expert invariant as `collect_dagger_round`, but sharded across
`--n-workers` spawn processes (each its own EGL/MuJoCo/CUDA + NVENC + π0 copy),
merged with `aggregate_datasets`. Keeps fly-overs (the label is the carry-low
expert → policy-induced fly-over states are corrective, not poison). Retrain
serially with `train_safe_pi0` afterward, as in `dagger.py`. **VRAM: ~18 GB per
worker → `--n-workers 4` is comfortably safe on the 96 GB box (default 4), 5+
crowds the 75–85 GB target.** Point `--dataset-repo-id/--dataset-root` at the
dataset the checkpoint was trained on (norm stats), `--repo-id/--root` at where
the new relabels are written, and `--seed` ≥ collection N to stay held out:

```bash
env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \
  -m sim.scripts.collect_dagger_parallel \
  --repo-id local/safe-cube-dagger --root data/safe_cube_dagger \
  --checkpoint outputs/safe_pi0_bc/checkpoints/last/pretrained_model \
  --dataset-repo-id local/safe-cube-mixed --dataset-root data/safe_cube_v5 \
  --alpha 0.5 --n-red-cubes 8 --max-steps 500 --n-episodes 400 --seed 2000 --n-workers 4
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
