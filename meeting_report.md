# SafetyCubes Project — Progress Report
*As of 2026-06-07*

---

## Goal

Train a π0 flow-matching VLA (3B parameter vision-language-action model) to **pick up a blue cube, carry it LOW while weaving between a field of red obstacle cubes (never over them), and place it on a green goal** — with the collision-avoidance constraint baked into the policy weights during training via a differentiable safety loss, not enforced at runtime.

The constraint is never enforced at deployment: the policy is a plain `image → action` VLA with no SDF, no cube positions, no filter. "Don't fly over" lives entirely in the fine-tuned weights.

---

## Architecture

**Policy class:** `SafePI0Policy` (subclass of LeRobot's `PI0Policy`)

**Loss:**
```
L = L_flow_matching + λ · L_safety
L_safety = L_obstacle + 4 · L_ceiling
```

**How the safety gradient reaches the network:**

1. π0 predicts a denoising velocity `v_θ(x_t, t, obs)` — a vector in joint-angle space representing the direction from data toward noise along the interpolation path. Its units are radians per unit of flow-time τ ∈ [0,1], not rad/s; it is never directly executed on the robot.
2. The clean action endpoint is recovered analytically: `â = x_t - t · v_θ`. This estimate is exact when `v_θ` is perfect, and has a direct gradient w.r.t. `v_θ`.
3. `â` is un-normalized and passed through differentiable FK (via URDF + `pytorch_kinematics`) to get the predicted ee world-frame trajectory.
4. The XY box-SDF of that trajectory against the red cube positions (privileged, from the batch) gives a clearance signal.
5. `L_obstacle = softplus(-α · (clearance - margin))` — near-zero when safe, grows linearly in violation depth (gradient never vanishes for deep violations).
6. `L_ceiling = softplus(α_ceiling · (ee_z - ceiling))` — spikes when ee height exceeds 35 mm, gated to the carry phase via `privileged.grasped`.
7. Gradients flow through the SDF and FK back into the VLM weights.

**Privileged information** (cube positions, grasped state) is read only by the loss, never fed to the network. The data processor passes `privileged.*` keys through to the loss while handing only `observation.*` to the VLM.

**Action format:** 6-dimensional absolute joint-position targets (5 arm joints + gripper), predicted as a chunk of 50 steps.

---

## Simulation Environment

**Task:** SO-101 arm picks up a blue cube, carries it low through a field of 8 red obstacle cubes (25 mm tall), places it on a fixed goal.

**Key geometry (tuned values):**

| Parameter | Value | Rationale |
|---|---|---|
| `n_red_cubes` | 8 | Comfortable clutter in the reachable arm band |
| `min_cube_separation` | 0.07 m | Decoupled from clearance — gives layout variety (4× more valid slots than 0.11 m), occasional tight pairs are routed around |
| `path_clearance_radius` | 0.055 m | Sized to the physical gripper finger, not just the cube |
| `carry_z` | 0.030 m | Low enough cube can't clear 25 mm red tops; high enough gripper clears table |
| `ee_height_ceiling` | 0.035 m | Stay-low limit; mirrored in the safety loss |

**Grasp mechanism:** kinematic "magnetic" lock — the cube's freejoint is written every substep to follow the grasp site when jaws close within 4 cm. Physics grasping was tried and failed (single moving jaw shoves the light cube away before contact; friction up to 6.0 didn't help). The magnetic grip includes a lerp from the attach offset to the jaw center as the jaws close, so the cube appears to be scooped in rather than teleporting.

**Grasp geometry:** TCP (`grasp_site`) placed at the jaw-gap center, not the `ee_site` (which sits 3 cm off-center at the fixed jaw). Critical frame gotcha: `gripper_frame_link` is 180° rotated about y from `gripper_link` body frame — x and z offsets must be negated when writing the site pos in URDF/MuJoCo. A missing negation caused a 6 cm grasp placement bug that was masked by the unconditional kinematic lock.

**Expert:** scripted BFS-waypoint planner, ~94% success rate. Side-grasps the blue cube at 0.25× cube height. Discards fly-over and failure episodes at collection time. The expert solves ~95% of layouts; `collect_demos --successes-only` discards the rest.

---

## Training Pipeline

**Stage 1: BC (Behavioral Cloning)**
Fine-tune π0 from `lerobot/pi0_base` on expert demonstrations with the safety loss active.

**Stage 2: DAgger**
Roll out the current policy, relabel visited states with the expert (privileged state → labels stay valid even where the policy misbehaves), retrain. DAgger exposes the policy to its own out-of-distribution failure states, which is exactly where the safety term matters most.

---

## Iteration History

### v1 — Initial pipeline (10 cubes, single camera)
- First working sim + expert + BC pipeline
- Single agentview camera only
- Expert ~95% success

### v2 — BC run 1: 8k steps, single camera
- Dataset: `data/safe_cube_v2`, 700 episodes, 124k frames
- Result: 0/20 success. Policy reached the cube and pantomimed pick→carry→place but never triggered the grasp (3 cm too coarse). Red contact: 0 (safety loss worked). Identified as undertrained (loss still descending at 8k).
- **Rollout evaluation gotcha discovered:** `PolicyRollout.act()` calls `policy.reset()` every step (designed for DAgger relabeling), which re-samples flow-matching noise every step. π0 never executes step 2+ of its 50-step chunk → gripper close (a deferred mid-chunk event) is never executed. Fix: use `act_queued` with a per-episode reset (`--action-chunking` flag in `record_policy_rollout.py`).

### v3 — BC run 2: 20k steps, dual camera + anti-fly-over loss
- New: wrist camera added as second policy input (eye-in-hand view for close-up grasp feedback)
- New: lateral-XY SDF loss (treats cubes as infinite vertical prisms — lifting over them doesn't help, must weave around) + grasp-gated height ceiling loss
- Dataset: `data/safe_cube_v3`, 1500 dual-camera episodes
- Result (20k checkpoint): grasp 0% → **85%** (wrist camera fixed the grasp). Ceiling violations: **0** (anti-fly-over loss worked). Task success: still 0% — policy never opened the gripper to place.
- Continuation to 25k: scalar red-contact appeared to worsen (25% → 55%), but video review revealed this was **better**, not worse. The 20k model avoids the obstacle field entirely; the 25k model drives *into* the field toward the goal and clips reds because it hasn't learned precise weaving yet. Higher red contact = more real task engagement. Decision: **DAgger should build on 25k**, not 20k.
- Root cause of persistent 0% success still not fully understood at this point.

### Critical bug: black frame dataset corruption
- **Discovery:** ~60–68% of all frames in the old datasets (`safe_cube_v3` BC base, `safe_cube_dagger`) were stored as pure black on disk (YAVG=16, confirmed at the raw file level with ffmpeg). Both cameras, whole episodes.
- **Root cause:** `MUJOCO_GL` was unset and `DISPLAY=:1` was present (the AnyDesk virtual display). MuJoCo bound a display-backed GLX context to the AnyDesk framebuffer. When the remote session disconnected or slept, the framebuffer powered down and `renderer.render()` returned black. The pattern was interleaved (tracking connect/disconnect cycles), not a clean cliff, which made it hard to notice.
- **Fix:** `sim/env.py` now calls `os.environ.setdefault("MUJOCO_GL", "egl")` before `import mujoco`. All collection runs use `env -u DISPLAY MUJOCO_GL=egl`. EGL renders headlessly on the GPU, immune to any X display state.
- **Impact:** This was the real cause of persistent 0% task success — not undertraining. All pre-fix checkpoints and datasets deleted as untrustworthy.

### v5 — Clean dataset + BC v6 (current)
- Dataset: `data/safe_cube_v5`, 1600 attempts → ~1500 clean success-only episodes, 8 red cubes, EGL-rendered, non-black verified.
- BC training: `outputs/safe_pi0_bc_v6/`, 20k steps, batch 48, gradient checkpointing, ~21h at ~5 s/step, ~77 GB VRAM.
- **Result: 75% rollout success (6/8), 0 ceiling violations, 0 fly-overs** (action-chunked eval, held-out seeds 2000–2007). First meaningful task success.

### DAgger Round 1 (in progress as of 2026-06-06)
- Dataset: `data/safe_cube_dagger`, 750 episodes, 330k frames, seeds 4000+, alpha=0.5 (50% policy / 50% expert mixing)
- Training: `outputs/safe_pi0_dagger_r1/`, 20k steps, starting from BC v6 checkpoint
- WandB: `safe_pi0_dagger_r1` run in `safe-cube` project

---

## Current State

| Metric | BC v6 (20k steps) |
|---|---|
| Task success | 75% (6/8) |
| Ceiling violations | 0 |
| Fly-overs | 0 |
| Eval seeds | 2000–2007 (held out from training) |

DAgger round 1 is training. The expectation is that DAgger will improve precision in the weaving phase (the policy's own rollout trajectories surface near-violation states that BC never saw, and expert relabeling gives corrective labels for them).

---

## Infrastructure Notes

- **Hardware:** RTX PRO 6000 Blackwell, 96 GB VRAM. Box reboots under sustained GPU load unless power-capped (user manages caps at 400W + 90% fan). All long runs are kept resumable.
- **VRAM target:** 75–85 GB. Batch 48 + gradient checkpointing hits ~77 GB at ~5 s/step.
- **Checkpoints:** ~22 GB each. `save_freq=1000–2000` controls disk usage.
- **Pretrained base:** `lerobot/pi0_base` (~14 GB), cached at `~/.cache/huggingface/hub/`.
- **All datasets and checkpoints are gitignored** (regeneratable). Only code is in git.
- **Eval seeds** must be ≥ collection N to avoid evaluating on training layouts.
