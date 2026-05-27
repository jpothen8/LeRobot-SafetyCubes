# Constraint-Aware VLA for Obstacle-Avoiding Cube Manipulation — Project Summary

## 1. Task

Train a vision-language-action (VLA) manipulation policy to:

**Pick up a blue cube, carry it low through a scattered field of red cubes, and drop it in a goal zone on the other side — without touching any red cube.**

The red cubes are obstacles. The manipulator must stay low and weave between them (no lifting up and over). Success means the blue cube is delivered to the drop-off zone with no red-cube contact; failure means a red cube was knocked (or the blue cube was dropped/missed).

The core requirement is that **collision avoidance is baked into the policy weights during training**, not enforced by a post-hoc runtime filter.

---

## 2. Core Idea: Safety as Part of the Training Loss

A safety filter can be written as **Bayesian MAP inference**:

```
p(u | safe, x)  ∝  p(u | x) · p(safe | u, x)^λ
   posterior        policy       safety
                    prior        likelihood
```

- `p(u | x)` — the policy's preferred action distribution (the **prior**).
- `p(safe | u, x)` — probability that action `u` in state `x` is safe (the **likelihood**).
- `λ` — a weight trading task fidelity against safety.
- `p(u | safe, x)` — the safe action distribution (the **posterior**).

Taking negative logs turns "pick the MAP safe action" into a two-term loss, which is used as a **training objective** rather than a runtime optimization. The filter lives in the gradients, not in an online solve. The constraint-aware loss has the form:

```
L = L_task        (imitation / flow-matching term — learn the demonstrated behavior)
  + λ · L_safety  (penalize actions whose predicted result enters a red cube)
```

The task splits cleanly along these two terms:

- **Blue cube interaction → task → imitation loss.** Grasping and placing the blue cube is the demonstrated behavior; contact with the blue cube is *good* contact.
- **Red cube interaction → constraint → safety loss.** Touching a red cube is the thing to avoid; it enters as a negative-clearance penalty.

---

## 3. Key Concepts

### 3.1 Behavior Cloning (BC) and the BC checkpoint
BC is plain supervised imitation: regress the policy output onto the demonstrated action given the observation. A **BC-trained checkpoint** is the saved weights after this pass — a warm start so the later DAgger rollouts begin from a competent policy rather than noise. For a foundation VLA, the BC checkpoint is the model after a BC fine-tune on the task demos.

### 3.2 DAgger (Dataset Aggregation)
BC suffers **covariate shift**: trained on the demonstration state distribution, the policy drifts into unseen states at deployment and compounds errors. DAgger trains on the **policy's own** state distribution while still labeling with expert actions:
1. Train initial policy (BC).
2. Roll out the policy; log visited states.
3. Query the expert for the correct action at those states.
4. Aggregate into the dataset; retrain.
5. Repeat.

Smooth mixing `π_collect = α^j·π_β + (1−α^j)·π_θ` slides from expert-heavy (collect demonstrations) to policy-heavy (collect deployment-distribution data). Early policy rollouts also naturally produce the **failure data** the safety term needs.

### 3.3 How safety gets baked into the weights
The weights determine the action output for every observation. The full training forward pass is:

```
image → π_θ → action chunk → FK → ee trajectory → SDF → clearance → σ → p_safe → −log → L_safety
```

Every operation is differentiable, so `L_safety`'s gradient flows back through the policy. The gradient at the weights is "the change that would have made this action safer for this observation." Over many steps, the policy's action distribution near red cubes is biased away from collision — not because it was programmed, but because gradient descent pushed it there. A runtime filter never updates the weights, so the policy never learns the obstacles exist.

DAgger and the constraint-aware loss compose: DAgger ensures training on the **right states**; the safety loss ensures the **right gradients** at those states.

---

## 4. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Safety likelihood | **Analytic SDF**, not a learned classifier | Obstacle layout is per-episode; a state-only learned classifier can't capture it. The SDF takes cube positions as parameters and is differentiable for free. |
| Localization | **No fiducials.** Sim provides ground-truth cube positions | Privileged training: positions enter only the loss, never the policy input. |
| Observation | Policy sees **images (+ proprio) only** | Asymmetric / privileged setup; deployable from pixels. |
| Held blue cube in SDF | **Ignored** (negligible offset) | Cube is small; the gripper bounding sphere absorbs it. |
| Motion constraint | **Must stay low and weave between cubes** | Makes the avoidance constraint do real work (no trivial lift-over). |
| Policy class | **Flow matching (π0 in LeRobot)** | Handles the multimodality of many valid paths around obstacles, which MSE-BC averages into collisions. |
| Safety attachment to flow | **Endpoint estimate** of the action chunk | Single velocity pass; no full ODE backprop, no inference-time guidance (which would be post-hoc). |
| Dynamics model | **None — use differentiable FK** | In sim with known cube positions, the ee pose is an exact differentiable function of the action via forward kinematics. No learned dynamics, no approximation error. |

---

## 5. Architecture

```
                 ┌─────────────────────────── privileged (sim only) ───────────────┐
                 │                                                                  │
   image y ──► π_θ (π0 flow-matching VLA) ──► action chunk â (endpoint estimate)    │
                 │                                   │                              │
                 │                                   ▼                              │
                 │                          differentiable FK ──► ee trajectory     │
                 │                                   │                              │
   cube positions (ground truth) ──────────────────►├──► box SDF ──► clearance      │
                 │                                   │                              │
                 │                                   ▼                              │
                 │                          L_safety = −log σ(α·(clearance−margin)) │
                 └──────────────────────────────────────────────────────────────────┘

   Total training loss:  L = L_flow_matching  +  λ · L_safety

   At deployment:  image y ──► π_θ ──► action.   (No SDF, no cube positions, no filter.)
```

### 5.1 The flow-matching wrinkle
A flow-matching policy learns a **velocity field** `v_θ(a_τ, τ, obs)`, not an action. Training regresses that velocity against the demo interpolant at random flow-times `τ`; the action only materializes at inference via an ODE solve. So the **task term is just the standard flow-matching loss** (no change), but the safety term has no action to attach to. Resolution: the **endpoint estimate**.

With the rectified-flow convention `τ=1 → data, τ=0 → noise`:
```
a_τ      = τ·a_1 + (1−τ)·a_0          (interpolant; a_0 = noise, a_1 = demo action)
target v = a_1 − a_0
â_1      = a_τ + (1−τ)·v_θ            (endpoint identity — the chunk we penalize)
```
This is one extra forward pass, reusing the same `v_θ` that produces the flow-matching loss. The estimate is accurate near `τ=1` and noisy near `τ=0`, so the safety loss is **weighted by τ** (or sampled from a high-τ band).

### 5.2 Action chunking is an asset
π0 predicts an **action chunk** (default 50 steps), so `â_1` is a short trajectory. Rolling it through FK gives a **predicted path**, and the safety penalty is applied at **every point** in the chunk (min clearance over the trajectory). This gives automatic multi-step lookahead — the policy is punished for plans that *will* clip a cube several steps ahead — which is exactly right for "stay low and weave."

### 5.3 SDF and safety loss
Axis-aligned box SDF (positive outside / safe, negative inside / collision):
```
d = |p − center| − half
sdf = ‖max(d, 0)‖ + min(max(d_x, d_y, d_z), 0)
clearance = min_over_cubes(sdf) − ee_radius
p_safe = σ(α·(clearance − margin))
L_safety = mean( τ · (−log p_safe) )   over all chunk points and batch
```

---

## 6. Implementation

Delivered as `safe_pi0_policy.py` — a `SafePI0Policy(PI0Policy)` subclass for LeRobot.

- **Reimplements** π0's flow-matching loss in `forward` (rather than calling `super().forward()`) so a single velocity prediction `v` feeds both the task loss and the endpoint estimate — avoids encoding the images through the 3B VLM twice per step.
- **Fully concrete & unit-testable in isolation:** `box_sdf`, `_sdf_clearance`, `_fk_chunk` (differentiable FK via `pytorch_kinematics`), `_safety_loss`.
- **Two version-specific hooks** marked `WIRE THIS IN`:
  1. `_flow_velocity` — bind to your LeRobot π0's velocity-field call (encode obs prefix + run action expert on `(a_t, t)`, return the velocity tensor).
  2. `_fk_chunk` — match your action space (delta vs. absolute joint targets; π0's `use_relative_actions`). If actions are in ee/task space, skip FK entirely.
- **Privileged inputs** (`cube_positions`, `cube_half_extents`) ride in the batch as extra observation keys the policy ignores and only the loss reads.

LeRobot integration is a single subclass + overridden `forward`; the training script, optimizer, dataloader, and fine-tuning from `pi0_base` stay stock.

---

## 7. Deployment Workflow

### 7.1 Data
The method **requires both success and failure data** (the safety term needs a negative class). Generate failures via noisy-expert rollouts and/or DAgger mixing (early policy fails naturally). Plan ~50–200 successes plus comparable failures, expanded by DAgger.

### 7.2 Sim-first pipeline
1. **Build the sim** (MuJoCo / Isaac Sim): manipulator, scattered red cubes, blue cube, goal zone; camera matching the intended real view; domain randomization (lighting, textures, cube layout, camera pose, image noise). Cube positions are the privileged state used only by the loss.
2. **Train in sim:** BC warm start → run the constraint-aware DAgger loop → iterate on `λ`, `α`, `margin`. Early-stop at peak rollout success (the evolving safety term makes the loss landscape non-stationary).
3. **(Optional) real-world fine-tune:** small real teleop set + short fine-tune (LoRA on the VLA backbone), if going to hardware.

### 7.3 Runtime (deployment)
1. Camera captures image.
2. VLA forward pass: image → action chunk.
3. Send to the arm.

No online optimization, no projection — identical to a vanilla VLA deployment, because the safety preference is in the weights.

### 7.4 Recommended runtime safety net (hardware)
Constraint-aware training gives **soft** safety, not a guarantee. Keep a thin hard-stop watchdog: compute clearance online and zero the command if the predicted next pose falls below a small clearance (e.g. 5 mm). A few lines, microseconds, protects the hardware on rare failures. Final stack: **trained-in safety (main) + thin watchdog (insurance).**

---

## 8. Hyperparameters & Gotchas

- **`λ` (safety weight)** — the central knob. Too small → safety ignored; too large → the policy collapses to pure avoidance and stops doing the task. Sweep first.
- **`α` (sigmoid sharpness)** ≈ 50 (units of 1/m); **`margin`** ≈ 2 cm; **`ee_radius`** ≈ 3 cm (gripper bounding sphere).
- **Rectified-flow convention is the silent killer.** If LeRobot's π0 defines flow time the opposite way (`τ=0 → data`), the endpoint formula is wrong and the safety gradient quietly points in a nonsense direction while training still *looks* fine. Match the interpolant **and** the endpoint line to π0's actual loss code.
- **Multimodality** — many valid paths around the field; MSE-BC averages opposite detours into a straight-through collision. This is why flow matching (multimodal) was chosen; if you ever fall back to an MSE head, expect this failure.
- **Compute** — π0 is 3B params. Feasible to fine-tune with the extra loss on a single 24GB+ GPU with bf16 + gradient checkpointing, but not free. **Prototype the loss plumbing on a smaller policy (Diffusion Policy / SmolVLA) first**, then swap to π0.
- **Test geometry before the big run** — `box_sdf`, `_sdf_clearance`, `_fk_chunk` run standalone. Unit-test them against hand-placed cubes and a known arm config so a sign error in geometry isn't discovered 40 epochs into a 3B fine-tune.
- **Pin down "stay low" enforcement** — height ceiling vs. tall cubes vs. near-planar carry. This determines whether the safety term is genuinely load-bearing or just garnish; decide before building the sim.

---

## 9. Why This Beats Post-Hoc Filtering

A runtime safety filter modifies actions the VLA was never trained to produce, pushing the next observation off-distribution; the VLA then outputs worse actions and the filter must intervene harder — a compounding loop. Integrating safety into the **training loss** lets the policy learn the constraint boundary in its weights, so deployed actions are already biased away from obstacles: no runtime filter, no compounding distribution shift, and the safety gradient actually teaches the policy that obstacles exist.

---

## 10. Artifacts & References

- **Code:** `safe_pi0_policy.py` — `SafePI0Policy` LeRobot subclass.
- **Method paper:** Cao, Joa, Borrelli, *A Simple Approach to Constraint-Aware Imitation Learning with Application to Autonomous Racing*, arXiv:2503.07737. Repo: `github.com/CadenzaCoda/ConstraintAwareIL`.
- **Policy backbone:** π0 flow-matching VLA in LeRobot (`--policy.type=pi0`, fine-tune from `lerobot/pi0_base`); adapted from Physical Intelligence's OpenPI.
- **Related (the approach being improved on):** CoDiG and other inference-time diffusion/flow guidance — runtime filters, cited as the alternative this project avoids.
