# OpenManipulator-X: Digital Twin, Torque Control & Trajectory Shaping

**Project report — architecture, mathematics, and experimental record**
Working directory: `New folder` · Target hardware: ROBOTIS OpenManipulator-X ("Chain"), OpenCR 1.5.3 controller, 4× Dynamixel servos (IDs 11–14)

---

## 1. What this project is

This is a from-scratch Python re-implementation of ROBOTIS's Processing/OpenCR control stack for the OpenManipulator-X, built up in five stages of increasing control sophistication:

1. **`main.py`** — a faithful port of the stock Processing sketch: live 3D digital twin (real pose vs. commanded "ghost" pose) talking to the *unmodified* stock firmware over its existing position-control serial protocol.
2. **`pd_lab.py`** — replaces position commands with a from-scratch **torque-based PD controller**, in simulation and (via new custom firmware) on real hardware.
3. **`ik_gravity_lab.py`** — adds **numerical inverse kinematics** (Cartesian XYZ targeting) and a **full coupled rigid-body dynamics model** (mass matrix, Coriolis terms, gravity compensation) instead of treating each joint as independent.
4. **`smooth_trajectory_lab.py`** — adds a **reference-trajectory generator** so the controller tracks a smoothly evolving target instead of reacting to a step input, removing the sharp jerk transient the previous stage exhibited.
5. **`path_follow_lab.py`** — extends the same computed-torque controller to track a **continuously moving Cartesian path** (a circle or rectangle) instead of a single point, re-solving IK every control tick.

Every stage keeps the same live 3D viewer and the same "real pose vs. target pose" visualization, so the five programs form a single continuous narrative: **position control → torque control → dynamics-aware Cartesian control → smooth trajectory tracking → continuous path following**, on the same physical arm, with every run's gains, targets, and telemetry logged to disk.

This directly instantiates the core problem of tele-robotic digital twins: a synchronized visual/state representation of a remote physical system, driven by a control architecture split across two clocks — a fast local loop on the robot's own controller and a slower, model-based correction computed on the host and pushed over a serial link. Section 2 makes that split explicit.

---

## 2. System architecture: the sync loop

Every program shares four building blocks (`om_python/`):

| Module | Role |
|---|---|
| `state.py` | Thread-safe `SharedState` — mirrors the Processing sketch's globals: `receive_joint_angle` (real pose, from telemetry) and `ctrl_joint_angle` (commanded/ghost pose, from the panel or IK solver) |
| `renderer.py` | GLFW/OpenGL 3D view, own thread — draws the real pose (grey) and ghost pose (light) simultaneously every frame, reading `SharedState` under a lock |
| `serial_link.py` / `torque_link.py` | Serial I/O, own thread — parses incoming telemetry lines and writes `receive_joint_angle` under the same lock |
| `*_panel.py` | DearPyGui control window, main thread — owns Kp/Kd, targets, run/record logic |

Three threads, one shared, locked state object — this **is** the digital-twin sync mechanism: the visualization is never driving the hardware or vice versa, it's just continuously reading whatever the state object currently holds, at whatever rate each producer updates it.

### The multi-rate control split (`ik_gravity_lab.py`, `smooth_trajectory_lab.py`, hardware mode)

The stock and custom firmware both run their fast feedback loop **locally on the OpenCR**, because serial round-trip latency is too slow for a PC-side loop to close a stiff position/velocity loop. Everything model-based and slow-changing is computed on the PC and pushed down asynchronously:

```
┌─────────────────────────── PC (Python) ───────────────────────────┐
│  renderer thread        panel/main thread       serial thread      │
│  (draw @ ~60 FPS)        (Kp/Kd, targets)        (parse telemetry)  │
│         │                      │                       │           │
│         └──────────────  SharedState (locked)  ─────────┘           │
│                                                                      │
│  gravity thread (20 Hz): G(q) ← rigid_body_dynamics.gravity_vector  │
│  traj. thread   (50 Hz): q_ref ← ReferenceTrajectory.step (ODE)     │
└──────────────────────────────┬───────────────────────────────────┬──┘
                    "gravity,g1..g4"                    "target,j1..j4"
                    "state,t,j..,v..,i.." (50 Hz)  ↑             │
                                                    │             ▼
┌───────────────────────────  OpenCR firmware  ───────────────────────┐
│  100 Hz local loop:  current_i = Kp_i·e_i + Kd_i·ė_i + gravity_ff_i  │
│  gravity_ff decays to 0 if not refreshed within 500 ms               │
└───────────────────────────────────────────────────────────────────┘
```

Four independent rates coexist: **100 Hz** firmware PD, **50 Hz** telemetry uplink, **50 Hz** reference-trajectory downlink (smooth-trajectory hardware mode only), **20 Hz** gravity-compensation downlink — chosen because gravity only depends on slowly-changing position, so it doesn't need the fast loop's bandwidth. The 500 ms gravity-decay timeout means a PC-side crash or disconnect can never leave a stale torque offset being applied indefinitely — the failure mode is "gravity comp silently turns off," not "arm keeps pushing."

---

## 3. Forward kinematics (`om_python/kinematics.py`)

The arm is a 4-DOF serial chain: waist (yaw, joint 1), shoulder (pitch, joint 2), elbow (pitch, joint 3), wrist (pitch, joint 4), with fixed offsets between joint origins taken directly from the Processing sketch's `translate()` calls (already in mm):

```
LINK2_OFFSET  = (0,   0,  36)      LINK5_OFFSET  = (124, 0,   0)
LINK3_OFFSET  = (0,   0,  40)      WRIST_OFFSET  = (130, 14,  0)
LINK4_OFFSET  = (24,  0, 128)      SPHERE_OFFSET = (0,  -7,   0)
```

The gripper-center position is the composition of homogeneous transforms:

```
T(q) = T(-model_trans)
     · T(LINK2)  · Rz(-θ1)
     · T(LINK3)  · Ry(θ2)
     · T(LINK4)  · Ry(θ3)
     · T(LINK5)  · Ry(θ4)
     · T(WRIST) · T(SPHERE)

gripper_center(q) = ( T(q) · [0,0,0,1]ᵀ )[:3] · (1 + model_scale)
```

`joint_positions(q)` returns the origin of each joint's own rotation axis — each entry only depends on the *preceding* joints, since rotating a joint moves everything distal to it, never its own axis. `link_com_positions(q)` approximates each link's center of mass as the midpoint between consecutive chain points (joint *i*, joint *i*+1, or the gripper tip for the last link) — a documented placeholder, since no mesh-based centroid or datasheet mass distribution was available.

**Why not a closed-form solution:** link 4's offset `(24, 0, 128)` is not axis-aligned, so a hand-derived closed-form IK is easy to get subtly wrong. Everything downstream (IK, dynamics) instead differentiates this same forward-kinematics function numerically, which is slower per call but structurally can't be wrong in the way a hand-derived formula can.

---

## 4. Numerical inverse kinematics (`om_python/inverse_kinematics.py`)

**Problem:** given target XYZ (mm), find joint angles `q ∈ ℝ⁴` such that `gripper_center(q) = target`. This is a 3-constraint, 4-unknown system — the arm is kinematically redundant for a pure position target, so there is a one-parameter family of exact solutions (before joint limits), not a unique one.

**Method — damped least squares (Levenberg–Marquardt style):**

Numerical Jacobian (forward differences, `ε = 1e-4`):

```
J[:, i] = ( gripper_center(q + ε·eᵢ) − gripper_center(q) ) / ε         J ∈ ℝ^{3×4}
```

Update step, solving in the smaller 3×3 space rather than the singular 4×4 `JᵀJ`:

```
e  = target − gripper_center(q)
Δq = Jᵀ (J Jᵀ + λ²I₃)⁻¹ e                    λ = 8   (damping)
q ← clamp( q + Δq, joint limits )
```

Iterated up to 200 times, stopping once `‖e‖ < 0.5 mm`. The damping term `λ²I₃` is what makes this well-behaved near singularities (where `JJᵀ` would otherwise be near-singular) and lets the solver report *failure with a residual* for unreachable targets instead of diverging — the returned pose is "the closest reachable point," and the caller (the panel UI) surfaces both the reachability flag and the residual error in mm.

**Joint-limit clamping.** Every caller through §7 solves IK once per run from a neutral initial guess, so a subtlety here never mattered until §8's path-following controller started re-solving IK every 10 ms, warm-started from the previous tick's answer: if a joint is already sitting at its limit and the computed step would push it further past that limit, the *next* line's `clamp_to_joint_limits` silently undoes that joint's share of the step — but the *other* joints' share of the same (now-inconsistent) step is not undone, so they can walk away from the target instead of compensating for it, and the residual grows without bound instead of settling at "closest reachable point." The fix, a standard joint-limit-clamping correction (Buss): before taking the step, zero the Jacobian column of any joint that is at its limit and being pushed further past it, and re-solve for the remaining free joints. See §8 for the measured before/after.

---

## 5. Simplified decoupled PD model (`pd_lab.py`, `om_python/dynamics.py`)

The first control stage treats each joint as an **independent** 2nd-order rotational system — no coupling, no gravity:

```
τᵢ = Kpᵢ·(θ_target,ᵢ − θᵢ) + Kdᵢ·(0 − θ̇ᵢ)
I·θ̈ᵢ = τᵢ − b·θ̇ᵢ
```

with placeholder `I = 0.05 kg·m²`, `b = 0.4 N·m·s/rad`, integrated with explicit Euler.

**Closed-form check on the default gains.** Substituting the PD law into the plant gives a standard linear 2nd-order system per joint:

```
I·θ̈ + (b + Kd)·θ̇ + Kp·θ = Kp·θ_target
```

with natural frequency and damping ratio

```
ωₙ = √(Kp / I)              ζ = (b + Kd) / (2√(Kp·I))
```

At the shipped defaults `Kp = 5`, `Kd = 0.6`:

```
ωₙ = √(5 / 0.05) = 10 rad/s
ζ  = (0.4 + 0.6) / (2·√(5·0.05)) = 1.0 / (2·0.5) = 1.0
```

**ζ = 1.0 exactly** — the default gains land on the critically-damped boundary (fastest settling with no overshoot), not by an explicit design step but as a consequence of the chosen placeholder numbers. This is a useful diagnostic for tuning: raising `Kp` without a matching increase in `Kd` pushes the joint underdamped (overshoot); raising `Kd` alone pushes it overdamped (sluggish).

This model has no gravity or inter-joint coupling term, so on real hardware (where gravity is real) it exhibits steady-state droop under load — visible in saved plots as a torque trace that doesn't decay to zero at rest. That's the motivation for stage 3.

---

## 6. Full rigid-body dynamics + computed-torque control (`ik_gravity_lab.py`, `om_python/rigid_body_dynamics.py`)

Each of the 4 moving links is modeled as a **point mass at its COM** (`LINK_MASS = 0.15 kg` each, approximate — no local URDF/datasheet was available), plus a per-joint rotor/local inertia term (`0.004 kg·m²` each) added to the mass matrix's diagonal so it stays well-conditioned even when a link's COM sits near its own rotation axis. All dynamics quantities are derived **numerically from the same forward-kinematics chain**, not a hand-derived closed form — same philosophy as the IK.

**Kinetic energy → mass matrix.** With `Jₗᵢₙₖ(q) = ∂COM_link/∂q` (numerical, forward differences):

```
T(q, q̇) = ½ q̇ᵀ M(q) q̇                M(q) = Σₗᵢₙₖ mₗᵢₙₖ · Jₗᵢₙₖᵀ Jₗᵢₙₖ  +  diag(I_rotor)
```

**Potential energy → gravity vector** (central differences, `ε = 1e-4`):

```
U(q)   = Σₗᵢₙₖ mₗᵢₙₖ · g · z_link(q)              (z = height of that link's COM)
G(q)ᵢ  = ∂U/∂qᵢ  ≈  ( U(q + ε·eᵢ) − U(q − ε·eᵢ) ) / 2ε
```

**Coriolis/centripetal matrix**, via Christoffel symbols of the first kind (derivatives of `M(q)` also by central differences):

```
Cᵢⱼ(q, q̇) = Σₖ ½·( ∂Mᵢⱼ/∂qₖ + ∂Mᵢₖ/∂qⱼ − ∂Mⱼₖ/∂qᵢ ) · q̇ₖ
```

giving the standard manipulator equation of motion `M(q)q̈ + C(q,q̇)q̇ + G(q) = τ`.

**Control law actually implemented** (point regulation when `target_velocity`/`target_accel` are omitted; trajectory tracking when supplied — see §7):

```
e      = q_target − q
ė      = q̇_target − q̇
pd     = Kp∘e + Kd∘ė                        (∘ = elementwise, per-joint gains)
τ      = M(q)·q̈_target + pd + C(q,q̇)·q̇ + G(q)
```

The simulated plant is then integrated by solving for the resulting acceleration and stepping explicit Euler:

```
q̈ = M(q)⁻¹( τ − C(q,q̇)q̇ − G(q) )  =  q̈_target + M(q)⁻¹·pd
q̇ ← q̇ + q̈·dt          q ← q + q̇·dt
```

with `q̈` hard-clamped to `±500 rad/s²` (`MAX_ANGULAR_ACCEL`) as a safety net — this model's real inertia (~0.005–0.02 kg·m² once rotor terms are included) is roughly **10× smaller** than the placeholder model in §5, so gains that were stable there (e.g. `Kp = 5`) are wildly too aggressive here; explicit-Euler integration of the stiffer system diverges to NaN in ~30 steps without the clamp. Defaults were rescaled accordingly: `Kp = 0.6`, `Kd = 0.15` (simulate mode).

**A precise note on "decoupled."** The code comments describe this as producing "a clean, Kp/Kd-tunable linear system regardless of pose." Working through the algebra above shows the closed-loop acceleration is `q̈ = q̈_target + M(q)⁻¹·pd`, *not* `q̈ = q̈_target + pd`. Exact feedback linearization (fully pose-independent per-joint response) would require premultiplying the PD term by `M(q)` as well — `τ = M(q)·(q̈_target + pd) + C(q,q̇)q̇ + G(q)` — so that `M(q)⁻¹` cancels exactly. As implemented, the effective stiffness/damping seen by each joint is shaped by `M(q)⁻¹`, which is close to diagonal for this point-mass approximation but not exactly so — meaning the response is *approximately* decoupled, and genuinely pose-dependent in principle, even though it looks clean in the recorded plots because the off-diagonal terms are small for this arm's mass distribution. Worth keeping in mind if this control law is reused on a heavier or more coupled mechanism.

**Gravity compensation on real hardware** can't run this whole computation on the microcontroller, so it's split: the firmware runs a plain local PD loop (100 Hz — `current_mA = Kp·e + Kd·ė + gravity_ff`) and the PC computes `G(q)` from the live telemetry at 20 Hz, scales it by a tunable **`mA per N·m`** constant (default 500 — there's no verified torque constant for these servos, so this is empirically tuned: raise it if the arm still droops, lower it if it overshoots at rest), and streams it down as a `gravity,g1,g2,g3,g4` feedforward that decays to zero after 500 ms without a refresh.

---

## 7. Reference-trajectory smoothing (`smooth_trajectory_lab.py`, `om_python/trajectory.py`)

In §6, clicking "Run" hands the controller the IK solution directly — the very first control tick sees the *entire* position error at once (a step input), which is what produces the sharp initial velocity/jerk spike visible in those plots even though the rest of the motion is well-behaved.

**Fix:** integrate a **critically-damped second-order reference model** every control tick (10 ms) instead of jumping to the goal:

```
q̈_ref = ωₙ²·(q_goal − q_ref) − 2ζωₙ·q̇_ref              ωₙ = 4 / T_s,   ζ = 1
q̇_ref ← q̇_ref + q̈_ref·dt
q_ref  ← q_ref  + q̇_ref·dt
```

`T_s` ("smoothing time," seconds) sets the pace — roughly the 4-time-constant settling time of a critically damped 2nd-order system with this `ωₙ`. This produces a continuously evolving `(q_ref, q̇_ref, q̈_ref)` that the computed-torque controller from §6 tracks by plugging `q̈_ref` in as `q̈_target` and `q̇_ref` as `q̇_target` — i.e. this is exactly the trajectory-tracking form of the control law in §6, now with genuinely nonzero feedforward velocity/acceleration terms instead of the default zero. The reference trajectory itself is virtual — it costs nothing physically, it's numbers evolving in a Python loop, decoupled from the arm's own mass-matrix dynamics.

**Measured effect** (same target and gains, smoothing on vs. off):

| Metric | Step input (§6) | Smoothed reference (§7) | Change |
|---|---|---|---|
| Peak jerk | ~8000 rad/s³ | ~370 rad/s³ | **>20× lower** |
| Peak joint speed | ~170 mm/s | ~33 mm/s | ~5× lower |
| Final accuracy vs. IK solution | — | within 0.02 mm | unchanged |

On real hardware, the firmware still only ever tracks a fixed setpoint — there's no trajectory concept on the microcontroller. Smoothing comes entirely from a second PC-side background thread that re-integrates the reference ODE against wall-clock time and streams the moving `q_ref` down as the firmware's `target` at 50 Hz, i.e. the PC continuously "leads" the setpoint along the curve.

---

## 8. Feedback-linearization path following (`path_follow_lab.py`, `om_python/paths.py`)

A fifth control mode: instead of driving to a single fixed target (§6) or smoothly approaching one (§7), the controller continuously tracks a moving Cartesian target that traces a **circle or rectangle** in a chosen plane. This reuses the *exact same* feedback-linearization/computed-torque controller from §6 — no new control law — only the reference fed into it changes.

**Reference generation (`paths.py`)** — two pure, stateless parametric paths (mm), as a function of time `t` and loop `period`:

```
circle_path(t) = center + radius · (cos(2π·(t mod period)/period), sin(...))     (in the chosen plane)
square_path(t) = constant-speed traversal of a square's perimeter, corners at t = 0, T/4, T/2, 3T/4
```

Each control tick: (1) sample the path at the current time to get an instantaneous Cartesian point; (2) solve IK for that point (§4), **warm-started from the previous tick's solution** so it converges in a handful of iterations instead of two hundred; (3) run that joint-space point through the same critically-damped reference filter from §7 — this is what rounds the square path's corners, since the filter physically cannot track an instantaneous change in velocity direction; (4) hand the filtered `(q_ref, q̇_ref, q̈_ref)` to the same `RigidBodyDynamics.step()` used everywhere else.

**A bug the other labs never triggered.** Every other lab calls `inverse_kinematics.solve()` once per run from a neutral initial guess. Continuously re-solving IK along a moving path, warm-started every 10 ms, exposed the joint-limit defect described in §4: on an 80 mm-radius test circle the residual grew unbounded to over 140 mm once a joint hit its limit, with no warning. After the fix, the same circle's worst-case residual is bounded at ~36 mm (the genuinely-unreachable portion of that circle correctly reports "not reached" instead of diverging); shrinking the default path to 60 mm radius/side at the same center keeps it fully within the reachable workspace across multiple loops, with residuals under 0.5 mm throughout.

**Measured trade-off: smoothing time vs. tracking fidelity vs. jerk.** Because the goal is now *persistently moving* rather than static, the reference filter's `smoothing_time` (§7) plays a different role than it does for a single point: it behaves like a low-pass corner frequency tracking a moving signal, not a one-time settling duration. Headless simulation of a 60 mm-radius circle (3 loops, 6 s/loop) shows the trade-off directly:

| `smoothing_time` | Steady-state tracking RMS error | Peak commanded jerk |
|---|---|---|
| 1.0 s | 28.5 mm | ~1,100 rad/s³ |
| **0.4 s (shipped default)** | **11.2 mm** | **~6,900 rad/s³** |
| 0.25 s | 6.6 mm | ~17,700 rad/s³ |
| 0.1 s | 1.9 mm | ~52,200 rad/s³ |

Smaller `smoothing_time` tracks the path more tightly but drives commanded jerk up sharply as the reference filter's bandwidth approaches the safety-clamped acceleration limit (§6). `0.4 s` was chosen as the shipped default because it keeps jerk roughly in the same range as the un-smoothed step response documented in §7, while still tracing a recognizable circle.

**Validation status.** The control loop (IK + reference filter + computed torque) has been validated headlessly in simulation — bounded, NaN-free, with the trade-off above measured directly — and the DearPyGui panel has been confirmed to launch and render correctly with these defaults. It has not yet been run to completion on real hardware; the hardware path (a background thread mirroring §7's trajectory-streaming thread, re-solving IK and pushing `target` at 50 Hz) reuses the same firmware and gravity-compensation bridge as §6/§7 unchanged. History is `path_gain_history.json`, storing shape, plane, center, size, period, loops, and smoothing time alongside gains.

---

## 9. Hardware bridge: custom firmware (`firmware/open_manipulator_torque_pd/open_manipulator_torque_pd.ino`)

The stock firmware's `JointDynamixel::setOperatingMode()` has no path to current/torque control, so real torque control needed new firmware talking to `DynamixelWorkbench` directly (bypassing the OpenManipulator library entirely), putting joints 11–14 into Current Control Mode.

**Wire protocol (57600 baud):**
```
PC → OpenCR:  gains,kp1,kp2,kp3,kp4,kd1,kd2,kd3,kd4   (mA/rad, mA/(rad/s))
              target,j1,j2,j3,j4                       (rad)
              gravity,g1,g2,g3,g4                       (mA feedforward, decays after 500 ms)
              torque,on | torque,off
OpenCR → PC:  state,t,j1..j4,v1..v4,i1..i4              (~50 Hz)
```

**Local control loop (100 Hz):**
```
error_dot = 0 − present_velocity
current_mA = Kp·error + Kd·error_dot + gravity_ff        (gravity_ff = 0 if stale > 500 ms)
current_mA = clamp(current_mA, ±400 mA)                  (MAX_CURRENT_MA)
```

**Safety, enforced in firmware regardless of what Python sends:**
- Boots and disarms into **position mode holding the present angle** — never snaps to zero, never goes limp.
- `MAX_CURRENT_MA = 400 mA` clamps commanded current in software *and* is written to the servo's own `Current_Limit` register at boot (belt-and-suspenders).
- Target angles are clamped to the real joint limits before use.
- `torque,off` (or a fresh boot) returns to position-mode holding the then-current angle.
- The Python side additionally requires **proof of life** before allowing a run: either the one-time `torque_pd_ready` boot line, or a successfully parsed `state,...` line (needed because OpenCR's native USB doesn't reset the board on reconnect, so the boot line won't reappear on a simple reconnect) — within a 5 s timeout, else the boot log is shown instead of silently pretending the connection worked.
- **EMERGENCY TORQUE OFF** is always available once in hardware mode and immediately disarms every joint.

**Bugs found and fixed while bringing this up on real hardware:**
1. ROBOTIS's own `DynamixelWorkbench::getPresentVelocityData()` (bundled with OpenCR 1.5.3) actually reads the `Goal_Velocity`/`Moving_Speed` register, **not** `Present_Velocity` — so the Kd term was being fed a copy of its own setpoint, not real feedback. Fixed by reading the `Present_Velocity` item directly.
2. `Connect` had no guard against a double-click, so a second click could try to reopen an already-open port (`PermissionError`/"Access is denied"). Fixed by guarding against concurrent connect attempts.
3. The very first hardware runs produced empty (header-only) telemetry simply because the firmware hadn't actually been flashed yet at the time.

---

## 10. Experiment tracking

Every run (across all four torque-based labs) appends an entry to a per-program JSON history file and saves a timestamped PNG + CSV to `plots/`:

| History file | Program | Extra fields stored |
|---|---|---|
| `pd_gain_history.json` | `pd_lab.py` | mode (simulated/hardware) |
| `ik_gain_history.json` | `ik_gravity_lab.py` | + `target_xyz` |
| `smooth_gain_history.json` | `smooth_trajectory_lab.py` | + `target_xyz`, `smoothing_time` |
| `path_gain_history.json` | `path_follow_lab.py` | + `shape`, `plane`, `center`, `size`, `period`, `loops`, `smoothing_time` |

Representative gain progression actually logged during development:

| Lab | Mode | Kp | Kd | Target XYZ (mm) | Notes |
|---|---|---|---|---|---|
| Gravity PD | simulated | `[0.6, 0.6, 0.6, 0.6]` | `[0.15, 0.15, 0.15, 0.15]` | (150.4, 0, 150) | N·m / N·m·(s/rad) scale |
| Gravity PD | hardware | `[300, 350, 350, 300]` | `[20, 25, 25, 20]` | (150, 10, 150) | mA / mA·(s/rad) scale |
| Smooth traj. | hardware | `[300, 350, 350, 300]` | `[20, 25, 25, 20]` | (150, 0, 150) | `smoothing_time = 2.0 s` |
| Smooth traj. | hardware | `[300, 350, 1000, 300]` | `[20, 25, 25, 20]` | (150, 50, 150) | joint-3 Kp raised to 1000, `T_s = 0.0` (step) |
| Smooth traj. | hardware | `[300, 350, 1000, 300]` | `[20, 25, 50, 20]` | (150, 50, 150) | joint-3 Kd raised to 50, `T_s = 0.25 s` |

Every saved plot (`plots/run_<timestamp>.png` + matching `.csv`) is a 6-panel grid — angle, position, velocity, angular velocity, torque/current, jerk, all 4 joints overlaid — plus a gridded 3D end-effector trajectory panel and a text readout of exact start/final XYZ and net displacement. In hardware mode, velocity/angular acceleration/jerk are derived from the streamed angle/velocity telemetry by finite differences (no onboard accelerometer), and "torque" is the servo's real measured current in mA, not an estimate.

---

## 11. Known approximations and limitations (consolidated)

- **Link masses (`0.15 kg` each) and rotor inertias (`0.004 kg·m²` each) are placeholders**, not datasheet or URDF values — no local spec sheet was available. They capture the right *order of magnitude* and coupling structure, not exact values.
- **Point-mass COM approximation** ignores each link's own rotational inertia about its own COM — reasonable for slender links, not exact.
- **No verified motor torque constant** — the hardware gravity feedforward's `mA per N·m` scale factor is empirically tuned, not derived from a datasheet `Kt`.
- **No integral term anywhere** — the simplified PD model (§5) shows steady-state droop under gravity on real hardware; this is an inherent limitation of PD-only control, not a bug, and is exactly what §6's gravity compensation is for.
- **Explicit-Euler integration** requires the `±500 rad/s²` acceleration clamp in the rigid-body model to stay numerically stable at aggressive gains; a bad gain saturates visibly rather than diverging to NaN.
- **IK is a redundant, iterative solve**, not closed-form — it converges to *a* reachable solution near the initial guess, not necessarily the unique or "natural" one a human would pick.
- **The computed-torque law is approximately, not exactly, decoupled** — see the analysis at the end of §6.
- **The path-following controller (§8) is simulation-validated only** — the control loop itself has no known open issues, but it has not yet been exercised to completion on real hardware.

---

## 12. Where this fits

Stripped of the specific hardware, this project is a working, instrumented testbed for the core mechanics of a tele-robotic digital twin: a synchronized real/target state visualization, a control architecture deliberately split across a fast local loop and a slower model-based correction pushed over a communication link, and an explicit, logged failure mode (feedforward decay, proof-of-life gating, emergency stop) for when that link degrades or drops. The progression from stock position control → local torque PD → host-computed dynamics compensation → host-driven trajectory shaping → continuous path following mirrors, at small scale, the layered architecture question at the center of synchronizing a digital twin with a remote physical system under real communication constraints — and the joint-limit bug §8 surfaced (latent since §4, only triggered by continuous re-solving) is a concrete reminder that this kind of layering can expose failure modes no single component's own tests would have caught in isolation.
