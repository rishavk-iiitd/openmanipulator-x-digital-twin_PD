# OpenManipulator-X — Python control panel + 3D viewer

A Python port of ROBOTIS's Processing sketch (`open_manipulator_chain.pde`)
for the OpenManipulator-X "Chain" arm. It talks to the **same, unmodified**
OpenCR firmware (`open_manipulator_chain.ino` + `processing.h`) over the
identical serial protocol, and renders the same OBJ meshes in a live 3D view.

Two windows, like the original:
- **OpenManipulator** — OpenGL 3D view of the arm (grey = real pose reported
  by the firmware, lighter ghost = the target pose you're commanding).
- **Control Interface** — DearPyGui panel with the same four tabs as the
  original controlP5 UI: Joint Space Control, Task Space Control, Hand
  Guiding, Motion.

## Setup

A venv already exists at `venv/` with all dependencies installed. To
recreate it from scratch:

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

## Running

```
venv\Scripts\python main.py --list-ports      # find your OpenCR's COM port
venv\Scripts\python main.py --port COM5       # run against real hardware
venv\Scripts\python main.py --simulate        # run without hardware; commands are printed to the console
venv\Scripts\python main.py                   # auto-picks the first available COM port
```

Baud rate defaults to 57600 (matches `Serial.begin(57600)` in the .ino).

## Using it

1. In the **Joint Space Control** tab, flip **Controller On/Off** — this
   sends `opm,ready` to the firmware, same as the original sketch's switch.
2. Drag the joint sliders (ranges match the original knobs: J1 ±3.14, J2
   -2.05..1.57, J3 -1.53..1.57, J4 -1.8..2.0) then **Send Joint Angle**, or
   just hit **Origin** / **Basic** for canned poses.
3. **Gripper knob** + **Set Gripper** sends a raw joint command; the
   **Gripper Open/Close** toggle sends the firmware's canned open/close
   grip move.
4. **Task Space Control** sends `task,<direction>` jogs (forward/back/left/
   right/up/down), handled entirely by the firmware's task-space IK.
5. **Hand Guiding**: turn torque off, physically move the arm, use **Save
   Joint Pose** to record waypoints on the firmware, then **Motion Start** /
   **Motion Repeat** to play them back.
6. **Motion** tab fires the two demo motions baked into the firmware
   (`motion,1` / `motion,2`).

In the 3D view: drag with the left mouse button to orbit, scroll to zoom,
`Q/A`, `W/S`, `E/D` pan the model on X/Y/Z, `I` resets the view — same
bindings as the Processing sketch. A reference grid is drawn on the plane
the arm's base sits on, with numeric mm labels every 100 mm (minor lines
every 50 mm) so you can gauge scale and distance directly against the
model — shared by both `main.py` and `pd_lab.py` since they use the same
`renderer.py`.

## Wire protocol (unchanged from the original)

```
PC -> OpenCR: opm,ready | opm,end
              joint,<j1>,<j2>,<j3>,<j4>
              gripper,<radians>
              grip,on | grip,off
              task,forward|back|left|right|up|down
              torque,on | torque,off
              get,clear | get,pose,<n> | get,on | get,off
              hand,once | hand,repeat | hand,stop
              motion,1 | motion,2

OpenCR -> PC: angle,<j1>,<j2>,<j3>,<j4>
              tool,<angle>
```

See `om_python/protocol.py`.

## Layout

```
main.py                  entry point (position-control app)
pd_lab.py                entry point (PD Control Lab)
ik_gravity_lab.py        entry point (Gravity-Compensated PD Lab)
smooth_trajectory_lab.py entry point (Smooth Trajectory PD Lab)
path_follow_lab.py       entry point (Feedback-Linearization Path-Follow Lab)
om_python/
  app.py                 wiring + CLI (--port/--baud/--simulate/--list-ports)
  state.py               thread-safe shared state (joint angles, camera, etc.)
  serial_link.py          serial I/O thread, protocol parsing (stock firmware)
  protocol.py             outgoing command strings (stock firmware)
  torque_link.py          serial I/O for the torque-PD firmware (used by
                           pd_lab.py, ik_gravity_lab.py, smooth_trajectory_lab.py,
                           path_follow_lab.py)
  mesh.py                 OBJ loader -> vertex/normal arrays
  kinematics.py           forward kinematics (gripper trail, joint positions,
                           link COMs, joint limits)
  renderer.py             GLFW + PyOpenGL 3D view (own thread)
  control_panel.py        DearPyGui control window for main.py (main thread)
  dynamics.py             placeholder per-joint PD simulation (pd_lab.py)
  inverse_kinematics.py   numerical IK (damped least squares, joint-limit-clamped)
  rigid_body_dynamics.py  M(q)/C(q,qdot)/G(q) computed-torque model
  trajectory.py           smooth reference-trajectory ODE generator
  paths.py                parametric circle/square Cartesian paths (path_follow_lab.py)
  gain_history.py         persists Kp/Kd (+ optional target) used per run
  plotting.py             saves the telemetry + end-effector-trajectory plot
                           (+ optional commanded-path overlay)
  pd_panel.py             DearPyGui control window for pd_lab.py (main thread)
  ik_gravity_panel.py     DearPyGui control window for ik_gravity_lab.py
  smooth_trajectory_panel.py  DearPyGui control window for smooth_trajectory_lab.py
  path_follow_panel.py    DearPyGui control window for path_follow_lab.py
meshes/*.obj              chain link meshes (copied from the Processing sketch's Chain/meshes)
firmware/
  open_manipulator_torque_pd/open_manipulator_torque_pd.ino  real torque-PD firmware
```

## PD Control Lab (`pd_lab.py`)

A second, standalone program built on the same package, for experimenting
with torque-based PD control -- in simulation, or for real on the arm:

```
venv\Scripts\python pd_lab.py
```

It opens the same 3D viewer plus a new **PD Control Lab** panel with a
**Simulate** / **Real Hardware** mode switch, a per-joint Kp/Kd table (4
rows, independently tunable), `Run to Basic Pose`, `Reset / Torque Off`, and
a gain history with an `EMERGENCY TORQUE OFF` button always available in
hardware mode.

### Simulate mode

Each joint is modeled as an independent, decoupled 2nd-order system with a
placeholder inertia/damping (`om_python/dynamics.py`) -- no hardware
involved. `tau_i = Kp_i*(target_i-theta_i) + Kd_i*(0-theta_dot_i)` drives
the arm from home to the Basic pose (`0, -60deg, 20deg, 40deg`) over 6
simulated seconds, live in the 3D view. Default gains: Kp=5, Kd=0.6 (N*m
scale) for every joint.

### Real Hardware mode

The stock `open_manipulator_chain.ino` firmware only accepts position
commands over serial (its `JointDynamixel::setOperatingMode()` has no path
to current/torque control), so real torque control needs different
firmware:

1. **Flash `firmware/open_manipulator_torque_pd/open_manipulator_torque_pd.ino`**
   to the OpenCR via the Arduino IDE (board: OpenCR). It talks to
   `DynamixelWorkbench` directly and puts joints 11-14 in Current Control
   Mode. It compiles clean against the OpenManipulator/DynamixelWorkbench
   versions installed on this machine (verified with `arduino-cli compile
   --fqbn OpenCR:OpenCR:OpenCR`).
   - **Safety, built into the firmware regardless of what Python sends:**
     boots in position mode holding wherever it physically is (never snaps
     to zero); commanded current is hard-clamped to +-400 mA
     (`MAX_CURRENT_MA`); target angles are clamped to the real joint
     limits; `torque,off` (or a fresh boot) returns to position-mode
     holding the then-current angle instead of going limp.
   - Test with the arm clear of obstructions, start with small gains, and
     keep a hand near the power switch. Default hardware gains in the UI:
     Kp=300/350/350/300 mA/rad, Kd=20/25/25/20 mA/(rad/s) for joints 1-4 --
     joints 2/3 carry real load against gravity, so they need noticeably
     more current than joint 1 to move at all; tune from there. With no
     integral/gravity-compensation term, expect some steady-state droop
     under gravity on those joints -- that's an inherent PD limitation, not
     a bug.
2. In the panel: switch to **Real Hardware**, enter the COM port, **Connect**.
   `Connect` waits up to 5s to see real proof the firmware is alive --
   either its one-time `torque_pd_ready` boot line (only printed right
   after a fresh flash/power-cycle) or a valid `state,...` telemetry line
   (the normal case when reconnecting to an already-running board -- unlike
   classic Arduino-Uno-style boards, OpenCR's native USB doesn't reset the
   board on every serial connect, so the boot line won't reappear). If
   neither shows up, the status bar shows the boot log instead of silently
   pretending it worked, and `Run to Basic Pose` is blocked until it
   succeeds. The 3D view's "goal" arm is driven live from the real arm's
   telemetry the whole time it's connected (not just during a recorded run)
   -- the visualization stays synced to whatever the physical arm is
   actually doing.
3. **EMERGENCY TORQUE OFF** immediately sends `torque,off`, returning every
   joint to position-mode holding its current angle.

Wire protocol (57600 baud, separate from `protocol.py`/`serial_link.py`
which target the stock firmware) is documented in
`om_python/torque_link.py` and at the top of the `.ino`.

**Bugs found and fixed while bringing this up on real hardware:** ROBOTIS's
own `DynamixelWorkbench::getPresentVelocityData()` (bundled with OpenCR
1.5.3) actually reads the `Goal_Velocity`/`Moving_Speed` register, not
`Present_Velocity` -- so the firmware's Kd term was fed bogus velocity
feedback. Fixed by reading the `Present_Velocity` item directly
(`getPresentVelocity()` in the `.ino`). Separately, `Connect` had no guard
against being clicked twice, so a second click could try to reopen a port
the first click already had open (`PermissionError`/"Access is denied") --
fixed by guarding against concurrent connect attempts and releasing any
existing connection first. And the very first hardware runs produced empty
telemetry (header-only CSVs) simply because the firmware hadn't actually
been flashed yet when they were tried.

### Both modes

- Every run is appended to `pd_gain_history.json` (per-joint Kp/Kd, mode,
  timestamp) and shown in the history list -- double-click an entry to
  reload those gains into the table.
- On completion, a plot is saved to `plots/run_<timestamp>.png` (+ a `.csv`
  of the raw numbers, including the end-effector's X/Y/Z) and opened
  automatically:
  - A 6-panel grid (angle, position, velocity, angular velocity,
    torque/current, jerk -- each with all 4 joints overlaid). "Position"/
    "velocity" are each joint's own distance from the base origin (via
    `kinematics.joint_positions`) and its time derivative, not the joint
    angle itself. In hardware mode, velocity, angular acceleration, and
    jerk are all derived from the streamed angle/velocity telemetry by
    finite differences (no accelerometer on board); "torque" there is the
    servo's real measured current (mA), not an N*m estimate.
  - A 3D end-effector trajectory panel (gridded X/Y/Z axes, start in green,
    end in red) via `kinematics.gripper_center`, plus a text panel right
    next to it giving the exact start/final X/Y/Z coordinates (mm) and net
    displacement.

## Gravity-Compensated PD Lab (`ik_gravity_lab.py`)

A third program built on the same package: instead of a fixed "Basic pose"
joint-space target, you type a target end-effector **XYZ (mm)**; instead of
treating each joint as independent (`dynamics.py`'s model), the controller
compensates for the arm's full coupled rigid-body dynamics. Has both a
Simulate mode and a Real Hardware mode, same split as `pd_lab.py`.

```
venv\Scripts\python ik_gravity_lab.py
```

- **Stage 1 -- IK**: `om_python/inverse_kinematics.py` solves for joint
  angles reaching the typed XYZ via damped least squares (Levenberg-
  Marquardt-style) on a numerically-differentiated Jacobian of
  `kinematics.gripper_center`, not a hand-derived closed form -- link4's
  offset `(24, 0, 128)` isn't axis-aligned, which makes closed-form
  solutions easy to get subtly wrong. Click **Solve IK** to preview the
  result as the ghost arm in the 3D view before running anything; it
  reports whether the target was actually reachable and the residual error
  in mm (targets outside the arm's reach converge to the closest reachable
  point instead of returning nonsense).
- **Stage 2 -- computed-torque control**: `om_python/rigid_body_dynamics.py`
  numerically derives the mass matrix `M(q)`, Coriolis matrix
  `C(q, qdot)`, and gravity vector `G(q)` from the same forward-kinematics
  chain (treating each link's mass as a point at its COM --
  `kinematics.link_com_positions` -- which is a documented approximation,
  same spirit as `dynamics.py`'s placeholder inertia; link masses and each
  joint's own rotor/local inertia are approximate, not datasheet values, no
  local URDF was available). The control law is
  `tau = Kp*e + Kd*edot + C(q,qdot)*qdot + G(q)`, which cancels the arm's
  own coupling/gravity terms so the closed-loop response is a clean,
  Kp/Kd-tunable linear system regardless of pose -- this is what actually
  fixes the steady-state droop the simplified PD lab has under gravity (the
  torque panel in the saved plot visibly holds a nonzero steady-state value
  instead of decaying to zero).
- This model's real (much smaller, ~0.005-0.02 kg\*m^2) inertia means gains
  need to be roughly 10x smaller than the simplified PD lab's to stay
  stable at the same 100 Hz timestep -- defaults are `Kp=0.6`, `Kd=0.15`
  per joint. Explicit-Euler integration of this stiffer, coupled system
  diverges to NaN within ~30 steps for gains that felt fine in the simpler
  model, so angular acceleration is hard-clamped (`MAX_ANGULAR_ACCEL` in
  `rigid_body_dynamics.py`) as a safety net -- too-aggressive gains
  saturate visibly rather than blow up.
- History is a separate file (`ik_gain_history.json`) from the other two
  programs', since the "target" here is an XYZ position rather than a fixed
  pose -- entries store the target XYZ alongside Kp/Kd.

### Real Hardware mode

Reuses `firmware/open_manipulator_torque_pd.ino` and `torque_link.py` from
`pd_lab.py` unchanged in spirit, but the firmware only ever runs a plain
per-joint PD loop locally (100 Hz on the OpenCR -- serial round-trip
latency is too slow for a PC-side loop to do that part, confirmed while
building `pd_lab.py`'s hardware mode). Porting the full numerically-derived
`M(q)`/`C(q,qdot)`/`G(q)` dynamics to run on the microcontroller isn't
practical, so instead:

- The firmware gained a new `gravity,g1,g2,g3,g4` command (mA feedforward,
  simply added to its existing `Kp*e + Kd*edot` output each control tick)
  and decays it to zero if not refreshed within 500 ms, so a PC-side
  disconnect can't leave a stale gravity offset being applied forever.
- On the PC, a background thread recomputes `G(q)` from the arm's latest
  streamed telemetry at 20 Hz (gravity only depends on position, which
  changes slowly -- no need for 100 Hz here) and pushes it over. The
  **Gravity comp scale (mA per N\*m)** field (default 500) converts the
  SI-unit `G(q)` into the current space the hardware Kp/Kd already operate
  in; there's no verified torque constant for these servos, so this is a
  tunable approximation -- if the arm still droops under gravity, raise it;
  if it overshoots/pushes past the target at rest, lower it.
- Hardware-mode Kp/Kd defaults (`300/350/350/300` mA/rad,
  `20/25/25/20` mA/(rad/s)) are the exact same values as `pd_lab.py`'s,
  imported from there rather than duplicated -- it's the same firmware PD
  loop, gravity compensation is strictly additive on top.
- Same safety behavior as `pd_lab.py`: boots/disarms into position-mode
  hold (never goes limp or snaps to zero), `Connect` requires seeing real
  proof of firmware life before `Run` is allowed, and **EMERGENCY TORQUE
  OFF** is always available once in hardware mode.

## Smooth Trajectory PD Lab (`smooth_trajectory_lab.py`)

A copy of `ik_gravity_lab.py` where the target fed to the controller is no
longer a step input. In the gravity-compensated lab, clicking Run hands the
PD controller the IK-solved joint angles directly, so the very first
control tick sees the *full* position error at once -- that's a step
input, and it's why those plots show a sharp initial velocity/jerk spike
even though the motion is otherwise well-behaved.

```
venv\Scripts\python smooth_trajectory_lab.py
```

- **`om_python/trajectory.py`** integrates a critically-damped 2nd-order
  reference-model ODE every control tick:
  `qddot_ref = omega_n^2*(goal - q_ref) - 2*zeta*omega_n*qdot_ref`
  (`zeta=1`), producing a smoothly evolving intermediate target
  `(q_ref, qdot_ref, qddot_ref)` that gradually glides from the current
  pose to the IK solution instead of jumping there -- this is what "the PD
  controls are dynamic, changed by solving an ODE" means concretely: the
  target the controller tracks is a new set of numbers every 10 ms,
  produced by numerically integrating that ODE, not one fixed number set
  once at the start. The **Smoothing time (s)** field controls the pace
  (roughly `4/omega_n`) -- smaller is snappier (closer to a step input),
  larger is smoother/slower.
- `rigid_body_dynamics.RigidBodyDynamics.step()` gained optional
  `target_velocity`/`target_accel` parameters (both default to zero,
  so `ik_gravity_lab.py`'s existing calls are unaffected) and now feeds
  forward `M(q) @ target_accel` in the torque law -- proper computed-torque
  *trajectory tracking*, not just point regulation, so the PD term only
  has to correct small tracking errors instead of driving the whole move.
- Measured result at the same target/gains as the un-smoothed version: peak
  jerk dropped from ~8000 rad/s^3 to ~370 (>20x), peak joint speed from
  ~170 mm/s to ~33 mm/s, while final accuracy stayed within 0.02 mm of the
  IK solution.
- **Real Hardware mode** works the same way as `ik_gravity_lab.py`'s (same
  firmware, same gravity feedforward), plus a second background thread that
  re-integrates the reference ODE against wall-clock time and streams the
  moving `q_ref` to the firmware as its `target` at 50 Hz -- the firmware
  itself only ever tracks a fixed setpoint, so smoothing on real hardware
  comes from the PC continuously "leading" that setpoint along the curve
  rather than the firmware knowing anything about trajectories.
- History is `smooth_gain_history.json` (separate from both other
  programs'), and additionally stores the smoothing time used.

## Path-Follow Lab (`path_follow_lab.py`)

A fifth program on the same package: instead of a single fixed XYZ target,
the goal continuously moves along a parametric **circle or square**
(`om_python/paths.py`), traced in a chosen plane (`xz` vertical/forward-
facing, `xy` horizontal, or `yz` vertical/side-on). This is the *same*
feedback-linearization (computed-torque) controller as
`ik_gravity_lab.py`/`smooth_trajectory_lab.py` -- no new control law, just a
continuously moving reference instead of a static one.

```
venv\Scripts\python path_follow_lab.py
```

- Each control tick: sample the path at the current time to get an
  instantaneous Cartesian point; solve IK for it (warm-started from the
  previous tick's solution, so it converges in a handful of iterations
  instead of two hundred); run that joint-space point through the same
  critically-damped reference filter as `smooth_trajectory_lab.py` -- this
  is what rounds the square path's corners, since the filter physically
  cannot track an instantaneous change in velocity direction; hand the
  filtered `(q_ref, qdot_ref, qddot_ref)` to the same
  `RigidBodyDynamics.step()` used everywhere else.
- **Bug found and fixed while building this lab:** every other program
  calls `inverse_kinematics.solve()` once per run from a neutral initial
  guess. Continuously re-solving IK along a moving path, warm-started every
  10 ms, exposed a latent defect: if a joint is already at its limit and
  the computed step would push it further past that limit, the existing
  per-iteration clamp silently undid that joint's share of the step, but
  *not* the other joints' share of the same now-inconsistent step -- so
  they could walk away from the target instead of compensating for it,
  and the residual grew unbounded (over 140 mm on one traced 80 mm-radius
  circle) instead of settling at "closest reachable point." Fixed with a
  standard joint-limit-clamping correction in `inverse_kinematics.solve()`:
  before taking a step, zero the Jacobian column of any joint that's at
  its limit and being pushed further past it, and re-solve for the
  remaining free joints. The shipped default path (60 mm radius/side,
  centered at the same (150, 0, 150) mm point used elsewhere) was chosen
  to stay fully within the reachable workspace across multiple loops
  (residuals under 0.5 mm throughout) with this fix in place.
- **Smoothing time means something different here than in the
  point-to-point lab.** Because the goal is persistently moving rather
  than static, `Smoothing time (s)` behaves like a low-pass corner
  frequency tracking a moving signal, not a one-time settling duration.
  Measured on a 60 mm-radius circle (3 loops, 6 s/loop): steady-state
  tracking RMS error is ~28.5 mm at `smoothing_time=1.0s`, ~11.2 mm at the
  shipped default `0.4s`, ~6.6 mm at `0.25s`, and ~1.9 mm at `0.1s` --
  tighter tracking costs sharply higher commanded jerk (from ~1,100 to
  ~52,000 rad/s^3 across that same range) as the reference filter's
  bandwidth approaches the safety-clamped acceleration limit. `0.4s` was
  chosen as the default to keep jerk roughly in the range of the
  un-smoothed step response documented above.
- `om_python/plotting.py` gained an optional `commanded_path` overlay:
  when given, the saved 3D end-effector plot draws the actual trajectory
  against the commanded path as a dashed reference curve, and the text
  panel reports the index-aligned RMS tracking error in mm.
- **Real Hardware mode** reuses the same firmware and gravity-compensation
  bridge as the other torque-based labs unchanged, plus a background
  thread that re-solves IK and re-integrates the reference filter against
  wall-clock time, streaming the moving `q_ref` to the firmware as its
  `target` at 50 Hz.
- History is `path_gain_history.json`, storing shape, plane, center, size,
  period, loops, and smoothing time alongside gains.
- **Validation status:** the control loop has been validated headlessly in
  simulation (bounded, NaN-free, with the trade-off above measured
  directly) and the panel has been confirmed to launch and render
  correctly with these defaults. It has not yet been run to completion on
  real hardware.

## Known deviations from the Processing sketch

- The in-3D-view title text isn't rendered (Processing used its own text
  renderer). The equivalent info lives in the control panel window instead.
  (GLUT bitmap fonts, used for the floor-grid scale labels below, do work
  fine on this machine -- the title text was just never revisited once the
  control panel covered the same information.)
- The gripper knob's angle-to-position mapping is copied verbatim from the
  original (`map(angle, 0.907, -1.13, 10, 35)` against a slider that ranges
  -10..10), which is a narrow, twitchy mapping in the original code too —
  left as-is for fidelity rather than "fixed".
- Controller knobs are sliders in DearPyGui rather than literal rotary
  knobs; behavior is identical, only the widget shape differs.
