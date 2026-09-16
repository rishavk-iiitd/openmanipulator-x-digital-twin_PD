/*******************************************************************************
 * open_manipulator_torque_pd.ino
 *
 * Standalone OpenCR sketch for real current(torque)-mode PD control of the
 * 4 OpenManipulator-X arm joints (Dynamixel IDs 11-14, same IDs the stock
 * OpenManipulator library uses -- see open_manipulator_libs/src/open_manipulator.cpp).
 *
 * This intentionally does NOT use the OpenManipulator library, because that
 * library's JointDynamixel::setOperatingMode() only implements
 * "position_mode" and "current_based_position_mode" -- there is no path to
 * pure Current Control Mode in it. This talks to DynamixelWorkbench
 * directly instead, which does expose setCurrentControlMode().
 *
 * THE CONTROL LAW -- all of it
 * ---------------------------
 *   read  q, qdot          (one sync-read, all four joints)
 *   i_cmd = Kp*(q_ref - q) + Kd*(qdot_ref - qdot) + gravity_ff
 *   write i_cmd            (one sync-write, all four joints)
 *
 * That is the whole loop. Per-joint Kp and Kd are fixed numbers sent once by
 * the PC before a run; gravity_ff is the one feedforward term, because a
 * current-mode arm with no gravity term simply falls. Nothing else is added,
 * and nothing is computed from accumulated error, so this is plain PD.
 *
 * Earlier revisions of this file also carried Coulomb friction compensation,
 * an inertia/acceleration feedforward, and PC-streamed pose-scheduled gains.
 * They are gone deliberately. Each one added a way for the arm to misbehave
 * that was harder to diagnose than the steady-state error it was meant to
 * remove -- in particular the error-driven friction term was a saturating
 * relay with more gain than Kp itself, which made every deceleration jerk.
 * Scheduling was measured to be unnecessary as well: with the gains below,
 * the closed-loop damping ratio only moves between 0.66 and 0.98 across the
 * entire workspace, so one fixed pair per joint is genuinely good enough.
 *
 * What IS kept from that work, because it is not complexity -- it is just
 * doing the same job correctly:
 *
 *   SYNC READ/WRITE. The original read Present_Position, Present_Velocity and
 *   Present_Current one joint at a time -- 12 bus transactions in the control
 *   loop plus 12 more in the telemetry path, 24 per iteration. Measured from
 *   the timestamps of every logged run, one loop() took 49 ms: the "100 Hz"
 *   PD loop was really closing at 20 Hz, and at 20 Hz the Kd term is a whole
 *   sample behind, which is phase lag rather than damping. Present_Current,
 *   Present_Velocity and Present_Position are contiguous (126..135), so one
 *   sync-read fetches all of it and one sync-write returns the commands: 2
 *   transactions instead of 24. Telemetry re-uses the loop's own readings.
 *   The achieved rate is reported in the "ctl" line so it is verifiable.
 *
 *   GRAVITY VISIBILITY. Across all 28 logged runs that parked stationary, the
 *   measured current equalled Kp*error to within 3.3 mA -- one 2.69 mA current
 *   LSB -- and differed from Kp*error + G(q) by 71 mA. The gravity term was
 *   never reaching the motors, so every "gravity-compensated" run was really
 *   plain PD, and nothing in the protocol could reveal it. Now every gravity
 *   update is acked, the value actually being applied is echoed in telemetry,
 *   and it is held rather than zeroed while armed so a dropped line cannot
 *   silently un-compensate a loaded arm.
 *
 *   RAMPED SETPOINTS. "goto" moves the setpoint along a quintic from the
 *   joint's MEASURED angle instead of stepping it, so error starts at zero
 *   and the current command never starts saturated. This is open-loop -- it
 *   only shapes where the setpoint goes -- so it cannot destabilise anything.
 *
 * SAFETY
 * ------
 * - Every joint boots in normal Position Control Mode, holding whatever
 *   angle it's physically at -- it will NOT snap to zero or move on power-up.
 * - "torque,on" is required to arm current-mode control, and arming latches
 *   the setpoint to each joint's present angle, so it cannot produce a kick.
 * - "torque,off" returns to position-mode holding the then-current angle.
 * - MAX_CURRENT_MA hard-clamps the command regardless of what gains arrive.
 * - CURRENT_SLEW_MA_PER_S rate-limits how fast the command may change.
 * - Target angles are clamped to the arm's real joint limits.
 * - Test with the arm clear of obstructions/people and be ready to
 *   power-cycle the OpenCR if anything looks wrong.
 *
 * Serial protocol (USB CDC), newline-terminated, comma-separated:
 *   PC -> OpenCR
 *     gains,kp1..kp4,kd1..kd4   (mA per rad, mA per rad/s)
 *     target,j1..j4             (rad -- IMMEDIATE setpoint, no ramp; used by
 *                                the streaming trajectory panels)
 *     goto,duration_s,j1..j4    (rad -- quintic ramp from the measured angle)
 *     gravity,g1..g4            (mA feedforward, added every tick)
 *     torque,on | torque,off
 *   OpenCR -> PC
 *     state,t,j1..j4,v1..v4,i1..i4   (~50 Hz; unchanged from the original, so
 *                                     every panel in om_python/ still parses it)
 *       t: seconds since boot, j: angle (rad), v: velocity (rad/s),
 *       i: measured current (mA) -- our torque proxy (tau ~ Kt * i)
 *     ctl,t,r1..r4,f1..f4,hz,armed   (~50 Hz; diagnostic)
 *       r: reference angle being chased, f: gravity feedforward applied (mA),
 *       hz: measured control-loop rate, armed: 1/0
 ******************************************************************************/

#include <DynamixelWorkbench.h>

#define DEVICE_NAME "/dev/ttyUSB0"   // matches the OpenManipulator library's
#define BAUD_RATE   1000000          // default -- proven to work on this board

const uint8_t NUM_JOINTS = 4;
uint8_t joint_id[NUM_JOINTS] = {11, 12, 13, 14};  // non-const: sync APIs take uint8_t*

// XM430-W350 control table (Protocol 2.0). Present_Current (126,2),
// Present_Velocity (128,4) and Present_Position (132,4) are contiguous,
// which is what lets one sync-read fetch all the feedback at once.
#define ADDR_GOAL_CURRENT       102
#define LEN_GOAL_CURRENT          2
#define ADDR_PRESENT_CURRENT    126
#define LEN_PRESENT_CURRENT       2
#define ADDR_PRESENT_VELOCITY   128
#define LEN_PRESENT_VELOCITY      4
#define ADDR_PRESENT_POSITION   132
#define LEN_PRESENT_POSITION      4
#define ADDR_FEEDBACK_BLOCK     ADDR_PRESENT_CURRENT
#define LEN_FEEDBACK_BLOCK       10   // 126..135 inclusive

const uint8_t SR_FEEDBACK     = 0;   // sync-read handler index
const uint8_t SW_GOAL_CURRENT = 0;   // sync-write handler index

// Real joint limits, copied from open_manipulator_libs/src/open_manipulator.cpp
const float JOINT_MIN[NUM_JOINTS] = {-3.14159265f, -2.05f, -1.57079633f, -1.8f};
const float JOINT_MAX[NUM_JOINTS] = { 3.14159265f,  1.57079633f, 1.53f,   2.0f};

const int16_t MAX_CURRENT_MA = 500;           // hard ceiling, any gains
const float CURRENT_SLEW_MA_PER_S = 4000.0f;  // full scale in ~0.12 s

const uint32_t CONTROL_PERIOD_US = 4000;  // 250 Hz target
const uint32_t STREAM_PERIOD_MS  = 20;    // 50 Hz telemetry, from cached data

// Gravity is HELD, not zeroed, while armed: a dropped serial line must not
// silently remove gravity compensation from a loaded arm. It is discarded
// only after a long silence, which means the PC has genuinely gone away.
const uint32_t GRAVITY_TIMEOUT_MS = 2000;

// Velocity feedback is quantised at 0.229 rev/min (~0.024 rad/s), and Kd is
// large enough (135-235 mA per rad/s) that each quantum steps the command by
// several mA. 12 Hz is still ~6x the ~1.9 Hz closed-loop bandwidth these
// gains produce, so it costs little phase there while removing the staircase.
const float VELOCITY_LPF_HZ = 12.0f;

// Dead zone on the PROPORTIONAL term only, in radians (~0.6 deg, about 7
// encoder counts). Inside it the P term is exactly zero and the joint is held
// by gravity compensation alone.
//
// This exists because of what a geared joint does when it arrives. Position
// comes back quantised at 0.00153 rad, and the gearbox will not move at all
// until the command beats its static friction. Without a dead zone, Kp keeps
// pushing on a joint that cannot respond smoothly: current builds, the joint
// breaks loose, kinetic friction is lower than static so it overshoots,
// sticks on the far side, and the cycle repeats. That is the classic
// stick-slip limit cycle, and at the Kp values needed to reach the target in
// the first place it is violent.
//
// Damping is deliberately NOT dead-zoned -- Kd must keep acting on any motion
// inside the band, otherwise the dead zone becomes a free-running gap.
const float POSITION_DEADBAND = 0.010f;

DynamixelWorkbench dxl_wb;

float kp[NUM_JOINTS] = {0, 0, 0, 0};
float kd[NUM_JOINTS] = {0, 0, 0, 0};
float gravity_ff[NUM_JOINTS] = {0, 0, 0, 0};
float last_cmd_ma[NUM_JOINTS] = {0, 0, 0, 0};
float applied_ff[NUM_JOINTS]  = {0, 0, 0, 0};  // echoed in the "ctl" line

// Measured feedback, refreshed once per control tick by readFeedback().
float q_meas[NUM_JOINTS]    = {0, 0, 0, 0};
float qdot_meas[NUM_JOINTS] = {0, 0, 0, 0};
float qdot_filt[NUM_JOINTS] = {0, 0, 0, 0};
float i_meas[NUM_JOINTS]    = {0, 0, 0, 0};

// Setpoint state. ramp_duration <= 0 means "parked at ramp_end", which is how
// an immediate "target" command is represented.
float ramp_start[NUM_JOINTS] = {0, 0, 0, 0};
float ramp_end[NUM_JOINTS]   = {0, 0, 0, 0};
float q_ref[NUM_JOINTS]      = {0, 0, 0, 0};
float qdot_ref[NUM_JOINTS]   = {0, 0, 0, 0};
float ramp_duration = 0.0f;
float ramp_time = 0.0f;

bool torque_enabled = false;
bool sync_read_ok = false;
bool sync_write_ok = false;

unsigned long last_control_us = 0;
unsigned long last_stream_ms = 0;
unsigned long last_gravity_update = 0;
uint32_t read_failures = 0;
uint32_t reported_read_failures = 0;
float measured_hz = 0.0f;
unsigned long hz_window_start_ms = 0;
uint32_t hz_window_ticks = 0;
String input_buffer = "";

float clampf(float v, float lo, float hi)
{
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

float clampToJointLimit(uint8_t i, float angle)
{
  return clampf(angle, JOINT_MIN[i], JOINT_MAX[i]);
}

/* ------------------------------------------------------------------ *
 * Bus I/O                                                             *
 * ------------------------------------------------------------------ */

// Last known good raw feedback. These persist across calls on purpose: if a
// read fails, holding the previous sample is the only safe thing to do --
// converting whatever happened to be in the buffer would feed garbage into
// the control law, and at these gains one bogus angle is a violent command.
int32_t raw_cur[NUM_JOINTS] = {0, 0, 0, 0};
int32_t raw_vel[NUM_JOINTS] = {0, 0, 0, 0};
int32_t raw_pos[NUM_JOINTS] = {0, 0, 0, 0};

void readFeedback(float dt)
{
  const char *log = NULL;
  int32_t cur[NUM_JOINTS], vel[NUM_JOINTS], pos[NUM_JOINTS];
  bool ok = false;

  if (sync_read_ok && dxl_wb.syncRead(SR_FEEDBACK, joint_id, NUM_JOINTS, &log))
  {
    ok = dxl_wb.getSyncReadData(SR_FEEDBACK, joint_id, NUM_JOINTS,
                                ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT, cur, &log)
      && dxl_wb.getSyncReadData(SR_FEEDBACK, joint_id, NUM_JOINTS,
                                ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY, vel, &log)
      && dxl_wb.getSyncReadData(SR_FEEDBACK, joint_id, NUM_JOINTS,
                                ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION, pos, &log);
    if (ok)
    {
      for (uint8_t i = 0; i < NUM_JOINTS; i++)
      {
        raw_cur[i] = cur[i];
        raw_vel[i] = vel[i];
        raw_pos[i] = pos[i];
      }
    }
  }

  if (!ok)
  {
    read_failures++;
    for (uint8_t i = 0; i < NUM_JOINTS; i++)
    {
      int32_t v = 0;
      if (dxl_wb.itemRead(joint_id[i], "Present_Position", &v, &log)) raw_pos[i] = v;
      if (dxl_wb.itemRead(joint_id[i], "Present_Velocity", &v, &log)) raw_vel[i] = v;
      if (dxl_wb.itemRead(joint_id[i], "Present_Current",  &v, &log)) raw_cur[i] = v;
    }
  }

  // Present_Current is a 2-byte signed field, so it arrives widened as an
  // unsigned 0..65535 and has to be narrowed back to int16_t. The 4-byte
  // velocity/position fields already carry their sign in the int32_t.
  float alpha = 1.0f;
  if (dt > 0.0f)
  {
    float tau = 1.0f / (2.0f * PI * VELOCITY_LPF_HZ);
    alpha = dt / (tau + dt);
  }
  for (uint8_t i = 0; i < NUM_JOINTS; i++)
  {
    q_meas[i]    = dxl_wb.convertValue2Radian(joint_id[i], raw_pos[i]);
    qdot_meas[i] = dxl_wb.convertValue2Velocity(joint_id[i], raw_vel[i]);
    i_meas[i]    = dxl_wb.convertValue2Current(joint_id[i], (int16_t)raw_cur[i]);
    qdot_filt[i] += alpha * (qdot_meas[i] - qdot_filt[i]);
  }
}

void writeGoalCurrent(const float *ma)
{
  const char *log = NULL;
  int32_t data[NUM_JOINTS];
  for (uint8_t i = 0; i < NUM_JOINTS; i++)
    data[i] = dxl_wb.convertCurrent2Value(joint_id[i], ma[i]);

  if (sync_write_ok &&
      dxl_wb.syncWrite(SW_GOAL_CURRENT, joint_id, NUM_JOINTS, data, 1, &log))
    return;

  for (uint8_t i = 0; i < NUM_JOINTS; i++)
    dxl_wb.itemWrite(joint_id[i], "Goal_Current", data[i], &log);
}

/* ------------------------------------------------------------------ *
 * Arm / disarm                                                        *
 * ------------------------------------------------------------------ */

void latchSetpoint(uint8_t i, float angle)
{
  ramp_start[i] = angle;
  ramp_end[i]   = angle;
  q_ref[i]      = angle;
  qdot_ref[i]   = 0.0f;
  last_cmd_ma[i] = 0.0f;
}

void disarmJoint(uint8_t i)
{
  const char *log = NULL;
  float present = 0.0f;
  dxl_wb.getRadian(joint_id[i], &present, &log);
  dxl_wb.jointMode(joint_id[i], 0, 0, &log);   // position mode (torque off internally)
  dxl_wb.goalPosition(joint_id[i], present, &log);
  dxl_wb.torqueOn(joint_id[i], &log);
  latchSetpoint(i, present);   // so a later arm cannot inherit a stale goal
}

void armJoint(uint8_t i)
{
  const char *log = NULL;
  float present = 0.0f;
  dxl_wb.getRadian(joint_id[i], &present, &log);

  dxl_wb.torqueOff(joint_id[i], &log);
  dxl_wb.setCurrentControlMode(joint_id[i], &log);
  dxl_wb.torqueOn(joint_id[i], &log);

  // Error starts at zero, so arming cannot kick whatever the gains are.
  latchSetpoint(i, present);
  qdot_filt[i] = 0.0f;
}

/* ------------------------------------------------------------------ *
 * Setup                                                               *
 * ------------------------------------------------------------------ */

void setup()
{
  Serial.begin(57600);

  const char *log = NULL;
  bool bus_ok = dxl_wb.init(DEVICE_NAME, BAUD_RATE, &log);
  Serial.print("bus_init,");
  Serial.println(bus_ok ? "ok" : "FAILED");

  for (uint8_t i = 0; i < NUM_JOINTS; i++)
  {
    uint16_t model_number = 0;
    bool ok = dxl_wb.ping(joint_id[i], &model_number, &log);
    Serial.print("ping,");
    Serial.print(joint_id[i]);
    Serial.print(',');
    Serial.println(ok ? "ok" : "FAILED");
    if (!ok) continue;

    dxl_wb.itemWrite(joint_id[i], "Return_Delay_Time", 0, &log);

    // Current_Limit lives in EEPROM, and EEPROM writes are REJECTED while
    // Torque_Enable is set. Resetting the OpenCR does not power-cycle the
    // servos, so after a warm reset they can still be torque-enabled from the
    // previous session and this write would silently fail. Force torque off
    // first so the hardware ceiling is actually applied.
    dxl_wb.torqueOff(joint_id[i], &log);
    int32_t limit_value = dxl_wb.convertCurrent2Value(joint_id[i], (float)MAX_CURRENT_MA);
    dxl_wb.itemWrite(joint_id[i], "Current_Limit", limit_value, &log);

    disarmJoint(i);  // boots in position mode, holding present angle
  }

  sync_read_ok  = dxl_wb.addSyncReadHandler(ADDR_FEEDBACK_BLOCK, LEN_FEEDBACK_BLOCK, &log);
  sync_write_ok = dxl_wb.addSyncWriteHandler(ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT, &log);
  Serial.print("sync,");
  Serial.print(sync_read_ok ? "read_ok" : "read_FAILED");
  Serial.print(',');
  Serial.println(sync_write_ok ? "write_ok" : "write_FAILED");

  last_control_us = micros();
  hz_window_start_ms = millis();
  Serial.println("torque_pd_ready");
}

/* ------------------------------------------------------------------ *
 * Setpoint                                                            *
 * ------------------------------------------------------------------ */

// Quintic (minimum-jerk) scalar: s(0)=0, s(1)=1, with zero velocity and zero
// acceleration at both ends. Purely a setpoint shape -- open-loop, so it
// cannot affect stability; it only keeps the command out of saturation.
void advanceRamp(float dt)
{
  if (ramp_duration <= 0.0f || ramp_time >= ramp_duration)
  {
    for (uint8_t i = 0; i < NUM_JOINTS; i++)
    {
      q_ref[i] = ramp_end[i];
      qdot_ref[i] = 0.0f;
    }
    return;
  }

  ramp_time += dt;
  if (ramp_time > ramp_duration) ramp_time = ramp_duration;

  float u = ramp_time / ramp_duration;
  float u2 = u * u, u3 = u2 * u, u4 = u3 * u, u5 = u4 * u;
  float s = 10.0f * u3 - 15.0f * u4 + 6.0f * u5;
  float sdot = (30.0f * u2 - 60.0f * u3 + 30.0f * u4) / ramp_duration;

  for (uint8_t i = 0; i < NUM_JOINTS; i++)
  {
    float span = ramp_end[i] - ramp_start[i];
    q_ref[i] = ramp_start[i] + span * s;
    qdot_ref[i] = span * sdot;
  }
}

/* ------------------------------------------------------------------ *
 * Control -- this is the entire law                                   *
 * ------------------------------------------------------------------ */

void runControl(float dt)
{
  bool gravity_stale = (millis() - last_gravity_update) > GRAVITY_TIMEOUT_MS;
  float cmd[NUM_JOINTS];

  for (uint8_t i = 0; i < NUM_JOINTS; i++)
  {
    float error = q_ref[i] - q_meas[i];
    // Damping acts on TRACKING error. Using (0 - qdot) instead, as the
    // original did, brakes the arm's own travel toward the target.
    float error_dot = qdot_ref[i] - qdot_filt[i];

    // Dead zone on P only: shift the error toward zero by the deadband rather
    // than clipping it, so the term stays continuous at the band edge instead
    // of stepping by Kp*POSITION_DEADBAND.
    float p_error = error;
    if (p_error > POSITION_DEADBAND)       p_error -= POSITION_DEADBAND;
    else if (p_error < -POSITION_DEADBAND) p_error += POSITION_DEADBAND;
    else                                   p_error = 0.0f;

    applied_ff[i] = gravity_stale ? 0.0f : gravity_ff[i];

    float current_ma = kp[i] * p_error + kd[i] * error_dot + applied_ff[i];
    current_ma = clampf(current_ma, -(float)MAX_CURRENT_MA, (float)MAX_CURRENT_MA);

    // Bounds how violently the command may change, so no gain combination
    // can deliver an impulsive slam.
    float max_step = CURRENT_SLEW_MA_PER_S * dt;
    current_ma = clampf(current_ma, last_cmd_ma[i] - max_step, last_cmd_ma[i] + max_step);
    last_cmd_ma[i] = current_ma;
    cmd[i] = current_ma;
  }

  writeGoalCurrent(cmd);
}

/* ------------------------------------------------------------------ *
 * Telemetry                                                           *
 * ------------------------------------------------------------------ */

// Both lines are printed from the values the control loop already read this
// tick, so streaming costs zero extra bus transactions.
void streamState()
{
  float t = millis() / 1000.0;

  // "ctl" is printed BEFORE "state" so that by the time the PC's on_state
  // callback fires, last_ctl already holds this same tick's reference and
  // feedforward -- the two land in one CSV row with no skew.
  Serial.print("ctl,");
  Serial.print(t, 4);
  for (uint8_t i = 0; i < NUM_JOINTS; i++) { Serial.print(','); Serial.print(q_ref[i], 5); }
  for (uint8_t i = 0; i < NUM_JOINTS; i++) { Serial.print(','); Serial.print(applied_ff[i], 2); }
  Serial.print(',');
  Serial.print(measured_hz, 1);
  Serial.print(',');
  Serial.print(torque_enabled ? 1 : 0);
  Serial.println();

  Serial.print("state,");
  Serial.print(t, 4);
  for (uint8_t i = 0; i < NUM_JOINTS; i++) { Serial.print(','); Serial.print(q_meas[i], 5); }
  for (uint8_t i = 0; i < NUM_JOINTS; i++) { Serial.print(','); Serial.print(qdot_meas[i], 5); }
  for (uint8_t i = 0; i < NUM_JOINTS; i++) { Serial.print(','); Serial.print(i_meas[i], 2); }
  Serial.println();
}

/* ------------------------------------------------------------------ *
 * Serial command handling                                             *
 * ------------------------------------------------------------------ */

// Returns how many values were parsed, so a truncated line is rejected
// instead of leaving uninitialised garbage in the output array.
uint8_t parseFloats(const String &s, float *out, uint8_t count)
{
  int start = 0;
  uint8_t parsed = 0;
  for (uint8_t i = 0; i < count; i++)
  {
    if (start > (int)s.length()) break;
    int comma = s.indexOf(',', start);
    String tok = (comma == -1) ? s.substring(start) : s.substring(start, comma);
    tok.trim();
    if (tok.length() == 0) break;
    out[i] = tok.toFloat();
    parsed++;
    if (comma == -1) break;
    start = comma + 1;
  }
  return parsed;
}

void handleLine(String line)
{
  line.trim();
  if (line.length() == 0) return;

  int firstComma = line.indexOf(',');
  String cmd = (firstComma == -1) ? line : line.substring(0, firstComma);
  String rest = (firstComma == -1) ? "" : line.substring(firstComma + 1);

  if (cmd == "gains")
  {
    float vals[8];
    if (parseFloats(rest, vals, 8) != 8) { Serial.println("gains_err"); return; }
    for (uint8_t i = 0; i < NUM_JOINTS; i++) kp[i] = vals[i];
    for (uint8_t i = 0; i < NUM_JOINTS; i++) kd[i] = vals[4 + i];
    Serial.println("gains_ack");
  }
  else if (cmd == "target")
  {
    // Immediate setpoint, no ramp -- what the streaming trajectory panels
    // (path_follow / circular_tracking / smooth_trajectory) push at their own
    // update rate. They must not be double-smoothed here.
    float vals[NUM_JOINTS];
    if (parseFloats(rest, vals, NUM_JOINTS) != NUM_JOINTS) { Serial.println("target_err"); return; }
    for (uint8_t i = 0; i < NUM_JOINTS; i++)
    {
      float goal = clampToJointLimit(i, vals[i]);
      ramp_start[i] = goal;
      ramp_end[i]   = goal;
    }
    ramp_duration = 0.0f;
    ramp_time = 0.0f;
    Serial.println("target_ack");
  }
  else if (cmd == "goto")
  {
    // Ramped point-to-point move, from the MEASURED angle so error starts at 0.
    float vals[NUM_JOINTS + 1];
    if (parseFloats(rest, vals, NUM_JOINTS + 1) != NUM_JOINTS + 1) { Serial.println("goto_err"); return; }
    ramp_duration = clampf(vals[0], 0.0f, 60.0f);
    ramp_time = 0.0f;
    for (uint8_t i = 0; i < NUM_JOINTS; i++)
    {
      ramp_start[i] = q_meas[i];
      ramp_end[i]   = clampToJointLimit(i, vals[1 + i]);
    }
    Serial.println("goto_ack");
  }
  else if (cmd == "gravity")
  {
    float vals[NUM_JOINTS];
    if (parseFloats(rest, vals, NUM_JOINTS) != NUM_JOINTS) { Serial.println("gravity_err"); return; }
    for (uint8_t i = 0; i < NUM_JOINTS; i++)
      gravity_ff[i] = clampf(vals[i], -(float)MAX_CURRENT_MA, (float)MAX_CURRENT_MA);
    last_gravity_update = millis();
    // Acked so the PC can PROVE compensation is arriving; the applied value
    // is also echoed in "ctl".
    Serial.println("gravity_ack");
  }
  else if (cmd == "torque")
  {
    if (rest == "on")
    {
      for (uint8_t i = 0; i < NUM_JOINTS; i++) armJoint(i);
      ramp_duration = 0.0f;
      ramp_time = 0.0f;
      torque_enabled = true;
      Serial.println("torque_on_ack");
    }
    else if (rest == "off")
    {
      torque_enabled = false;
      for (uint8_t i = 0; i < NUM_JOINTS; i++) disarmJoint(i);
      Serial.println("torque_off_ack");
    }
  }
}

void readSerial()
{
  while (Serial.available())
  {
    char c = Serial.read();
    if (c == '\n')
    {
      handleLine(input_buffer);
      input_buffer = "";
    }
    else if (c != '\r')
    {
      input_buffer += c;
    }
  }
}

/* ------------------------------------------------------------------ *
 * Main loop                                                           *
 * ------------------------------------------------------------------ */

void loop()
{
  readSerial();

  unsigned long now_us = micros();
  if ((uint32_t)(now_us - last_control_us) >= CONTROL_PERIOD_US)
  {
    float dt = (now_us - last_control_us) * 1e-6f;
    if (dt > 0.2f) dt = 0.2f;   // first tick after a long stall
    last_control_us = now_us;

    // Feedback is read every tick whether or not we are armed, so telemetry
    // (and "goto"'s starting angle) stay live in position-hold mode too.
    readFeedback(dt);
    if (torque_enabled)
    {
      advanceRamp(dt);
      runControl(dt);
    }
    hz_window_ticks++;
  }

  unsigned long now_ms = millis();
  if (now_ms - hz_window_start_ms >= 500)
  {
    measured_hz = hz_window_ticks * 1000.0f / (now_ms - hz_window_start_ms);
    hz_window_start_ms = now_ms;
    hz_window_ticks = 0;
  }

  if (now_ms - last_stream_ms >= STREAM_PERIOD_MS)
  {
    last_stream_ms = now_ms;
    streamState();

    // A bus read that had to fall back to per-joint itemRead() costs ~20x the
    // time of the sync read and drags the loop rate down, so say so rather
    // than letting it show up as a lower "hz" with no explanation.
    if (read_failures != reported_read_failures)
    {
      reported_read_failures = read_failures;
      Serial.print("warn,read_failures,");
      Serial.println((unsigned long)read_failures);
    }
  }
}
