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
 * SAFETY
 * ------
 * - Every joint boots in normal Position Control Mode, holding whatever
 *   angle it's physically at -- it will NOT snap to zero or move on power-up.
 * - "torque,on" is required to arm current-mode PD control; "torque,off"
 *   (or if the gains/target look implausible) returns to position-mode
 *   holding the then-current angle, so there's no sudden drop.
 * - MAX_CURRENT_MA hard-clamps the commanded current in firmware,
 *   independent of whatever Kp/Kd the PC sends -- a bad gain cannot exceed it.
 * - Target angles are clamped to the arm's real joint limits (from
 *   open_manipulator.cpp) before use.
 * - Test with the arm clear of obstructions/people, start with small Kp/Kd,
 *   and be ready to power-cycle the OpenCR if anything looks wrong.
 *
 * Serial protocol (57600 baud), newline-terminated, comma-separated:
 *   PC -> OpenCR
 *     gains,kp1,kp2,kp3,kp4,kd1,kd2,kd3,kd4   (mA per rad, mA per rad/s)
 *     target,j1,j2,j3,j4                       (radians)
 *     gravity,g1,g2,g3,g4                      (mA feedforward, added to the
 *                                                PD output every control tick)
 *     torque,on | torque,off
 *   OpenCR -> PC (streamed continuously at ~50 Hz)
 *     state,t,j1,j2,j3,j4,v1,v2,v3,v4,i1,i2,i3,i4
 *       t: seconds since boot
 *       j: present angle (rad), v: present velocity (rad/s)
 *       i: present current (mA) -- the real, measured current, used as our
 *          torque proxy (tau ~ Kt * i)
 *
 * The fast position/velocity PD loop runs locally at 100 Hz (serial
 * round-trip latency is too slow for a PC-side loop -- see the "gains"
 * command). Gravity compensation is different: it only depends on the
 * arm's current pose, which changes slowly, so the PC computes it (full
 * mass-matrix-derived M(q)/C(q,qdot)/G(q) dynamics -- impractical to port
 * to this microcontroller) and periodically pushes it over as "gravity".
 * If no update arrives for GRAVITY_TIMEOUT_MS, it decays to zero rather
 * than keep applying a stale value from before e.g. a disconnect.
 ******************************************************************************/

#include <DynamixelWorkbench.h>

#define DEVICE_NAME "/dev/ttyUSB0"   // matches the OpenManipulator library's
#define BAUD_RATE   1000000          // default -- proven to work on this board

const uint8_t NUM_JOINTS = 4;
const uint8_t JOINT_ID[NUM_JOINTS] = {11, 12, 13, 14};

// Real joint limits, copied from open_manipulator_libs/src/open_manipulator.cpp
const float JOINT_MIN[NUM_JOINTS] = {-3.14159265f, -2.05f, -1.57079633f, -1.8f};
const float JOINT_MAX[NUM_JOINTS] = { 3.14159265f,  1.57079633f, 1.53f,   2.0f};

const int16_t MAX_CURRENT_MA = 400;   // hard ceiling regardless of Kp/Kd
const uint32_t CONTROL_PERIOD_MS = 10;  // 100 Hz PD loop
const uint32_t STREAM_PERIOD_MS  = 20;  // 50 Hz telemetry
const uint32_t GRAVITY_TIMEOUT_MS = 500;  // gravity_ff decays to 0 if stale

DynamixelWorkbench dxl_wb;

float kp[NUM_JOINTS] = {0, 0, 0, 0};
float kd[NUM_JOINTS] = {0, 0, 0, 0};
float target[NUM_JOINTS] = {0, 0, 0, 0};
float gravity_ff[NUM_JOINTS] = {0, 0, 0, 0};
bool torque_enabled = false;

unsigned long last_control_time = 0;
unsigned long last_stream_time = 0;
unsigned long last_gravity_update = 0;
String input_buffer = "";

float clampToJointLimit(uint8_t i, float angle)
{
  if (angle < JOINT_MIN[i]) return JOINT_MIN[i];
  if (angle > JOINT_MAX[i]) return JOINT_MAX[i];
  return angle;
}

// DynamixelWorkbench::getVelocity()/getPresentVelocityData() actually reads
// the "Goal_Velocity" (falling back to "Moving_Speed") register, NOT
// "Present_Velocity" -- confirmed in dynamixel_workbench.cpp. That's not
// useful for feedback, so read the real present-velocity item directly.
float getPresentVelocity(uint8_t id)
{
  const char *log = NULL;
  int32_t raw = 0;
  if (!dxl_wb.itemRead(id, "Present_Velocity", &raw, &log)) return 0.0f;
  return dxl_wb.convertValue2Velocity(id, raw);
}

void disarmJoint(uint8_t i)
{
  const char *log = NULL;
  float present = 0.0f;
  dxl_wb.getRadian(JOINT_ID[i], &present, &log);
  dxl_wb.jointMode(JOINT_ID[i], 0, 0, &log);   // position mode (torque off internally)
  dxl_wb.goalPosition(JOINT_ID[i], present, &log);
  dxl_wb.torqueOn(JOINT_ID[i], &log);
  target[i] = present;
}

void armJoint(uint8_t i)
{
  const char *log = NULL;
  dxl_wb.torqueOff(JOINT_ID[i], &log);
  dxl_wb.setCurrentControlMode(JOINT_ID[i], &log);
  dxl_wb.torqueOn(JOINT_ID[i], &log);
}

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
    bool ok = dxl_wb.ping(JOINT_ID[i], &model_number, &log);
    Serial.print("ping,");
    Serial.print(JOINT_ID[i]);
    Serial.print(',');
    Serial.println(ok ? "ok" : "FAILED");
    if (!ok)
    {
      continue;
    }

    dxl_wb.itemWrite(JOINT_ID[i], "Return_Delay_Time", 0, &log);
    // Hardware-level current ceiling too (belt-and-suspenders vs. the
    // software clamp in runPD()).
    int32_t limit_value = dxl_wb.convertCurrent2Value(JOINT_ID[i], (float)MAX_CURRENT_MA);
    dxl_wb.itemWrite(JOINT_ID[i], "Current_Limit", limit_value, &log);

    disarmJoint(i);  // boots in position mode, holding present angle
  }

  Serial.println("torque_pd_ready");
}

void runPD()
{
  const char *log = NULL;

  bool gravity_stale = (millis() - last_gravity_update) > GRAVITY_TIMEOUT_MS;

  for (uint8_t i = 0; i < NUM_JOINTS; i++)
  {
    float angle = 0.0f;
    dxl_wb.getRadian(JOINT_ID[i], &angle, &log);
    float vel = getPresentVelocity(JOINT_ID[i]);

    float error = target[i] - angle;
    float error_dot = 0.0f - vel;
    float ff = gravity_stale ? 0.0f : gravity_ff[i];
    float current_ma = kp[i] * error + kd[i] * error_dot + ff;

    if (current_ma > MAX_CURRENT_MA) current_ma = MAX_CURRENT_MA;
    if (current_ma < -MAX_CURRENT_MA) current_ma = -MAX_CURRENT_MA;

    int16_t value = dxl_wb.convertCurrent2Value(JOINT_ID[i], current_ma);
    dxl_wb.itemWrite(JOINT_ID[i], "Goal_Current", value, &log);
  }
}

void streamState()
{
  const char *log = NULL;
  float angle[NUM_JOINTS], vel[NUM_JOINTS], cur[NUM_JOINTS];

  for (uint8_t i = 0; i < NUM_JOINTS; i++)
  {
    angle[i] = 0.0f;
    dxl_wb.getRadian(JOINT_ID[i], &angle[i], &log);
    vel[i] = getPresentVelocity(JOINT_ID[i]);

    int32_t raw = 0;
    dxl_wb.itemRead(JOINT_ID[i], "Present_Current", &raw, &log);
    cur[i] = dxl_wb.convertValue2Current(JOINT_ID[i], (int16_t)raw);
  }

  Serial.print("state,");
  Serial.print(millis() / 1000.0, 4);
  for (uint8_t i = 0; i < NUM_JOINTS; i++) { Serial.print(','); Serial.print(angle[i], 5); }
  for (uint8_t i = 0; i < NUM_JOINTS; i++) { Serial.print(','); Serial.print(vel[i], 5); }
  for (uint8_t i = 0; i < NUM_JOINTS; i++) { Serial.print(','); Serial.print(cur[i], 2); }
  Serial.println();
}

void parseFloats(const String &s, float *out, uint8_t count)
{
  int start = 0;
  for (uint8_t i = 0; i < count; i++)
  {
    int comma = s.indexOf(',', start);
    String tok = (comma == -1) ? s.substring(start) : s.substring(start, comma);
    out[i] = tok.toFloat();
    if (comma == -1) break;
    start = comma + 1;
  }
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
    parseFloats(rest, vals, 8);
    for (uint8_t i = 0; i < NUM_JOINTS; i++) kp[i] = vals[i];
    for (uint8_t i = 0; i < NUM_JOINTS; i++) kd[i] = vals[4 + i];
    Serial.println("gains_ack");
  }
  else if (cmd == "target")
  {
    float vals[NUM_JOINTS];
    parseFloats(rest, vals, NUM_JOINTS);
    for (uint8_t i = 0; i < NUM_JOINTS; i++) target[i] = clampToJointLimit(i, vals[i]);
    Serial.println("target_ack");
  }
  else if (cmd == "gravity")
  {
    float vals[NUM_JOINTS];
    parseFloats(rest, vals, NUM_JOINTS);
    for (uint8_t i = 0; i < NUM_JOINTS; i++)
    {
      float g = vals[i];
      if (g > MAX_CURRENT_MA) g = MAX_CURRENT_MA;
      if (g < -MAX_CURRENT_MA) g = -MAX_CURRENT_MA;
      gravity_ff[i] = g;
    }
    last_gravity_update = millis();
  }
  else if (cmd == "torque")
  {
    if (rest == "on")
    {
      for (uint8_t i = 0; i < NUM_JOINTS; i++) armJoint(i);
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

void loop()
{
  readSerial();

  unsigned long now = millis();
  if (torque_enabled && (now - last_control_time >= CONTROL_PERIOD_MS))
  {
    last_control_time = now;
    runPD();
  }

  if (now - last_stream_time >= STREAM_PERIOD_MS)
  {
    last_stream_time = now;
    streamState();
  }
}
