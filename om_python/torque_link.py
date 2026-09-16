"""Serial link for the real-hardware torque-control firmware
(firmware/open_manipulator_torque_pd), a different protocol from
protocol.py/serial_link.py (which target the stock position-only
open_manipulator_chain.ino).

Wire protocol (USB CDC):
    PC -> OpenCR
        gains,kp1..kp4,kd1..kd4      (mA per rad, mA per rad/s)
        target,j1..j4                (rad, immediate setpoint -- no ramp)
        goto,duration_s,j1..j4       (rad, quintic ramp from the measured angle)
        gravity,g1..g4               (mA feedforward, applied every tick)
        torque,on | torque,off
    OpenCR -> PC (streamed ~50 Hz)
        state,t,j1..j4,v1..v4,i1..i4
        ctl,t,r1..r4,f1..f4,hz,armed

The "ctl" line and the per-command acks exist because the logged runs in
plots/ proved the gravity feedforward was never actually reaching the
motors -- measured current matched Kp*error to within one 2.69 mA current
LSB across all 28 parked runs, and differed from Kp*error + G(q) by 71 mA.
Nothing in the old protocol could have revealed that, so the firmware now
reports the gravity feedforward it is really applying (f1..f4) and the
control rate it is really achieving (hz), and gravity_acks lets a caller
check that its updates are landing instead of trusting that they are.
"""
import threading

import serial
import serial.tools.list_ports


class TorqueLink:
    def __init__(self, on_state=None, on_ctl=None):
        """on_state: optional callback(t, angles[4], velocities[4], currents_mA[4])
        called from the reader thread for every telemetry line received.
        on_ctl: optional callback(t, q_ref[4], applied_ff_mA[4], hz, armed)
        for the firmware's control-diagnostic line -- reference angle, the
        feedforward actually applied, and the measured loop rate.
"""
        self.on_state = on_state
        self.on_ctl = on_ctl
        self.ser = None
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.connected = False
        self.ready = False  # True once the firmware's boot line was actually seen
        self.last_state = None  # (t, angles, velocities, currents)
        self.last_ctl = None    # (t, q_ref, applied_ff, hz, armed)
        self.boot_log = []
        self.errors = []        # any *_err line the firmware rejected
        self.acks = {}          # ack name -> count, e.g. {"goto_ack": 1}
        self.gravity_acks = 0   # proof gravity updates are landing
        self._got_ready = threading.Event()

    @staticmethod
    def list_ports():
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port, baud=57600, ready_timeout=5.0):
        """Opens the port and waits to see real evidence the firmware is
        alive: either its one-time "torque_pd_ready" boot line (only seen
        right after a fresh flash/power-cycle -- unlike classic
        Arduino-Uno-style boards, the OpenCR's native USB doesn't reset the
        board on every serial connect, so a *reconnect* to an
        already-running board will never see that line again) or a
        successfully-parsed "state,..." telemetry line (proof it's actually
        streaming, for that reconnect case). Returns True once either is
        seen, False if ready_timeout elapses with nothing recognized (the
        port is still left open either way; check self.boot_log)."""
        self.ser = serial.Serial(port, baud, timeout=1)
        self._got_ready.clear()
        self.boot_log = []
        self.errors = []
        self.acks = {}
        self.connected = True

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        self.ready = self._got_ready.wait(timeout=ready_timeout)
        return self.ready

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                raw = self.ser.readline()
            except serial.SerialException:
                break
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if line:
                self._handle_line(line)

    def _handle_line(self, line):
        if line == "torque_pd_ready":
            self._got_ready.set()
            self.boot_log.append(line)
            return
        if line == "gravity_ack":
            self.gravity_acks += 1
            self.acks[line] = self.acks.get(line, 0) + 1
            return
        if line.endswith("_ack"):
            # Tracked by name so a caller can tell "the firmware did not
            # understand this command" (stale flash) from "the command failed".
            self.acks[line] = self.acks.get(line, 0) + 1
            return
        if line.startswith(("bus_init", "ping,", "sync,")):
            self.boot_log.append(line)
            print(f"[torque_link] {line}")
            return
        if line.endswith("_err"):
            self.errors.append(line)
            print(f"[torque_link] REJECTED: {line}")
            return

        parts = line.split(",")
        if parts[0] == "ctl":
            self._handle_ctl(parts)
            return
        if parts[0] != "state":
            print(f"[torque_link] {line}")
            return
        try:
            vals = [float(v) for v in parts[1:14]]
        except ValueError:
            return
        if len(vals) != 13:
            return

        t = vals[0]
        angles = vals[1:5]
        velocities = vals[5:9]
        currents = vals[9:13]
        self.last_state = (t, angles, velocities, currents)
        self._got_ready.set()  # a valid telemetry line is proof enough the link is alive
        if self.on_state:
            self.on_state(t, angles, velocities, currents)

    def _handle_ctl(self, parts):
        try:
            vals = [float(v) for v in parts[1:12]]
        except ValueError:
            return
        if len(vals) != 11:
            return
        t, q_ref, applied_ff = vals[0], vals[1:5], vals[5:9]
        hz, armed = vals[9], bool(vals[10])
        self.last_ctl = (t, q_ref, applied_ff, hz, armed)
        if self.on_ctl:
            self.on_ctl(t, q_ref, applied_ff, hz, armed)

    def _write_line(self, text):
        if not self.ser:
            return
        with self._write_lock:
            self.ser.write((text + "\n").encode())

    def send_gains(self, kp, kd):
        vals = list(kp) + list(kd)
        self._write_line("gains," + ",".join(f"{float(v):.6g}" for v in vals))

    def send_target(self, angles):
        """Immediate setpoint, no ramp -- for streaming a reference trajectory
        that is already smooth (path_follow / circular_tracking / smooth_
        trajectory panels). Use send_goto() for a point-to-point move."""
        self._write_line("target," + ",".join(f"{float(a):.6g}" for a in angles))

    def send_goto(self, angles, duration_s):
        """Point-to-point move with a quintic ramp generated on the firmware's
        fast loop, starting from the joint's measured angle."""
        vals = [duration_s] + list(angles)
        self._write_line("goto," + ",".join(f"{float(v):.6g}" for v in vals))

    def send_gravity(self, gravity_ma):
        self._write_line("gravity," + ",".join(f"{float(g):.6g}" for g in gravity_ma))

    def torque_on(self):
        self._write_line("torque,on")

    def torque_off(self):
        self._write_line("torque,off")

    def close(self):
        self._stop.set()
        if self.ser is not None:
            try:
                self.torque_off()
            except Exception:
                pass
            self.ser.close()
