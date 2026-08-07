"""Serial link for the real-hardware torque-PD firmware
(firmware/open_manipulator_torque_pd), a different protocol from
protocol.py/serial_link.py (which target the stock position-only
open_manipulator_chain.ino).

Wire protocol (57600 baud):
    PC -> OpenCR
        gains,kp1,kp2,kp3,kp4,kd1,kd2,kd3,kd4
        target,j1,j2,j3,j4
        gravity,g1,g2,g3,g4   (mA feedforward; decays to 0 on firmware if
                                not refreshed within ~500ms)
        torque,on | torque,off
    OpenCR -> PC (streamed ~50 Hz)
        state,t,j1,j2,j3,j4,v1,v2,v3,v4,i1,i2,i3,i4
"""
import threading

import serial
import serial.tools.list_ports


class TorqueLink:
    def __init__(self, on_state=None):
        """on_state: optional callback(t, angles[4], velocities[4], currents_mA[4])
        called from the reader thread for every telemetry line received."""
        self.on_state = on_state
        self.ser = None
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.connected = False
        self.ready = False  # True once the firmware's boot line was actually seen
        self.last_state = None  # (t, angles, velocities, currents)
        self.boot_log = []
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
        if line.startswith(("bus_init", "ping,")):
            self.boot_log.append(line)
            print(f"[torque_link] {line}")
            return

        parts = line.split(",")
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

    def _write_line(self, text):
        if not self.ser:
            return
        with self._write_lock:
            self.ser.write((text + "\n").encode())

    def send_gains(self, kp, kd):
        vals = list(kp) + list(kd)
        self._write_line("gains," + ",".join(str(v) for v in vals))

    def send_target(self, angles):
        self._write_line("target," + ",".join(str(a) for a in angles))

    def send_gravity(self, gravity_ma):
        self._write_line("gravity," + ",".join(str(g) for g in gravity_ma))

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
