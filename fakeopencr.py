"""A stand-in for the OpenCR running open_manipulator_torque_pd.ino.

Speaks the exact wire protocol over a fake serial object, and runs the same
PD + gravity law on a simulated arm, so the whole PC-side hardware path can be
exercised end to end without the robot plugged in.
"""
import threading
import time

import numpy as np

from om_python import rigid_body_dynamics as rbd

S = 560.98
MAXI = 500.0
SLEW = 4000.0
JR = 0.011

# Real XM430-W350 sensor resolution. Without these the model is far too clean:
# quantisation is what turns a joint that is merely stuck into one that
# chatters, because the measured error only moves in steps and the controller
# reacts to each step.
POS_LSB = 2 * np.pi / 4096.0     # 0.00153 rad
VEL_LSB = 0.229 * 2 * np.pi / 60  # 0.02398 rad/s
CUR_LSB = 2.69                    # mA


class FakeOpenCR:
    def __init__(self, start=(0.0, -1.28, 0.24, 0.79), emit_ready=True,
                 support_goto=True, drop_gravity=False, report_hz=250.0):
        self.support_goto = support_goto      # False = emulate the OLD sketch
        self.drop_gravity = drop_gravity      # True  = emulate the gravity bug
        self.report_hz = report_hz            # what the firmware claims to achieve
        self.q = np.array(start, float)
        self.qd = np.zeros(4)
        self.kp = np.zeros(4)
        self.kd = np.zeros(4)
        self.gravity = np.zeros(4)
        self.last_cmd = np.zeros(4)
        self.cmd = np.zeros(4)
        self.armed = False
        self.ramp_start = self.q.copy()
        self.ramp_end = self.q.copy()
        self.ramp_T = 0.0
        self.ramp_t = 0.0
        self.q_ref = self.q.copy()
        self.qd_ref = np.zeros(4)
        self.applied_ff = np.zeros(4)
        self.rx = ""
        self.out = []
        self.lock = threading.Lock()
        self.closed = False
        self.seen = []                         # every command line received
        if emit_ready:
            self._emit("bus_init,ok")
            for i in (11, 12, 13, 14):
                self._emit(f"ping,{i},ok")
            self._emit("sync,read_ok,write_ok")
            self._emit("torque_pd_ready")
        self.t0 = time.time()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---- serial.Serial interface used by TorqueLink ----
    def readline(self):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            with self.lock:
                if self.out:
                    return (self.out.pop(0) + "\n").encode()
            time.sleep(0.002)
        return b""

    def write(self, data):
        with self.lock:
            self.rx += data.decode()
            while "\n" in self.rx:
                line, self.rx = self.rx.split("\n", 1)
                self._handle(line.strip())

    def close(self):
        self.closed = True
        self._stop.set()

    # ---- firmware behaviour ----
    def _emit(self, line):
        self.out.append(line)

    def _handle(self, line):
        if not line:
            return
        self.seen.append(line)
        cmd, _, rest = line.partition(",")
        try:
            vals = [float(v) for v in rest.split(",")] if rest else []
        except ValueError:
            vals = []
        if cmd == "gains" and len(vals) == 8:
            self.kp = np.array(vals[:4])
            self.kd = np.array(vals[4:])
            self._emit("gains_ack")
        elif cmd == "gravity" and len(vals) == 4:
            if not self.drop_gravity:
                self.gravity = np.clip(np.array(vals), -MAXI, MAXI)
                self._emit("gravity_ack")
        elif cmd == "target" and len(vals) == 4:
            self.ramp_start = np.array(vals)
            self.ramp_end = np.array(vals)
            self.ramp_T = 0.0
            self.ramp_t = 0.0
            self._emit("target_ack")
        elif cmd == "goto" and len(vals) == 5:
            if not self.support_goto:
                return                          # old sketch: unknown command
            self.ramp_T = max(0.0, vals[0])
            self.ramp_t = 0.0
            self.ramp_start = self.q.copy()
            self.ramp_end = np.array(vals[1:])
            self._emit("goto_ack")
        elif cmd == "torque":
            if rest == "on":
                self.armed = True
                self.ramp_start = self.ramp_end = self.q_ref = self.q.copy()
                self.ramp_T = self.ramp_t = 0.0
                self.last_cmd = np.zeros(4)
                self._emit("torque_on_ack")
            elif rest == "off":
                self.armed = False
                self.cmd = np.zeros(4)
                self._emit("torque_off_ack")

    def _loop(self):
        dt = 1 / 250.
        last_stream = 0.0
        t = 0.0
        while not self._stop.is_set():
            with self.lock:
                self._advance_ramp(dt)
                G = rbd.gravity_vector(self.q)
                q_m = np.round(self.q / POS_LSB) * POS_LSB
                qd_m = np.round(self.qd / VEL_LSB) * VEL_LSB
                if self.armed:
                    e = self.q_ref - q_m
                    ed = self.qd_ref - qd_m
                    self.applied_ff = self.gravity.copy()
                    raw = np.clip(self.kp * e + self.kd * ed + self.applied_ff,
                                  -MAXI, MAXI)
                    self.cmd = np.clip(raw, self.last_cmd - SLEW * dt,
                                       self.last_cmd + SLEW * dt)
                    self.last_cmd = self.cmd.copy()
                    self.cmd = np.round(self.cmd / CUR_LSB) * CUR_LSB
                else:
                    self.cmd = np.zeros(4)
                # plant: torque minus gravity, with gearbox stiction
                if self.armed:
                    tau = self.cmd / S - G
                    Jd = np.diag(rbd.mass_matrix(self.q)) + JR
                    fs = np.array([25., 60., 110., 30.]) / S
                    for i in range(4):
                        dr = tau[i]
                        if abs(self.qd[i]) < 1e-3:
                            if abs(dr) <= fs[i]:
                                self.qd[i] = 0.0
                                continue
                            dr -= 0.8 * fs[i] * np.sign(dr)
                        else:
                            dr -= 0.8 * fs[i] * np.sign(self.qd[i])
                        self.qd[i] += dr / Jd[i] * dt
                    self.q += self.qd * dt
                t += dt
                if t - last_stream >= 0.02:
                    last_stream = t
                    self._stream(t)
            time.sleep(dt * 0.25)          # run ~4x real time so tests are quick

    def _advance_ramp(self, dt):
        if self.ramp_T <= 0.0 or self.ramp_t >= self.ramp_T:
            self.q_ref = self.ramp_end.copy()
            self.qd_ref = np.zeros(4)
            return
        self.ramp_t = min(self.ramp_t + dt, self.ramp_T)
        u = self.ramp_t / self.ramp_T
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        sd = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / self.ramp_T
        span = self.ramp_end - self.ramp_start
        self.q_ref = self.ramp_start + span * s
        self.qd_ref = span * sd

    def _stream(self, t):
        f = lambda v, n: ",".join(f"{x:.{n}f}" for x in v)
        self._emit(f"ctl,{t:.4f},{f(self.q_ref,5)},{f(self.applied_ff,2)},{self.report_hz:.1f},"
                   f"{1 if self.armed else 0}")
        q_m = np.round(self.q / POS_LSB) * POS_LSB
        qd_m = np.round(self.qd / VEL_LSB) * VEL_LSB
        self._emit(f"state,{t:.4f},{f(q_m,5)},{f(qd_m,5)},"
                   f"{f(self.cmd,2)}")
