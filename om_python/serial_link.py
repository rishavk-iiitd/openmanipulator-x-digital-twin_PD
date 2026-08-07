import threading

import serial
import serial.tools.list_ports

from .utils import map_range


class SerialLink:
    """Talks to the OpenCR board over the same line-based protocol the
    Processing sketch used (see protocol.py). In --simulate mode, outgoing
    commands are printed instead of written to a port, so the rest of the
    app can run without hardware attached."""

    def __init__(self, state, port=None, baud=57600, simulate=False):
        self.state = state
        self.port_name = port
        self.baud = baud
        self.simulate = simulate

        self.ser = None
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def list_ports():
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self):
        if self.simulate:
            self.state.connected = True
            print("[serial] simulate mode: no hardware connection, commands will be printed")
            return

        ports = self.list_ports()
        if self.port_name is None:
            if not ports:
                raise RuntimeError(
                    "No serial ports found. Plug in the OpenCR board, or pass --simulate."
                )
            self.port_name = ports[0]

        self.ser = serial.Serial(self.port_name, self.baud, timeout=1)
        self.state.connected = True
        print(f"[serial] connected to {self.port_name} @ {self.baud}")

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

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
        cmd = line.split(",")

        if cmd[0] == "angle":
            try:
                vals = [float(v) for v in cmd[1:5]]
            except ValueError:
                return
            if len(vals) != 4:
                return
            with self.state.lock:
                self.state.receive_joint_angle[:] = vals

        elif cmd[0] == "tool":
            try:
                angle = float(cmd[1])
            except (ValueError, IndexError):
                return
            pos = map_range(angle, -0.15, 0.10, 10.0, 35.0)
            with self.state.lock:
                self.state.receive_gripper_pos[0] = pos
                self.state.receive_gripper_pos[1] = pos * -2
                self.state.ctrl_gripper_pos[0] = pos
                self.state.ctrl_gripper_pos[1] = pos * -2

        else:
            print(f"[serial] unrecognized line: {line!r}")

    def write_line(self, text):
        if self.simulate:
            print(f"[simulate -> OpenCR] {text}")
            return
        if not self.ser:
            print(f"[serial] not connected, dropping: {text}")
            return
        with self._write_lock:
            self.ser.write((text + "\n").encode())

    def close(self):
        self._stop.set()
        if self.ser is not None:
            self.ser.close()
