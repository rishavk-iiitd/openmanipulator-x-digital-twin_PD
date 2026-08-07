import argparse
import threading

from .control_panel import ControlPanel
from .renderer import ManipulatorView
from .serial_link import SerialLink
from .state import SharedState


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="OpenManipulator-X Python control panel + 3D viewer "
                     "(talks to the same OpenCR firmware as the Processing sketch)."
    )
    parser.add_argument("--port", default=None,
                         help="Serial port for the OpenCR board, e.g. COM5. "
                              "Default: first available port.")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--simulate", action="store_true",
                         help="Run without hardware; outgoing commands are printed instead of sent.")
    parser.add_argument("--list-ports", action="store_true",
                         help="List available serial ports and exit.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.list_ports:
        ports = SerialLink.list_ports()
        if not ports:
            print("No serial ports found.")
        for p in ports:
            print(p)
        return

    state = SharedState()
    link = SerialLink(state, port=args.port, baud=args.baud, simulate=args.simulate)
    try:
        link.connect()
    except RuntimeError as exc:
        print(f"[warn] {exc}")
        print("[warn] falling back to --simulate mode")
        link.simulate = True
        link.connect()

    view = ManipulatorView(state)
    view_thread = threading.Thread(target=view.run, daemon=True)
    view_thread.start()

    panel = ControlPanel(state, link)
    try:
        panel.run()  # blocks on the main thread until the control window closes
    finally:
        link.close()


if __name__ == "__main__":
    main()
