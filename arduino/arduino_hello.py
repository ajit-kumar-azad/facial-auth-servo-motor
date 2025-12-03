#!/usr/bin/env python3
import time
import sys
import argparse
from serial import Serial
from serial.tools import list_ports

DEFAULT_BAUD = 9600

def pick_port(auto=True):
    ports = list(list_ports.comports())
    if not ports:
        raise SystemExit("No serial ports found. Is the Uno connected?")
    # Heuristic: prefer ACM* (typical for Uno R3)
    acm = [p.device for p in ports if "USB" in p.device]
    if auto and acm:
        return acm[0]
    # fall back to the first port
    return ports[0].device

def main():
    ap = argparse.ArgumentParser(description="Hello World to Arduino Uno")
    ap.add_argument("--port", help="Serial port (e.g., /dev/ttyACM0). If omitted, auto-detect.")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate (default 9600)")
    args = ap.parse_args()

    port = args.port or pick_port(auto=True)
    print(f"Using port: {port} @ {args.baud}")

    # Opening the port resets the Uno; give it a moment
    with Serial(port, args.baud, timeout=2) as ser:
        # Clear any stale input
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # Wait for "READY" from Arduino after reset
        start = time.time()
        ready = None
        while time.time() - start < 5:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print(f"< {line}")
            if line == "READY":
                ready = True
                break
        if not ready:
            print("Did not see READY; proceeding anyway…")

        # Send HELLO
        cmd = "HELLO\n"
        print(f"> {cmd.strip()}")
        ser.write(cmd.encode())
        reply = ser.readline().decode(errors="ignore").strip()
        print(f"< {reply}")

        # Toggle LED ON, then OFF
        for cmd in ("LED ON\n", "LED OFF\n"):
            time.sleep(0.5)
            print(f"> {cmd.strip()}")
            ser.write(cmd.encode())
            print(f"< {ser.readline().decode(errors='ignore').strip()}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
