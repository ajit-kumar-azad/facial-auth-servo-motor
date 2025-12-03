import time
from serial import Serial
from serial.tools import list_ports

def get_port():
    ports = list(list_ports.comports())
    if not ports:
        raise SystemExit("No serial ports found!")
    for p in ports:
        if "ACM" in p.device or "USB" in p.device:
            return p.device
    return ports[0].device

def send_angle(angle, ser):
    cmd = f"SERVO {angle}\n"
    print(f"> {cmd.strip()}")
    ser.write(cmd.encode())
    reply = ser.readline().decode().strip()
    print(f"< {reply}")

def main():
    port = get_port()
    print("Using port:", port)

    with Serial(port, 9600, timeout=2) as ser:
        time.sleep(2)                     # allow reset
        ser.reset_input_buffer()

        # Wait for READY
        print("Waiting for READY...")
        print("<", ser.readline().decode().strip())

        # Test servo movement
        for angle in [0, 90, 180, 90]:
            send_angle(angle, ser)
            time.sleep(1)

if __name__ == "__main__":
    main()
