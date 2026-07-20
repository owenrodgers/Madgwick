import queue
import threading
import serial
from typing import Tuple, Optional
import struct

FRAME_HEADER = 0x77777777

def main():
    port = serial.Serial('/dev/cu.usbmodem1302', 115200, timeout = 0.1)

    # set up channels + threads
    msg_chan : queue.Queue[Tuple[float, float, float]] = queue.Queue()

    # spawn publisher
    publisher = threading.Thread(target = publish_from_stm, args = [msg_chan, port])
    publisher.daemon = True
    publisher.start()

    compute_gyro_bias(msg_chan)

def publish_from_stm(
        tx_msg_chan: queue.Queue[Tuple[float, float, float]], 
        port : serial.Serial
    ) -> None:

    while True:
        if (reading := read_from_serial(port)) != None:
            tx_msg_chan.put(reading)

def compute_gyro_bias(
        rx_msg_chan : queue.Queue[Tuple[float, float, float]],
        num_samples : int = 2000,
        warmup_samples : int = 100,
        max_std_dps : float = 0.05,
    ) -> Tuple[float, float, float]:
    """
    Reads gyro samples while the sensor is stationary and computes the
    average per-axis bias to subtract from every future reading in firmware.

    num_samples:     how many samples to average over after warmup.
    warmup_samples:  samples discarded first, to let the sensor/filter settle.
    max_std_dps:     if the per-axis standard deviation exceeds this, we warn
                      that the sensor probably wasn't held still.
    """
    print(f"Computing gyro bias — keep the sensor perfectly still.")
    print(f"Discarding {warmup_samples} warmup samples, then averaging {num_samples} samples...")

    # discard warmup samples (let the sensor/filter settle before trusting data)
    for i in range(warmup_samples):
        try:
            rx_msg_chan.get(timeout=5.0)
        except queue.Empty:
            print(f"Timeout during warmup (got {i}/{warmup_samples} samples). Check the connection.")
            return (0.0, 0.0, 0.0)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    while len(xs) < num_samples:
        try:
            x, y, z = rx_msg_chan.get(timeout=5.0)
        except queue.Empty:
            print(f"Timeout waiting for data (got {len(xs)}/{num_samples} samples). "
                  f"Computing bias from what we have.")
            break

        xs.append(x)
        ys.append(y)
        zs.append(z)

        if len(xs) % 500 == 0:
            print(f"  {len(xs)}/{num_samples} samples...")

    if not xs:
        print("No samples collected, aborting.")
        return (0.0, 0.0, 0.0)

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    mean_z = sum(zs) / n

    std_x = (sum((v - mean_x) ** 2 for v in xs) / n) ** 0.5
    std_y = (sum((v - mean_y) ** 2 for v in ys) / n) ** 0.5
    std_z = (sum((v - mean_z) ** 2 for v in zs) / n) ** 0.5

    print(f"\n=== Gyro Bias ({n} samples) ===")
    print(f"  bias_x = {mean_x:.6f}  (std={std_x:.6f})")
    print(f"  bias_y = {mean_y:.6f}  (std={std_y:.6f})")
    print(f"  bias_z = {mean_z:.6f}  (std={std_z:.6f})")

    if max(std_x, std_y, std_z) > max_std_dps:
        print(f"\nWARNING: std dev exceeds {max_std_dps} on at least one axis — "
              f"the sensor probably moved during collection. Re-run while holding it still.")

    print("\nFirmware constants:")
    print(f"const GYRO_BIAS_X: f32 = {mean_x:.6f};")
    print(f"const GYRO_BIAS_Y: f32 = {mean_y:.6f};")
    print(f"const GYRO_BIAS_Z: f32 = {mean_z:.6f};")
    print("\nApply in firmware as: corrected = raw - bias")

    return (mean_x, mean_y, mean_z)

def read_from_serial(port: serial.Serial) -> Optional[Tuple[float, float, float]]:
    header_bytes = struct.pack('I', FRAME_HEADER)
    buffer = bytearray()
    
    while len(buffer) < 4:
        char = port.read(1)
        if not char: 
            return None # Timeout
        buffer.extend(char)
        
        if len(buffer) == 4 and bytes(buffer) != header_bytes:
            buffer.pop(0)

    len_bytes = port.read(4)
    if len(len_bytes) < 4: return None
    payload_len = struct.unpack('I', len_bytes)[0]
        
    payload = port.read(payload_len)
    if len(payload) == payload_len:
        return struct.unpack("<3f", payload)
    
    return None


if __name__ == "__main__":
    main()