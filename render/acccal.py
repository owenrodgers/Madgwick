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

    # --- Set these to match how the sensor is resting on the table ---
    # up_axis: which axis is (anti-)parallel to gravity right now ('x', 'y', or 'z')
    # up_sign: +1 if that axis reads +1g when pointing up, -1 if it reads -1g
    compute_accel_bias(msg_chan, up_axis='z', up_sign=+1)

def publish_from_stm(
        tx_msg_chan: queue.Queue[Tuple[float, float, float]], 
        port : serial.Serial
    ) -> None:

    while True:
        if (reading := read_from_serial(port)) != None:
            tx_msg_chan.put(reading)

def compute_accel_bias(
        rx_msg_chan : queue.Queue[Tuple[float, float, float]],
        up_axis : str = 'z',
        up_sign : int = +1,
        num_samples : int = 2000,
        warmup_samples : int = 100,
        max_std_g : float = 0.01,
    ) -> Tuple[float, float, float]:
    """
    Reads accelerometer samples while the sensor sits still on a flat, level
    surface and computes the per-axis bias to subtract from every future
    reading in firmware.

    Unlike gyro bias, one axis is expected to read +/-g (whichever axis is
    vertical), not zero — so you must tell this function which axis is up
    and in which direction, matching how the sensor is physically resting.

    up_axis:        'x', 'y', or 'z' — the axis currently aligned with gravity.
    up_sign:        +1 if up_axis reads +1g in this orientation, -1 if -1g.
    num_samples:    how many samples to average over after warmup.
    warmup_samples: samples discarded first, to let the sensor/filter settle.
    max_std_g:      if the per-axis standard deviation exceeds this (in g),
                     we warn that the sensor probably wasn't held still.
    """
    if up_axis not in ('x', 'y', 'z'):
        raise ValueError("up_axis must be 'x', 'y', or 'z'")
    if up_sign not in (1, -1):
        raise ValueError("up_sign must be +1 or -1")

    print(f"Computing accelerometer bias — keep the sensor perfectly still and level.")
    print(f"Expecting axis '{up_axis}' to read {up_sign * 1.0:+.5f} g due to gravity; "
          f"other axes should read ~0 g.")
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

    # expected reading in this orientation: 0g on the two horizontal axes,
    # +/-1g on the vertical axis
    expected = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    expected[up_axis] = up_sign * 1.0

    mean = {'x': mean_x, 'y': mean_y, 'z': mean_z}
    bias = {axis: mean[axis] - expected[axis] for axis in ('x', 'y', 'z')}

    print(f"\n=== Accelerometer Bias ({n} samples, units: g) ===")
    print(f"  mean_x = {mean_x:.6f} g  (std={std_x:.6f})  expected={expected['x']:+.6f} g  bias_x={bias['x']:.6f} g")
    print(f"  mean_y = {mean_y:.6f} g  (std={std_y:.6f})  expected={expected['y']:+.6f} g  bias_y={bias['y']:.6f} g")
    print(f"  mean_z = {mean_z:.6f} g  (std={std_z:.6f})  expected={expected['z']:+.6f} g  bias_z={bias['z']:.6f} g")

    if max(std_x, std_y, std_z) > max_std_g:
        print(f"\nWARNING: std dev exceeds {max_std_g} g on at least one axis — "
              f"the sensor probably moved during collection. Re-run while holding it still.")

    # sanity check: does the measured vector magnitude roughly match 1g?
    measured_mag = (mean_x ** 2 + mean_y ** 2 + mean_z ** 2) ** 0.5
    if abs(measured_mag - 1.0) > 0.05:
        print(f"\nWARNING: measured vector magnitude ({measured_mag:.4f} g) is far from "
              f"expected 1 g. Check up_axis/up_sign, or the sensor may not be level.")

    print("\nFirmware constants (in g):")
    print(f"const ACCEL_BIAS_X: f32 = {bias['x']:.6f};")
    print(f"const ACCEL_BIAS_Y: f32 = {bias['y']:.6f};")
    print(f"const ACCEL_BIAS_Z: f32 = {bias['z']:.6f};")
    print("\nApply in firmware as: corrected = raw - bias  (raw and bias both in g)")

    return (bias['x'], bias['y'], bias['z'])

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