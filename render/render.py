import queue
import threading
import time
import serial
from typing import Tuple, Optional
import struct
import pyvista as pv
import numpy as np
import time

FRAME_HEADER = 0x77777777

def main():
    port = serial.Serial('/dev/cu.usbmodem1302', 115200, timeout = 0.1)
    mesh = pv.read("tcsgunship.obj")
    plotter = pv.Plotter()

    # set up channels + threads
    msg_chan : queue.Queue[Tuple[float, float, float]] = queue.Queue()

    # spawn publisher
    publisher = threading.Thread(target = publish_from_stm, args = [msg_chan, port])
    publisher.daemon = True
    publisher.start()

    render(msg_chan, mesh, plotter)

def publish_from_stm(
        tx_msg_chan: queue.Queue[Tuple[float, float, float]], 
        port : serial.Serial
    ) -> None:

    while True:
        if (reading := read_from_serial(port)) != None:
            tx_msg_chan.put(reading)

def render(
        rx_msg_chan : queue.Queue[Tuple[float, float, float]],
        mesh,    # unknown mesh type
        plotter, #unknown plotter type 
    ) -> None:

    mesh.rotate_x(90, inplace=True) 
    mesh.rotate_z(180, inplace=True)

    original_points = mesh.points.copy()

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="cyan", show_edges=True)
    plotter.add_axes()
    plotter.camera_position = [(5, 5, 5), (0, 0, 0), (0, 0, 1)]
    plotter.show(interactive_update=True)

    pitch, roll, yaw = 0.0, 0.0, 0.0
    print("Starting rotation loop. Close the window to exit.")

    try:
        while plotter.render_window.IsCurrent():
            latest_msg = None
            while True:
                try:
                    latest_msg = rx_msg_chan.get_nowait()
                except queue.Empty:
                    break

            if latest_msg is not None:
                pitch, roll, yaw = latest_msg[1], latest_msg[2], latest_msg[0]

            mesh.points = original_points.copy()
            mesh.rotate_x(pitch, inplace=True)
            mesh.rotate_z(roll, inplace=True)
            mesh.rotate_y(yaw, inplace=True)
            
            plotter.update()
            time.sleep(0.001)
            
    except Exception as e:
        print(f"Window closed or error occurred: {e}")

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