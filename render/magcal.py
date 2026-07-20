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

    # set up channels + threads
    msg_chan : queue.Queue[Tuple[float, float, float]] = queue.Queue()

    # spawn publisher
    publisher = threading.Thread(target = publish_from_stm, args = [msg_chan, port])
    publisher.daemon = True
    publisher.start()

    render(msg_chan)

def publish_from_stm(
        tx_msg_chan: queue.Queue[Tuple[float, float, float]], 
        port : serial.Serial
    ) -> None:

    while True:
        if (reading := read_from_serial(port)) != None:
            tx_msg_chan.put(reading)

def render(
        rx_msg_chan : queue.Queue[Tuple[float, float, float]],
    ) -> None:
    from collections import deque
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Button
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    HISTORY = 5000

    xs: deque[float] = deque(maxlen=HISTORY)
    ys: deque[float] = deque(maxlen=HISTORY)
    zs: deque[float] = deque(maxlen=HISTORY)

    state = {'collecting': True}

    fig = plt.figure(figsize=(20, 6))
    ax_xy = fig.add_subplot(1, 4, 1)
    ax_xz = fig.add_subplot(1, 4, 2)
    ax_yz = fig.add_subplot(1, 4, 3)
    ax_3d = fig.add_subplot(1, 4, 4, projection='3d')

    projections = [
        (ax_xy, 'X', 'Y', 'tab:blue',   'XY'),
        (ax_xz, 'X', 'Z', 'tab:orange', 'XZ'),
        (ax_yz, 'Y', 'Z', 'tab:green',  'YZ'),
    ]

    scatters = {}
    centroid_markers = {}

    for ax, xlabel, ylabel, color, name in projections:
        scatters[name] = ax.scatter([], [], s=8, alpha=0.6, c=color, label='samples')
        centroid_markers[name], = ax.plot([], [], 'r+', markersize=12, markeredgewidth=2, label='centroid')
        ax.plot(0, 0, 'k+', markersize=10, label='origin')
        ax.set_xlabel(f'{xlabel} (uT)')
        ax.set_ylabel(f'{ylabel} (uT)')
        ax.set_title(f'{name} projection')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    scatter_3d = ax_3d.scatter([], [], [], s=8, alpha=0.6, c='tab:purple', label='samples')
    centroid_marker_3d, = ax_3d.plot([], [], [], 'r+', markersize=12, markeredgewidth=2, label='centroid')
    ax_3d.plot([0], [0], [0], 'k+', markersize=10, label='origin')
    ax_3d.set_xlabel('X (uT)')
    ax_3d.set_ylabel('Y (uT)')
    ax_3d.set_zlabel('Z (uT)')
    ax_3d.set_title('3D reconstruction')
    ax_3d.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Magnetometer Projections — Iron Bias Visualization (last {HISTORY} samples)')
    fig.subplots_adjust(bottom=0.2)

    # --- results text box, hidden until calibration is run ---
    results_text = fig.text(
        0.5, 0.02, '', ha='center', va='bottom', fontsize=9, family='monospace'
    )

    def fit_ellipsoid(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Fit a general ellipsoid to points and return (center, soft_iron_matrix)."""
        D = np.column_stack([
            x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z, np.ones_like(x)
        ])
        # homogeneous least squares: smallest right singular vector of D
        _, _, vt = np.linalg.svd(D, full_matrices=False)
        v = vt[-1, :]

        A4 = np.array([
            [v[0], v[3], v[4], v[6]],
            [v[3], v[1], v[5], v[7]],
            [v[4], v[5], v[2], v[8]],
            [v[6], v[7], v[8], v[9]],
        ])
        A3 = A4[:3, :3]
        center = np.linalg.solve(-A3, v[6:9])

        T = np.eye(4)
        T[3, :3] = center
        R = T @ A4 @ T.T

        R3 = R[:3, :3] / -R[3, 3]
        evals, evecs = np.linalg.eigh(R3)
        evals = np.clip(evals, 1e-9, None)  # guard against noise/degenerate fits
        radii = np.sqrt(1.0 / evals)

        mean_radius = float(np.mean(radii))
        soft_iron = evecs @ np.diag(mean_radius / radii) @ evecs.T

        return center, soft_iron

    def update_projection(ax, scatter, centroid_marker, a: deque, b: deque):
        if not a:
            return
        data = np.column_stack([a, b])
        scatter.set_offsets(data)
        ca, cb = float(np.mean(a)), float(np.mean(b))
        centroid_marker.set_data([ca], [cb])
        span = max(np.ptp(a), np.ptp(b), 1.0)
        pad = span * 0.1 + 1.0
        mid_a, mid_b = (min(a) + max(a)) / 2, (min(b) + max(b)) / 2
        half = span / 2 + pad
        ax.set_xlim(mid_a - half, mid_a + half)
        ax.set_ylim(mid_b - half, mid_b + half)

    def update_3d(a: deque, b: deque, c: deque):
        if not a:
            return
        scatter_3d._offsets3d = (list(a), list(b), list(c))
        ca, cb, cc = float(np.mean(a)), float(np.mean(b)), float(np.mean(c))
        centroid_marker_3d.set_data([ca], [cb])
        centroid_marker_3d.set_3d_properties([cc])
        span = max(np.ptp(a), np.ptp(b), np.ptp(c), 1.0)
        pad = span * 0.1 + 1.0
        mid_a, mid_b, mid_c = (min(a) + max(a)) / 2, (min(b) + max(b)) / 2, (min(c) + max(c)) / 2
        half = span / 2 + pad
        ax_3d.set_xlim(mid_a - half, mid_a + half)
        ax_3d.set_ylim(mid_b - half, mid_b + half)
        ax_3d.set_zlim(mid_c - half, mid_c + half)

    def update(_frame):
        if state['collecting']:
            while True:
                try:
                    x, y, z = rx_msg_chan.get_nowait()
                except queue.Empty:
                    break
                xs.append(x)
                ys.append(y)
                zs.append(z)

            update_projection(ax_xy, scatters['XY'], centroid_markers['XY'], xs, ys)
            update_projection(ax_xz, scatters['XZ'], centroid_markers['XZ'], xs, zs)
            update_projection(ax_yz, scatters['YZ'], centroid_markers['YZ'], ys, zs)
            update_3d(xs, ys, zs)

        return list(scatters.values()) + list(centroid_markers.values()) + [scatter_3d, centroid_marker_3d]

    ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)

    # --- Stop/Resume + calibrate button ---
    button_ax = fig.add_axes([0.44, 0.08, 0.12, 0.05])
    button = Button(button_ax, 'Stop && Calibrate')

    def on_click(_event):
        if state['collecting']:
            state['collecting'] = False
            button.label.set_text('Resume')

            if len(xs) < 10:
                results_text.set_text('Not enough points collected to calibrate (need >= 10).')
            else:
                x_arr, y_arr, z_arr = np.array(xs), np.array(ys), np.array(zs)
                try:
                    hard_iron, soft_iron = fit_ellipsoid(x_arr, y_arr, z_arr)
                    np.set_printoptions(precision=4, suppress=True)
                    msg = (
                        f"Hard iron offset (uT):  x={hard_iron[0]:.3f}  y={hard_iron[1]:.3f}  z={hard_iron[2]:.3f}\n"
                        f"Soft iron correction matrix:\n{soft_iron}"
                    )
                    results_text.set_text(msg)
                    print("=== Magnetometer Calibration ===")
                    print(msg)
                    print("\nApply with: calibrated = soft_iron @ (raw - hard_iron)")
                except np.linalg.LinAlgError as e:
                    results_text.set_text(f'Fit failed: {e}. Try collecting more varied orientations.')
        else:
            state['collecting'] = True
            button.label.set_text('Stop && Calibrate')
            results_text.set_text('')

        fig.canvas.draw_idle()

    button.on_clicked(on_click)

    plt.show()

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