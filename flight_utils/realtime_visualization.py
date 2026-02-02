"""Simple real-time 2D visualization for trajectory and drone position."""
import numpy as np
import matplotlib.pyplot as plt


class RealtimePlot:
    def __init__(self, pos_ref=None, window_size=5.0):
        """pos_ref: (3, N) numpy array of reference positions"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.pos_ref = pos_ref
        if pos_ref is not None and pos_ref.shape[1] > 0:
            xs = pos_ref[0, :]
            ys = pos_ref[1, :]
            self.ax.plot(xs, ys, '-', color='gray', alpha=0.6, label='reference')
            margin = max(np.ptp(xs), np.ptp(ys), 0.5) * 0.5 + 0.1
            cx = np.mean(xs)
            cy = np.mean(ys)
            self.ax.set_xlim(cx - margin, cx + margin)
            self.ax.set_ylim(cy - margin, cy + margin)
        else:
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)

        self.current_dot, = self.ax.plot([], [], 'ro', label='drone')
        self.text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Real-time Trajectory & Position')
        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, pos, info=None):
        """Update current drone position. pos is (x, y, z) or array-like."""
        x, y = float(pos[0]), float(pos[1])
        self.current_dot.set_data([x], [y])
        if info is not None:
            self.text.set_text(info)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        try:
            plt.close(self.fig)
        except Exception:
            pass
