"""
Flight data visualization utilities
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


def plot_trajectory_3d(pos_actual, pos_desired, pos_reference=None):
    """
    Plot 3D trajectory comparison
    
    Args:
    - pos_actual: Actual position (3, N)
    - pos_desired: Desired position (3, N)
    - pos_reference: Reference trajectory position (3, N) (optional)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot reference trajectory
    if pos_reference is not None:
        ax.plot(pos_reference[0, :], pos_reference[1, :], pos_reference[2, :],
                color='blue', linestyle='-', linewidth=2, label='Reference Trajectory')
        ax.plot([pos_reference[0, 0]], [pos_reference[1, 0]], [pos_reference[2, 0]],
                marker='D', color='blue', markersize=8, label='Reference Start')
        ax.plot([pos_reference[0, -1]], [pos_reference[1, -1]], [pos_reference[2, -1]],
                marker='X', color='blue', markersize=8, label='Reference End')
    
    # Plot desired trajectory
    ax.plot(pos_desired[0, :], pos_desired[1, :], pos_desired[2, :],
            color='green', linestyle='--', linewidth=1.5, label='Desired Trajectory')
    
    # Plot actual trajectory
    ax.plot(pos_actual[0, :], pos_actual[1, :], pos_actual[2, :],
            color='red', linestyle='-', linewidth=2, label='Actual Trajectory')
    ax.plot([pos_actual[0, 0]], [pos_actual[1, 0]], [pos_actual[2, 0]],
            marker='s', color='red', markersize=8, label='Actual Start')
    ax.plot([pos_actual[0, -1]], [pos_actual[1, -1]], [pos_actual[2, -1]],
            marker='v', color='red', markersize=8, label='Actual End')
    
    # Adjust axis range
    if pos_reference is not None:
        all_pos = np.concatenate([pos_reference, pos_desired, pos_actual], axis=1)
    else:
        all_pos = np.concatenate([pos_desired, pos_actual], axis=1)
    
    X = all_pos[0, :]
    Y = all_pos[1, :]
    Z = all_pos[2, :]
    
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Trajectory Comparison')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    return fig, ax


def plot_position_tracking(time_list, pos_actual, pos_desired):
    """
    Plot position tracking curves
    
    Args:
    - time_list: Time list
    - pos_actual: Actual position (3, N)
    - pos_desired: Desired position (3, N)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    
    # X axis
    ax1.plot(time_list, pos_actual[0, :], 'r-', linewidth=2, label='Actual X')
    ax1.plot(time_list, pos_desired[0, :], 'b--', linewidth=1.5, label='Desired X')
    ax1.set_ylabel('X [m]')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title('Position Tracking: X axis')
    
    # Y axis
    ax2.plot(time_list, pos_actual[1, :], 'r-', linewidth=2, label='Actual Y')
    ax2.plot(time_list, pos_desired[1, :], 'b--', linewidth=1.5, label='Desired Y')
    ax2.set_ylabel('Y [m]')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_title('Position Tracking: Y axis')
    
    # Z axis
    ax3.plot(time_list, pos_actual[2, :], 'r-', linewidth=2, label='Actual Z')
    ax3.plot(time_list, pos_desired[2, :], 'b--', linewidth=1.5, label='Desired Z')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Z [m]')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    ax3.set_title('Position Tracking: Z axis')
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_tracking_error(time_list, pos_actual, pos_desired):
    """
    Plot position tracking error
    
    Args:
    - time_list: Time list
    - pos_actual: Actual position (3, N)
    - pos_desired: Desired position (3, N)
    """
    error = pos_desired - pos_actual
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    
    # X error
    ax1.plot(time_list, error[0, :], 'r-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Error [m]')
    ax1.grid(True)
    ax1.set_title(f'Position Tracking Error: X axis (RMS: {np.sqrt(np.mean(error[0, :]**2)):.4f} m)')
    
    # Y error
    ax2.plot(time_list, error[1, :], 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Error [m]')
    ax2.grid(True)
    ax2.set_title(f'Position Tracking Error: Y axis (RMS: {np.sqrt(np.mean(error[1, :]**2)):.4f} m)')
    
    # Z error
    ax3.plot(time_list, error[2, :], 'b-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Error [m]')
    ax3.grid(True)
    ax3.set_title(f'Position Tracking Error: Z axis (RMS: {np.sqrt(np.mean(error[2, :]**2)):.4f} m)')
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_velocity(time_list, vel_actual):
    """
    Plot velocity curves
    
    Args:
    - time_list: Time list
    - vel_actual: Actual velocity (3, N)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    
    # VX
    ax1.plot(time_list, vel_actual[0, :], 'r-', linewidth=2)
    ax1.set_ylabel('VX [m/s]')
    ax1.grid(True)
    ax1.set_title('Velocity: X axis')
    
    # VY
    ax2.plot(time_list, vel_actual[1, :], 'g-', linewidth=2)
    ax2.set_ylabel('VY [m/s]')
    ax2.grid(True)
    ax2.set_title('Velocity: Y axis')
    
    # VZ
    ax3.plot(time_list, vel_actual[2, :], 'b-', linewidth=2)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('VZ [m/s]')
    ax3.grid(True)
    ax3.set_title('Velocity: Z axis')
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_all_results(time_list, pos_actual, pos_desired, 
                     pos_reference=None, save_figs=False, fig_dir=None):
    """
    Plot all results
    
    Args:
    - time_list: Time list
    - pos_actual: Actual position (3, N)
    - pos_desired: Desired position (3, N)
    - pos_reference: Reference trajectory (optional)
    - save_figs: Whether to save figures
    - fig_dir: Directory to save figures
    """
    figs = []
    
    # 3D trajectory
    fig1, _ = plot_trajectory_3d(pos_actual, pos_desired, pos_reference)
    figs.append(('3D_Trajectory', fig1))
    
    # Position tracking
    fig2, _ = plot_position_tracking(time_list, pos_actual, pos_desired)
    figs.append(('Position_Tracking', fig2))
    
    # Tracking error
    fig3, _ = plot_tracking_error(time_list, pos_actual, pos_desired)
    figs.append(('Tracking_Error', fig3))
    
    if save_figs and fig_dir:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        for fig_name, fig in figs:
            fig_path = os.path.join(fig_dir, f'{fig_name}.png')
            fig.savefig(fig_path, dpi=150)
            print(f"Figure saved: {fig_path}")
    
    plt.show()
    return figs
