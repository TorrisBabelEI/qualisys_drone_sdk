"""
Trajectory loading and interpolation utilities
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_trajectory(csv_file):
    """
    Load trajectory data from CSV file
    
    Args:
    - csv_file: Path to CSV file
    
    Returns:
    - time_data: Time data (actual time, not normalized)
    - positions: Position data (3, N) where 3 rows are x, y, z
    """
    try:
        data = pd.read_csv(csv_file, header=None).values
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        raise
    
    # Ensure there are 7 rows of data
    if data.shape[0] < 7:
        raise ValueError(f"CSV file should have at least 7 rows, got {data.shape[0]} rows")
    
    # Extract time and position data
    time_data = data[0, :].flatten()  # Row 1: time (actual time in seconds)
    x = data[1, :].flatten()  # Row 2: x position
    y = data[2, :].flatten()  # Row 3: y position
    z = data[3, :].flatten()  # Row 4: z position
    
    # Combine position data
    positions = np.vstack([x, y, z])
    
    return time_data, positions


def compute_average_speed(positions, time_data):
    """
    Compute average speed along trajectory
    
    Args:
    - positions: Position data (3, N) where 3 rows are x, y, z
    - time_data: Time data in seconds
    
    Returns:
    - avg_speed: Average speed in m/s
    """
    if len(time_data) < 2:
        raise ValueError("Need at least 2 time points to compute speed")
    
    # Compute distances between consecutive points
    diffs = np.diff(positions, axis=1)
    distances = np.linalg.norm(diffs, axis=0)
    total_distance = np.sum(distances)
    
    # Compute total time
    total_time = time_data[-1] - time_data[0]
    
    if total_time <= 0:
        raise ValueError("Trajectory time must be positive")
    
    avg_speed = total_distance / total_time
    return avg_speed


def validate_trajectory_speed(positions, time_data, speed_limit, safety_margin=0.8):
    """
    Validate that trajectory average speed does not exceed speed limit
    
    Args:
    - positions: Position data (3, N)
    - time_data: Time data in seconds
    - speed_limit: Speed limit in m/s
    - safety_margin: Safety margin (e.g., 0.8 means use 80% of limit)
    
    Returns:
    - is_valid: Boolean indicating if trajectory is valid
    - avg_speed: Computed average speed
    - max_allowed_speed: Maximum allowed speed based on safety margin
    
    Raises:
    - ValueError: If trajectory exceeds speed limit
    """
    avg_speed = compute_average_speed(positions, time_data)
    max_allowed_speed = speed_limit * safety_margin
    
    if avg_speed > max_allowed_speed:
        raise ValueError(
            f"Trajectory average speed ({avg_speed:.3f} m/s) exceeds "
            f"safe limit ({max_allowed_speed:.3f} m/s at {safety_margin*100:.0f}% of "
            f"speed limit {speed_limit:.3f} m/s). "
            f"Please increase flight_time to reduce average speed."
        )
    
    return True, avg_speed, max_allowed_speed


def scale_trajectory_time(time_data, flight_time):
    """
    Scale trajectory time to match desired flight time
    
    Args:
    - time_data: Original time data in seconds
    - flight_time: Desired flight time in seconds
    
    Returns:
    - scaled_time: Time data scaled to flight_time
    
    Note:
    Time is scaled linearly while positions remain unchanged.
    For proper execution, the drone must follow the scaled time.
    """
    if time_data[0] != 0:
        raise ValueError("Time data must start at 0")
    
    original_duration = time_data[-1] - time_data[0]
    if original_duration <= 0:
        raise ValueError("Original trajectory duration must be positive")
    
    scale_factor = flight_time / original_duration
    scaled_time = time_data * scale_factor
    
    return scaled_time


def interpolate_position(t, time_data, positions):
    """
    Perform linear interpolation to get position at a given time
    
    Args:
    - t: Current time in seconds
    - time_data: Trajectory time points (in seconds)
    - positions: Trajectory position data (3, N)
    
    Returns:
    - interpolated_pos: Interpolated position [x, y, z]
    """
    # Create interpolation function for each dimension
    f_x = interp1d(time_data, positions[0, :], kind='linear', fill_value='extrapolate')
    f_y = interp1d(time_data, positions[1, :], kind='linear', fill_value='extrapolate')
    f_z = interp1d(time_data, positions[2, :], kind='linear', fill_value='extrapolate')
    
    # Perform interpolation
    x_interp = float(f_x(t))
    y_interp = float(f_y(t))
    z_interp = float(f_z(t))
    
    return np.array([x_interp, y_interp, z_interp])


def get_trajectory_reference(csv_file, flight_time, speed_limit, safety_margin=0.8):
    """
    Get complete trajectory reference data with validation
    
    Args:
    - csv_file: Path to CSV file
    - flight_time: Desired flight time in seconds (can be None to use original)
    - speed_limit: Speed limit in m/s (from world.speed_limit)
    - safety_margin: Safety margin (e.g., 0.8 means use 80% of limit)
    
    Returns:
    - t_ref: Reference time points (in seconds, scaled if flight_time provided)
    - pos_ref: Reference position data (3, N)
    
    Raises:
    - ValueError: If trajectory exceeds speed limit
    """
    time_data, positions = load_trajectory(csv_file)
    
    # Validate speed with original trajectory time
    validate_trajectory_speed(positions, time_data, speed_limit, safety_margin)
    
    # If flight_time is different from original, scale the time
    if flight_time is not None:
        # Check if scaling would cause speed violations
        scaled_time = scale_trajectory_time(time_data, flight_time)
        validate_trajectory_speed(positions, scaled_time, speed_limit, safety_margin)
        t_ref = scaled_time
    else:
        t_ref = time_data
    
    return t_ref, positions
