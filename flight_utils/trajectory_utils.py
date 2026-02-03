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
    
    # Ensure there are at least 4 rows of data (time, x, y, z)
    if data.shape[0] < 4:
        raise ValueError(f"CSV file should have at least 4 rows, got {data.shape[0]} rows")
    
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


def validate_trajectory_speed(positions, time_data, speed_limit, safety_margin=0.8, raise_on_violation=True):
    """
    Validate that trajectory average speed does not exceed speed limit.

    Supports non-raising mode and returns the minimal flight_time required
    to meet the speed constraint when violated.

    Args:
    - positions: Position data (3, N)
    - time_data: Time data in seconds
    - speed_limit: Speed limit in m/s
    - safety_margin: Safety margin (e.g., 0.8 means use 80% of limit)
    - raise_on_violation: If True (default), raise ValueError on violation.
                         If False, return (is_valid, avg_speed, max_allowed_speed, required_flight_time)

    Returns:
    - If valid: (True, avg_speed, max_allowed_speed, None)
      If violated and raise_on_violation=True: raises ValueError
      If violated and raise_on_violation=False: (False, avg_speed, max_allowed_speed, required_flight_time)
    """
    avg_speed = compute_average_speed(positions, time_data)
    max_allowed_speed = speed_limit * safety_margin

    # compute total_distance and minimal flight_time needed to satisfy speed constraint
    diffs = np.diff(positions, axis=1)
    distances = np.linalg.norm(diffs, axis=0)
    total_distance = np.sum(distances)
    required_flight_time = total_distance / max_allowed_speed

    if avg_speed > max_allowed_speed:
        if raise_on_violation:
            raise ValueError(
                f"Trajectory average speed ({avg_speed:.3f} m/s) exceeds "
                f"safe limit ({max_allowed_speed:.3f} m/s at {safety_margin*100:.0f}% of "
                f"speed limit {speed_limit:.3f} m/s). "
                f"Please increase flight_time to at least {required_flight_time:.3f} s to reduce average speed."
            )
        else:
            return False, avg_speed, max_allowed_speed, required_flight_time

    return True, avg_speed, max_allowed_speed, None


def scale_trajectory_time(time_data, flight_time):
    """
    Scale trajectory time to match desired flight time.

    For equally spaced `time_data` (typical), this replaces the sequence with
    `np.linspace(0, flight_time, num=n_samples)` to ensure uniform spacing and
    precise end-time.
    """
    if time_data[0] != 0:
        raise ValueError("Time data must start at 0")

    n = len(time_data)
    if n < 2:
        raise ValueError("Need at least 2 time points to scale trajectory")

    if flight_time <= 0:
        raise ValueError("flight_time must be positive")

    scaled_time = np.linspace(0, flight_time, num=n)

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


def get_trajectory_reference(csv_file, flight_time, speed_limit, safety_margin=0.8, auto_adjust=False):
    """
    Get complete trajectory reference data with validation.

    Args:
    - csv_file: Path to CSV file
    - flight_time: Desired flight time in seconds (can be None to use original)
    - speed_limit: Speed limit in m/s (from world.speed_limit)
    - safety_margin: Safety margin (e.g., 0.8 means use 80% of limit)
    - auto_adjust: If True, automatically increase `flight_time` to the minimal
                   value required to satisfy speed limits when necessary.

    Returns:
    - t_ref: Reference time points (in seconds, scaled if flight_time provided)
    - pos_ref: Reference position data (3, N)

    Raises:
    - ValueError: If trajectory exceeds speed limit (unless auto_adjust=True)
    """
    time_data, positions = load_trajectory(csv_file)

    # Check original trajectory; get minimal required flight_time if it violates
    valid, avg_speed, max_allowed_speed, required_time = validate_trajectory_speed(
        positions, time_data, speed_limit, safety_margin, raise_on_violation=False
    )

    if not valid:
        if flight_time is None:
            if auto_adjust:
                flight_time = required_time
            else:
                raise ValueError(
                    f"Original trajectory too fast (avg {avg_speed:.3f} m/s > safe {max_allowed_speed:.3f} m/s). "
                    f"Minimum flight_time required: {required_time:.3f} s."
                )
        else:
            if flight_time < required_time:
                if auto_adjust:
                    flight_time = required_time
                else:
                    raise ValueError(
                        f"Provided flight_time ({flight_time:.3f} s) is too short; "
                        f"minimum required is {required_time:.3f} s to meet speed limits."
                    )

    # If flight_time is provided (or possibly auto-adjusted), scale and revalidate
    if flight_time is not None:
        scaled_time = scale_trajectory_time(time_data, flight_time)
        # This will raise if the scaled trajectory still violates the limit
        validate_trajectory_speed(positions, scaled_time, speed_limit, safety_margin)
        t_ref = scaled_time
    else:
        t_ref = time_data

    return t_ref, positions
