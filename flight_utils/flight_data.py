"""
UAV flight data recording, saving and processing utilities
"""

import numpy as np
import os
import csv
from datetime import datetime


class FlightDataRecorder:
    """Record data during UAV flight process"""
    
    def __init__(self, cf_idx):
        """
        Initialize recorder
        
        Args:
        - cf_idx: Crazyflie index number
        """
        self.cf_idx = cf_idx
        
        # Data storage
        self.time_list = []
        self.pos_actual = np.empty((3, 0))  # Actual position (x, y, z)
        self.vel_actual = np.empty((3, 0))  # Actual velocity (vx, vy, vz)
        self.pos_desired = np.empty((3, 0))  # Desired position (x, y, z)
        self.idx_iter = 0  # Iteration counter
        
    def record_state(self, t, actual_pose, desired_pos):
        """
        Record state at a given time
        
        Args:
        - t: Current time in seconds
        - actual_pose: Actual pose (Pose object with x, y, z, vx, vy, vz)
        - desired_pos: Desired position [x, y, z]
        """
        self.time_list.append(t)
        
        # Record actual position
        actual_pos = np.array([[actual_pose.x], [actual_pose.y], [actual_pose.z]])
        self.pos_actual = np.concatenate((self.pos_actual, actual_pos), axis=1)
        
        # Record actual velocity
        actual_vel = np.array([[actual_pose.vx], [actual_pose.vy], [actual_pose.vz]])
        self.vel_actual = np.concatenate((self.vel_actual, actual_vel), axis=1)
        
        # Record desired position
        desired_pos_array = np.array([[desired_pos[0]], [desired_pos[1]], [desired_pos[2]]])
        self.pos_desired = np.concatenate((self.pos_desired, desired_pos_array), axis=1)
        
        self.idx_iter += 1
        
    def save_to_csv(self, save_dir='traj/out', cf_idx=None):
        """
        Save flight data to CSV file
        
        Args:
        - save_dir: Directory to save data (default: traj/out)
        - cf_idx: Crazyflie index (if None, use initialized value)
        """
        if cf_idx is None:
            cf_idx = self.cf_idx
            
        # Create save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Generate filename
        time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = os.path.join(save_dir, f"cf_{cf_idx:02d}_{time_str}.csv")
        
        # Prepare data
        # Row layout: time, x_actual, y_actual, z_actual, vx_actual, vy_actual, vz_actual,
        #            x_desired, y_desired, z_desired
        data_to_save = np.vstack([
            np.array(self.time_list),
            self.pos_actual,
            self.vel_actual,
            self.pos_desired
        ])
        
        # Save as CSV
        np.savetxt(file_name, data_to_save, delimiter=',')
        
        print(f"Flight data saved to: {file_name}")
        return file_name


class FlightDataAnalyzer:
    """Analyze flight data and compute statistics"""
    
    def __init__(self, time_list, pos_actual, vel_actual, pos_desired):
        """
        Initialize analyzer
        
        Args:
        - time_list: Time list
        - pos_actual: Actual position (3, N)
        - vel_actual: Actual velocity (3, N)
        - pos_desired: Desired position (3, N)
        """
        self.time_list = np.array(time_list)
        self.pos_actual = pos_actual
        self.vel_actual = vel_actual
        self.pos_desired = pos_desired
        
    def compute_tracking_error(self):
        """
        Compute position tracking error
        
        Returns:
        - error: Position error (3, N)
        - rms_error: RMS error for each dimension
        """
        error = self.pos_desired - self.pos_actual
        rms_error = np.sqrt(np.mean(error**2, axis=1))
        return error, rms_error
        
    def get_statistics(self):
        """
        Acquire flight statistics
        
        Returns:
        - stats: Dictionary containing various statistics
        """
        error, rms_error = self.compute_tracking_error()
        
        stats = {
            'total_time': self.time_list[-1] if len(self.time_list) > 0 else 0,
            'num_samples': len(self.time_list),
            'pos_rms_error': rms_error,
            'pos_mean_error': np.mean(np.abs(error), axis=1),
            'pos_max_error': np.max(np.abs(error), axis=1),
        }
        
        return stats
