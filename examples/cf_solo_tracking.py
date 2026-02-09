"""
Solo Crazyflie Trajectory Tracking

This script makes a Crazyflie follow a trajectory defined in a CSV file.
The CSV file should have 7 rows:
  - Row 1: Time in seconds (actual time, with physical meaning)
  - Rows 2-4: X, Y, Z positions
  - Rows 5-7: VX, VY, VZ velocities (not used for tracking)

The trajectory will be checked to ensure average speed does not exceed
the drone's speed limit. The flight_time parameter can be used to scale
the trajectory execution time.

ESC to land at any time.
"""


import pynput
from time import sleep, time
import sys
import os

from qfly import Pose, QualisysCrazyflie, World, utils

import json
import numpy as np

# Add utils to path for custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flight_utils.trajectory_utils import load_trajectory, scale_trajectory_time, interpolate_position, get_trajectory_reference
from flight_utils.flight_data import FlightDataRecorder, FlightDataAnalyzer
from flight_utils.visualization import plot_all_results
from flight_utils.realtime_visualization import RealtimePlot

# Select required crazyflie index
cf_idx = 1

cf_json = f'config_crazyflie_{cf_idx}.json'
with open(cf_json, 'r') as cfg:
    cf_specs = json.load(cfg)

# SETTINGS
cf_body_name = cf_specs["NAME_SINGLE_BODY"]  # QTM rigid body name
cf_uri = cf_specs["URI"]  # Crazyflie address
cf_marker_ids = [1, 2, 3, 4] # Active marker IDs
mocap_ip = cf_specs["QUALISYS_IP"]  # IP address for QTM capture data
traj_file_name = 'safe_trajectory.csv'  # Trajectory file name (will be loaded from traj/ref/)
flight_time = 40  # Total flight time in seconds (can be different from trajectory original time)
save_flag = True  # Whether to save flight data to CSV
safety_margin = 0.8  # Safety margin for speed check (use 80% of speed limit)

# Lab limits (x_min, x_max), (y_min, y_max)
lab_xlim = (-2.4, 2.4)
lab_ylim = (-1.8, 1.6)


# Watch key presses with a global variable
last_key_pressed = None
fly = True


# Set up keyboard callback
def on_press(key):
    """React to keyboard."""
    global last_key_pressed, fly
    last_key_pressed = key
    if key == pynput.keyboard.Key.esc:
        fly = False


# Listen to the keyboard
listener = pynput.keyboard.Listener(on_press=on_press)
listener.start()


# Set up world with expanse covering lab space
world = World(expanse=2.5)

# Load trajectory from CSV file
traj_path = os.path.join('traj', 'ref', traj_file_name)
try:
    t_ref, pos_ref = get_trajectory_reference(
        traj_path, 
        flight_time,
        world.speed_limit,
        safety_margin
    )
    print(f"Trajectory loaded: {pos_ref.shape[1]} waypoints, flight time: {flight_time}s")
    
    # Validate trajectory is within lab bounds
    x_vals, y_vals = pos_ref[0, :], pos_ref[1, :]
    if np.any(x_vals < lab_xlim[0]) or np.any(x_vals > lab_xlim[1]):
        print(f"ERROR: Trajectory X values [{x_vals.min():.2f}, {x_vals.max():.2f}] exceed lab limits {lab_xlim}")
        exit(1)
    if np.any(y_vals < lab_ylim[0]) or np.any(y_vals > lab_ylim[1]):
        print(f"ERROR: Trajectory Y values [{y_vals.min():.2f}, {y_vals.max():.2f}] exceed lab limits {lab_ylim}")
        exit(1)
    print(f"Trajectory validated within lab bounds X{lab_xlim}, Y{lab_ylim}")
    
except Exception as e:
    print(f"Error loading trajectory: {e}")
    exit(1)

# Initialize data recorder
recorder = FlightDataRecorder(cf_idx)

# Initialize realtime plot (optional)
try:
    plot = RealtimePlot(pos_ref, lab_xlim=lab_xlim, lab_ylim=lab_ylim)
    plot.set_instructions('Tracking mode | No input control | ESC to abort')
except Exception as e:
    print(f"Realtime plot unavailable: {e}")
    plot = None


# Prepare for liftoff
with QualisysCrazyflie(cf_body_name,
                       cf_uri,
                       world,
                       marker_ids=cf_marker_ids,
                       qtm_ip=mocap_ip) as qcf:

    # Let there be time
    t_start = time()
    hover_time = 3.0  # Default hover time, will be updated when trajectory actually starts
    trajectory_started = False
    dt = 0
    
    print("Flight completed successfully")
    print("Press ESC to land at any time.")

    # MAIN LOOP WITH SAFETY CHECK
    while fly and qcf.is_safe():

        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            print("ESC pressed, landing...")
            break

        # Mind the clock
        dt = time() - t_start

        # Take off and hover for up to 8 seconds with position validation
        if dt < 8:
            # Get current actual position
            current_pose = qcf.pose
            
            first_pos = pos_ref[:, 0]  # First column is first waypoint
            target = Pose(first_pos[0], first_pos[1], first_pos[2])
            qcf.safe_position_setpoint(target)
            
            # Check if we're stable at start position (after minimum 2s)
            if current_pose is not None and dt > 2:
                distance_to_start = ((current_pose.x - first_pos[0])**2 + 
                                   (current_pose.y - first_pos[1])**2 + 
                                   (current_pose.z - first_pos[2])**2)**0.5
                if distance_to_start < 0.30:  # Within 30cm of start point
                    if not trajectory_started:  # Only print once
                        print(f"[t={dt:.1f}s] Stable at start position, beginning trajectory...")
                    # Record the actual hover time when trajectory starts
                    hover_time = dt
                    trajectory_started = True
                    continue
            
            print(f'[t={dt:.1f}s] {"Taking off" if dt < 2 else "Stabilizing"} at start position...')
            continue

        # Calculate trajectory time (time since trajectory started)
        traj_time = dt - hover_time

        # Check if trajectory is completed
        if traj_time > flight_time:
            print(f"Trajectory completed at t={dt:.2f}s")
            break

        # Start trajectory tracking after hover phase
        if not trajectory_started and traj_time >= 0:
            print("Beginning trajectory tracking...")
            trajectory_started = True

        # Interpolate desired position from trajectory
        try:
            desired_pos = interpolate_position(traj_time, t_ref, pos_ref)
        except Exception as e:
            print(f"Error interpolating position at t={dt:.2f}s: {e}")
            break

        # Get current actual position
        current_pose = qcf.pose

        # Create target pose with desired position
        target = Pose(desired_pos[0], desired_pos[1], desired_pos[2])

        # Send setpoint to Crazyflie
        qcf.safe_position_setpoint(target)

        # Record data for analysis (only during trajectory tracking)
        if traj_time >= 0:
            recorder.record_state(traj_time, current_pose, desired_pos)

        # Update realtime plot (reduced frequency)
        if plot is not None and current_pose is not None and int(dt * 10) % 5 == 0:  # Update at 20Hz
            try:
                plot.update([current_pose.x, current_pose.y, current_pose.z], info=f't={dt:.1f}s')
            except Exception:
                pass

        # Print progress every 2 seconds
        if int(dt) % 2 == 0 and dt > 0 and int(dt * 10) % 20 == 0:  # Every 2 seconds
            if current_pose is not None:
                print(f'[t={dt:.1f}s] Pos: ({current_pose.x:.2f}, {current_pose.y:.2f}, {current_pose.z:.2f}) m')

        # Small sleep to avoid busy waiting
        sleep(0.01)

    # Land with timeout
    print("Landing...")
    landing_start_time = time()
    LANDING_TIMEOUT = 3  # seconds - force exit if landing takes too long
    
    while qcf.pose is not None and qcf.pose.z > 0.1:
        # Check landing timeout
        if time() - landing_start_time > LANDING_TIMEOUT:
            print(f"Landing timeout after {LANDING_TIMEOUT}s - forcing exit for safety")
            break
        qcf.land_in_place()
        sleep(0.01)

# Save flight data if requested
if save_flag:
    print("Saving flight data...")
    csv_file = recorder.save_to_csv(save_dir='traj/out', cf_idx=cf_idx)
    print(f"Data saved to {csv_file}")
    
    # Analyze and print statistics
    analyzer = FlightDataAnalyzer(
        recorder.time_list,
        recorder.pos_actual,
        recorder.pos_desired
    )
    
    stats = analyzer.get_statistics()
    print("\n=== Flight Statistics ===")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Position RMS error (XYZ): {stats['pos_rms_error']}")
    print(f"Position mean error (XYZ): {stats['pos_mean_error']}")
    print(f"Position max error (XYZ): {stats['pos_max_error']}")
    
    # Plot results
    print("Plotting results...")
    # Set non-interactive backend for post-flight plots
    import matplotlib
    matplotlib.use('Agg')
    
    plot_all_results(
        recorder.time_list,
        recorder.pos_actual,
        recorder.pos_desired,
        pos_reference=pos_ref
    )
else:
    print("Flight data not saved (save_flag=False)")

print("Done!")