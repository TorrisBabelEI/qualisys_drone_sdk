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

from utils.trajectory_utils import load_trajectory, scale_trajectory_time, interpolate_position, get_trajectory_reference
from utils.flight_data import FlightDataRecorder, FlightDataAnalyzer
from utils.visualization import plot_all_results

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
traj_file_name = 'cf_solo_trajectory.csv'  # Trajectory file name (will be loaded from traj/ref/)
flight_time = 60  # Total flight time in seconds (can be different from trajectory original time)
save_flag = False  # Whether to save flight data to CSV
safety_margin = 0.8  # Safety margin for speed check (use 80% of speed limit)


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


# Set up world - the World object comes with sane defaults
world = World()

# Load trajectory from CSV file
traj_path = os.path.join('traj', 'ref', traj_file_name)
print(f"Loading trajectory from {traj_path}...")
try:
    t_ref, pos_ref = get_trajectory_reference(
        traj_path, 
        flight_time,
        world.speed_limit,
        safety_margin
    )
    print(f"Trajectory loaded: {pos_ref.shape[1]} waypoints, flight time: {flight_time}s")
except Exception as e:
    print(f"Error loading trajectory: {e}")
    exit(1)

# Initialize data recorder
recorder = FlightDataRecorder(cf_idx)


# Prepare for liftoff
with QualisysCrazyflie(cf_body_name,
                       cf_uri,
                       world,
                       marker_ids=cf_marker_ids,
                       qtm_ip=mocap_ip) as qcf:

    # Let there be time
    t_start = time()
    dt = 0
    
    print("Beginning trajectory tracking...")
    print("Press ESC to land at any time.")

    # MAIN LOOP WITH SAFETY CHECK
    while fly and qcf.is_safe():

        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            print("ESC pressed, landing...")
            break

        # Mind the clock
        dt = time() - t_start

        # Check if trajectory is completed
        if dt > flight_time:
            print(f"Trajectory completed at t={dt:.2f}s")
            break

        # Interpolate desired position from trajectory
        try:
            desired_pos = interpolate_position(dt, t_ref, pos_ref)
        except Exception as e:
            print(f"Error interpolating position at t={dt:.2f}s: {e}")
            break

        # Get current actual position
        current_pose = qcf.pose

        # Create target pose with desired position
        target = Pose(desired_pos[0], desired_pos[1], desired_pos[2])

        # Send setpoint to Crazyflie
        qcf.safe_position_setpoint(target)

        # Record data for analysis
        recorder.record_state(dt, current_pose, desired_pos)

        # Print progress every 1 second
        if int(dt) % 1 == 0 and dt > 0:
            print(f'[t={dt:.2f}s] Pos: ({current_pose.x:.3f}, {current_pose.y:.3f}, {current_pose.z:.3f}) m')

        # Small sleep to avoid busy waiting
        sleep(0.01)

    # Land
    print("Landing...")
    while qcf.pose.z > 0.1:
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
        recorder.vel_actual,
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
    plot_all_results(
        recorder.time_list,
        recorder.pos_actual,
        recorder.pos_desired,
        recorder.vel_actual,
        pos_reference=pos_ref
    )
else:
    print("Flight data not saved (save_flag=False)")

print("Done!")