"""
Multi Crazyflie Trajectory Tracking

This script makes multiple Crazyflies follow trajectories defined in CSV files.
Each drone can have its own trajectory file or share the same trajectory.
The CSV file should have 7 rows:
  - Row 1: Time in seconds (actual time, with physical meaning)
  - Rows 2-4: X, Y, Z positions
  - Rows 5-7: VX, VY, VZ velocities (not used for tracking)

ESC to land at any time.
"""


import pynput
from time import sleep, time
import sys
import os

from qfly import Pose, QualisysCrazyflie, World, ParallelContexts, utils

import json
import numpy as np

# Add utils to path for custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flight_utils.trajectory_utils import load_trajectory, scale_trajectory_time, interpolate_position, get_trajectory_reference
from flight_utils.flight_data import FlightDataRecorder, FlightDataAnalyzer
from flight_utils.visualization import plot_all_results
from flight_utils.realtime_visualization import RealtimePlot

# SETTINGS
# List of Crazyflie indices to control
cf_indices = [1, 2]

# Load configuration for each Crazyflie
cf_specs_list = []
cf_body_names = []
cf_uris = []
cf_marker_ids_list = []
mocap_ip = None

for cf_idx in cf_indices:
    cf_json = f'config_crazyflie_{cf_idx}.json'
    try:
        with open(cf_json, 'r') as cfg:
            cf_specs = json.load(cfg)
            cf_specs_list.append(cf_specs)
            cf_body_names.append(cf_specs["NAME_SINGLE_BODY"])
            cf_uris.append(cf_specs["URI"])
            # Generate marker IDs based on cf_idx: cf_01 -> [11,12,13,14], cf_02 -> [21,22,23,24], etc.
            marker_ids = [int(f"{cf_idx}{i}") for i in range(1, 5)]
            cf_marker_ids_list.append(marker_ids)
            if mocap_ip is None:
                mocap_ip = cf_specs["QUALISYS_IP"]
    except FileNotFoundError:
        print(f"Config file {cf_json} not found")
        exit(1)

# Trajectory settings (can be customized per drone if needed)
traj_file_name = ['cf_01_traj_ref.csv',
                  'cf_02_traj_ref.csv']  # Will be loaded from traj/ref/
flight_time = 60  # Total flight time in seconds
save_flag = False  # Whether to save flight data
safety_margin = 0.8  # Safety margin for speed check

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

# Initialize data recorders for each drone
recorders = [FlightDataRecorder(cf_idx) for cf_idx in cf_indices]


# Stack up context managers for parallel control
_qcfs = [QualisysCrazyflie(cf_body_name,
                           cf_uri,
                           world,
                           marker_ids=cf_marker_id,
                           qtm_ip=mocap_ip)
         for cf_body_name, cf_uri, cf_marker_id
         in zip(cf_body_names, cf_uris, cf_marker_ids_list)]

with ParallelContexts(*_qcfs) as qcfs:

    # Let there be time
    t_start = time()
    hover_time = 3.0  # Default hover time, will be updated when trajectory actually starts
    trajectory_started = False
    dt = 0
    
    print("Flight completed successfully")
    print("Press ESC to land at any time.")

    # Initialize realtime plot for first drone
    plot = None

    # MAIN LOOP WITH SAFETY CHECK
    while fly and all(qcf.is_safe() for qcf in qcfs):

        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            print("ESC pressed, landing...")
            break

        # Mind the clock
        dt = time() - t_start

        # Take off and hover for up to 8 seconds with position validation
        if dt < 8:
            first_pos = pos_ref[:, 0]  # First column is first waypoint
            target = Pose(first_pos[0], first_pos[1], first_pos[2])
            
            # Send same target to all drones
            for qcf in qcfs:
                qcf.safe_position_setpoint(target)
            
            # Check if all drones are stable at start position (after minimum 2s)
            if dt > 2:
                all_stable = True
                for qcf in qcfs:
                    if qcf.pose is not None:
                        distance_to_start = ((qcf.pose.x - first_pos[0])**2 + 
                                           (qcf.pose.y - first_pos[1])**2 + 
                                           (qcf.pose.z - first_pos[2])**2)**0.5
                        if distance_to_start > 0.15:  # Not within 15cm
                            all_stable = False
                            break
                    else:
                        all_stable = False
                        break
                
                if all_stable:
                    if not trajectory_started:  # Only print once
                        print(f"[t={dt:.1f}s] All drones stable at start position, beginning trajectory...")
                    # Record the actual hover time when trajectory starts
                    hover_time = dt
                    trajectory_started = True
                    # Record initial state at t=0 for each drone
                    first_pos = pos_ref[:, 0]
                    for drone_idx, qcf in enumerate(qcfs):
                        if qcf.pose is not None:
                            recorders[drone_idx].record_state(0, qcf.pose, first_pos)
                    continue
            
            print(f'[t={dt:.1f}s] {"Taking off" if dt < 2 else "Stabilizing"} {len(qcfs)} drones at start position...')
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

        # Cycle all drones
        for drone_idx, qcf in enumerate(qcfs):
            
            # Interpolate desired position from trajectory
            try:
                desired_pos = interpolate_position(traj_time, t_ref, pos_ref)
            except Exception as e:
                print(f"Error interpolating position at t={dt:.2f}s: {e}")
                fly = False
                break

            # Get current actual position
            current_pose = qcf.pose

            # Create target pose with desired position
            target = Pose(desired_pos[0], desired_pos[1], desired_pos[2])

            # Send setpoint to Crazyflie
            qcf.safe_position_setpoint(target)

            # Record data for analysis (only during trajectory tracking)
            if traj_time >= 0:
                recorders[drone_idx].record_state(traj_time, current_pose, desired_pos)

        # Update realtime plot with drone 0 position (reduced frequency)
        if plot is not None and int(dt * 10) % 5 == 0:  # Update at 20Hz
            try:
                first_pose = qcfs[0].pose
                if first_pose is not None:
                    plot.update([first_pose.x, first_pose.y, first_pose.z], info=f't={dt:.1f}s')
            except Exception:
                pass

        # Print progress every 2 seconds
        if int(dt * 2) % 2 == 0 and dt > 0 and int(dt * 10) % 20 == 0:  # Every 2 seconds
            if dt < 8:
                print(f'[t={dt:.1f}s] Hovering {len(qcfs)} drones at start position')
            else:
                print(f'[t={dt:.1f}s] Flying {len(qcfs)} drones...')

        # Small sleep to avoid busy waiting
        sleep(0.01)

    # Land all drones with timeout
    print("Landing all drones...")
    landing_start_time = time()
    LANDING_TIMEOUT = 5  # seconds - force exit if landing takes too long
    
    while all(qcf.pose is not None and qcf.pose.z > 0.1 for qcf in qcfs):
        # Check landing timeout
        if time() - landing_start_time > LANDING_TIMEOUT:
            print(f"Landing timeout after {LANDING_TIMEOUT}s - forcing exit for safety")
            break
        for qcf in qcfs:
            if qcf.pose is not None and qcf.pose.z > 0.1:
                qcf.land_in_place()
        sleep(0.01)

# Save and analyze flight data for each drone
for drone_idx, recorder in enumerate(recorders):
    cf_idx = cf_indices[drone_idx]
    
    if save_flag:
        print(f"\nSaving data for Crazyflie {cf_idx}...")
        csv_file = recorder.save_to_csv(save_dir='traj/out', cf_idx=cf_idx)
        print(f"Data saved to {csv_file}")
        
        # Analyze and print statistics
        analyzer = FlightDataAnalyzer(
            recorder.time_list,
            recorder.pos_actual,
            recorder.pos_desired
        )
        
        stats = analyzer.get_statistics()
        print(f"\n=== Crazyflie {cf_idx} Flight Statistics ===")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Number of samples: {stats['num_samples']}")
        print(f"Position RMS error (XYZ): {stats['pos_rms_error']}")
        print(f"Position mean error (XYZ): {stats['pos_mean_error']}")
        print(f"Position max error (XYZ): {stats['pos_max_error']}")
    else:
        print(f"Crazyflie {cf_idx} flight data not saved (save_flag=False)")

# Plot results if data was saved
if save_flag:
    print("\nPlotting results...")
    for drone_idx, recorder in enumerate(recorders):
        # Set non-interactive backend for post-flight plots
        import matplotlib
        matplotlib.use('Agg')
        
        cf_idx = cf_indices[drone_idx]
        plot_all_results(
            recorder.time_list,
            recorder.pos_actual,
            recorder.pos_desired,
            pos_reference=pos_ref,
            save_figs=False
        )

print("Done!")
