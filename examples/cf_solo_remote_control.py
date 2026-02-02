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

# Input and realtime visualization helpers
from flight_utils.input_handlers import KeyboardController, JoystickController
from flight_utils.realtime_visualization import RealtimePlot

# Add utils to path for custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flight_utils.trajectory_utils import load_trajectory, scale_trajectory_time, interpolate_position, get_trajectory_reference
from flight_utils.flight_data import FlightDataRecorder, FlightDataAnalyzer
from flight_utils.visualization import plot_all_results

# --- Configuration for manual control ---
# Supported devices: 'keyboard' | 'joystick'
INPUT_DEVICE = 'keyboard'
MAX_FLIGHT_TIME = 100  # seconds from hover start
MOVEMENT_STEP = 0.002  # meters per command (approx per 0.01s loop -> ~0.2 m/s)
CONTROL_RATE = 100.0  # Hz approximate
DEADZONE_JOYSTICK = 0.2


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

    print("Beginning remote-control session...")
    print("Press ESC to land at any time.")

    # Initialize input controller
    controller = None
    try:
        if INPUT_DEVICE == 'keyboard':
            controller = KeyboardController()
            controller.start()
        elif INPUT_DEVICE == 'joystick':
            controller = JoystickController(deadzone=DEADZONE_JOYSTICK)
        else:
            print(f"Unknown INPUT_DEVICE='{INPUT_DEVICE}', defaulting to keyboard")
            controller = KeyboardController(); controller.start()
    except Exception as e:
        print(f"Input device init failed: {e}. Falling back to keyboard behavior.")
        controller = KeyboardController(); controller.start()

    # Initialize realtime plot
    try:
        plot = RealtimePlot(pos_ref, lab_xlim=lab_xlim, lab_ylim=lab_ylim)
        # Show operation instructions on plot
        instr = (
            f"Device: {INPUT_DEVICE} | Arrows/Joystick to move | ESC to land | Max {MAX_FLIGHT_TIME}s\n"
            "Move step: {MOVEMENT_STEP} m per tick"
        )
        plot.set_instructions(instr)
    except Exception as e:
        print(f"Failed to initialize realtime plot: {e}")
        plot = None

    # TAKEOFF -> hover at safe expanse
    hover_target = Pose(world.origin.x, world.origin.y, world.expanse)
    print("Taking off and ascending to hover altitude...")
    takeoff_start = time()
    takeoff_timeout = 12.0  # seconds
    while time() - takeoff_start < takeoff_timeout:
        qcf.safe_position_setpoint(hover_target)
        sleep(0.02)
        if qcf.pose is not None and qcf.pose.z > min(0.4, world.expanse * 0.5):
            break

    # Wait a short moment for stable hover
    stable_start = time()
    while time() - stable_start < 1.0:
        qcf.safe_position_setpoint(hover_target)
        sleep(0.02)

    # Start control timer from now
    hover_start_time = time()
    print(f"Hover achieved. Controls enabled (device={INPUT_DEVICE}). Max flight time: {MAX_FLIGHT_TIME}s")

    # MAIN CONTROL LOOP
    last_progress_print = 0
    while fly and qcf.is_safe():

        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            print("ESC pressed, landing...")
            break

        # Elapsed control time
        elapsed = time() - hover_start_time
        if elapsed > MAX_FLIGHT_TIME:
            print(f"Max flight time reached ({MAX_FLIGHT_TIME}s), landing...")
            break

        # Read inputs
        dx = dy = 0.0
        try:
            if controller is not None:
                dir_x, dir_y = controller.get_direction()
                dx = dir_x * MOVEMENT_STEP
                dy = dir_y * MOVEMENT_STEP
        except Exception as e:
            # If joystick fails, ignore and continue hovering
            dx = dy = 0.0

        # Update hover target position based on inputs (XY only)
        if dx != 0.0 or dy != 0.0:
            new_x = hover_target.x + dx
            new_y = hover_target.y + dy
            try:
                from flight_utils.bounds import clamp_xy
                new_x, new_y = clamp_xy(new_x, new_y, lab_xlim, lab_ylim)
            except Exception:
                # Fallback to world bounds if clamp not available
                xmin = world.origin.x - world.expanse
                xmax = world.origin.x + world.expanse
                ymin = world.origin.y - world.expanse
                ymax = world.origin.y + world.expanse
                new_x = min(max(new_x, xmin), xmax)
                new_y = min(max(new_y, ymin), ymax)

            hover_target = Pose(
                new_x,
                new_y,
                hover_target.z,
            )

        # Send setpoint
        qcf.safe_position_setpoint(hover_target)

        # Record data (use elapsed as timestamp relative to hover start)
        try:
            cur_pose = qcf.pose
        except Exception:
            cur_pose = None

        if cur_pose is not None:
            recorder.record_state(elapsed, cur_pose, np.array([hover_target.x, hover_target.y, hover_target.z]))

        # Update realtime plot
        if plot is not None and cur_pose is not None:
            info = f't={elapsed:.1f}s z={cur_pose.z:.2f}m'
            try:
                plot.update([cur_pose.x, cur_pose.y, cur_pose.z], info=info)
            except Exception:
                pass

        # Print status every second
        if int(elapsed) != last_progress_print and elapsed > 0:
            last_progress_print = int(elapsed)
            if cur_pose is not None:
                print(f'[t={elapsed:.1f}s] Pos: ({cur_pose.x:.3f}, {cur_pose.y:.3f}, {cur_pose.z:.3f}) m')

        sleep(1.0 / CONTROL_RATE)

    # Clean up
    try:
        if controller is not None and hasattr(controller, 'stop'):
            controller.stop()
    except Exception:
        pass

    if plot is not None:
        try:
            plot.close()
        except Exception:
            pass

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