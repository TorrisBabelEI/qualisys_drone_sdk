"""
Solo Crazyflie Remote Control

This script allows manual control of a Crazyflie using keyboard or joystick input.
The drone will take off to hover altitude and then respond to user input commands.

Controls:
KEYBOARD:
- Arrow keys: Move in XY plane (horizontal movement)
- W key: Increase altitude by 0.001m per press
- S key: Decrease altitude by 0.001m per press
- ESC: Land

JOYSTICK:
- Right stick: Move in XY plane (horizontal movement)
- Left stick up/down: Increase/decrease altitude (variable speed based on deflection)
- Back/Select button (button 6/left at the front): Land

Safety Features:
- Altitude limits: 0.4m minimum, 1.8m maximum
- XY movement constrained within lab boundaries
- MOCAP tracking loss protection (automatic landing if tracking fails)

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

# Input and realtime visualization helpers
from flight_utils.input_handlers import KeyboardController, JoystickController
from flight_utils.realtime_visualization import RealtimePlot
from flight_utils.bounds import clamp_xy
from flight_utils.trajectory_utils import load_trajectory, scale_trajectory_time, interpolate_position, get_trajectory_reference
from flight_utils.flight_data import FlightDataRecorder, FlightDataAnalyzer
from flight_utils.visualization import plot_all_results

# --- Configuration for manual control ---
# Supported devices: 'keyboard' | 'joystick'
INPUT_DEVICE = 'keyboard'
MAX_FLIGHT_TIME = 100  # seconds from hover start
MOVEMENT_STEP = 1e-3  # meters per command (approx per 0.01s loop -> ~0.2 m/s)
ALTITUDE_STEP = 5e-4  # meters per command for keyboard altitude control
MIN_ALTITUDE = 0.4  # minimum altitude in meters
MAX_ALTITUDE = 1.8  # maximum altitude in meters
DEADZONE_JOYSTICK = 0.2


# Select required crazyflie index
cf_idx = 5

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


# Set up world with expanse covering lab space
world = World(expanse=2.5)

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
except Exception:
    print('Reference Trajectory is Not Found. Pure Remote Control Mode.')

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
            controller = KeyboardController()
            controller.start()
    except Exception as e:
        print(f"Input device init failed: {e}. Falling back to keyboard behavior.")
        controller = KeyboardController()
        controller.start()

    # Initialize realtime plot
    try:
        plot = RealtimePlot(pos_ref, lab_xlim=lab_xlim, lab_ylim=lab_ylim)
        # Show operation instructions on plot
        instr = (
            f"Device: {INPUT_DEVICE} | Arrows/Right stick: XY | W/S/Left stick: altitude | ESC/Back button: land\n"
            f"XY step: {MOVEMENT_STEP}m | Z step: {ALTITUDE_STEP}m | Alt limits: {MIN_ALTITUDE}-{MAX_ALTITUDE}m"
        )
        plot.set_instructions(instr)
    except Exception as e:
        print(f"Failed to initialize realtime plot: {e}")
        plot = None

    # TAKEOFF -> determine hover target based on current position if available
    initial_hover_xy = None
    hover_target = None
    print("Taking off and ascending to hover altitude...")
    takeoff_start = time()
    takeoff_timeout = 12.0  # seconds
    while time() - takeoff_start < takeoff_timeout:
        # If we have a current pose and haven't set hover xy, capture it as the hover target
        if qcf.pose is not None and initial_hover_xy is None:
            try:
                x0 = float(qcf.pose.x)
                y0 = float(qcf.pose.y)
                # If the current pose is outside lab limits, abort for safety
                xmin, xmax = lab_xlim
                ymin, ymax = lab_ylim
                if not (xmin <= x0 <= xmax and ymin <= y0 <= ymax):
                    print(f"Initial pose (x={x0:.3f}, y={y0:.3f}) outside lab limits {lab_xlim, lab_ylim}. Aborting and landing for safety.")
                    # immediate landing
                    while qcf.pose is not None and qcf.pose.z > 0.1:
                        qcf.land_in_place()
                        sleep(0.01)
                    print("Landed. Exiting.")
                    sys.exit(1)

                # Accept initial position as hover target (already checked within lab limits)
                initial_hover_xy = (x0, y0)
                hover_target = Pose(x0, y0, world.expanse)
            except Exception:
                initial_hover_xy = None

        # While waiting for a valid current pose, do NOT fallback to world origin (avoid overshoot).
        # Only send a setpoint when we have a concrete hover_target (derived from real pose).
        if hover_target is not None:
            qcf.safe_position_setpoint(hover_target)

        sleep(0.02)
        if qcf.pose is not None and qcf.pose.z > 0.6 and hover_target is not None:
            break

    # If after timeout we still don't have a valid hover target, abort for safety
    if hover_target is None:
        print(f"Failed to obtain a valid initial pose within {takeoff_timeout} seconds. Aborting and landing for safety.")
        try:
            while qcf.pose is not None and qcf.pose.z > 0.1:
                qcf.land_in_place()
                sleep(0.01)
        except Exception:
            pass
        sys.exit(1)

    # Wait a short moment for stable hover and ensure hover remains within lab limits
    stable_start = time()
    while time() - stable_start < 1.0:
        # Update hover target if we have a more accurate current pose
        if qcf.pose is not None:
            try:
                x0 = float(qcf.pose.x)
                y0 = float(qcf.pose.y)
                # If pose drifts outside lab limits, abort immediately
                xmin, xmax = lab_xlim
                ymin, ymax = lab_ylim
                if not (xmin <= x0 <= xmax and ymin <= y0 <= ymax):
                    print(f"Pose drifted outside lab limits during stabilization (x={x0:.3f}, y={y0:.3f}). Landing for safety.")
                    while qcf.pose is not None and qcf.pose.z > 0.1:
                        qcf.land_in_place()
                        sleep(0.01)
                    print("Landed. Exiting.")
                    sys.exit(1)

                # Use current position directly (already verified inside lab limits)
                hover_target = Pose(x0, y0, world.expanse)
            except Exception:
                pass
        qcf.safe_position_setpoint(hover_target)
        sleep(0.02)

    # Start control timer from now
    hover_start_time = time()
    print(f"Hover achieved at x={hover_target.x:.3f}, y={hover_target.y:.3f}. Controls enabled (device={INPUT_DEVICE}). Max flight time: {MAX_FLIGHT_TIME}s")

    # MAIN CONTROL LOOP
    last_progress_print = 0
    while fly and qcf.is_safe():

        # Terminate upon Esc command or joystick exit button
        if last_key_pressed == pynput.keyboard.Key.esc:
            print("ESC pressed, landing...")
            break
        
        # Check joystick exit button
        if controller.is_exit_pressed():
            print("Joystick exit button pressed, landing...")
            break

        # Elapsed control time
        elapsed = time() - hover_start_time
        if elapsed > MAX_FLIGHT_TIME:
            print(f"Max flight time reached ({MAX_FLIGHT_TIME}s), landing...")
            break

        # Read inputs
        dir_x = dir_y = dir_z = 0.0
        try:
            dir_x, dir_y = controller.get_direction()
            dx = dir_x * MOVEMENT_STEP
            dy = dir_y * MOVEMENT_STEP
            
            dir_z = controller.get_altitude_direction()
            dz = dir_z * ALTITUDE_STEP
        except Exception:
            pass  # Ignore input

        # Update hover target position based on inputs
        if np.sqrt(dir_x**2 + dir_y**2 + dir_z**2) > 1e-4:
            new_x = hover_target.x + dx
            new_y = hover_target.y + dy
            new_z = max(MIN_ALTITUDE, min(MAX_ALTITUDE, hover_target.z + dz))  # Clamp altitude
            
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

            hover_target = Pose(new_x, new_y, new_z)

        # Send setpoint
        qcf.safe_position_setpoint(hover_target)

        # Record data (use elapsed as timestamp relative to hover start)
        try:
            cur_pose = qcf.pose
        except Exception:
            cur_pose = None

        if cur_pose is not None:
            recorder.record_state(elapsed, cur_pose, np.array([hover_target.x, hover_target.y, hover_target.z]))

        # Update realtime plot (reduced frequency)
        if plot is not None and cur_pose is not None and int(elapsed * 10) % 5 == 0:  # Update at 20Hz
            info = f't={elapsed:.1f}s z={cur_pose.z:.2f}m'
            try:
                plot.update([cur_pose.x, cur_pose.y, cur_pose.z], info=info)
            except Exception:
                pass

        # Print status every 2 seconds
        if int(elapsed * 0.5) != last_progress_print and elapsed > 0 and int(elapsed * 10) % 20 == 0:  # Every 2 seconds
            last_progress_print = int(elapsed * 0.5)
            if cur_pose is not None:
                print(f'[t={elapsed:.1f}s] Pos: ({cur_pose.x:.2f}, {cur_pose.y:.2f}, {cur_pose.z:.2f}) m')


    # Clean up
    try:
        controller.stop()
    except Exception:
        pass

    if plot is not None:
        try:
            plot.close()
        except Exception:
            pass

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