import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import os

class TrajectoryAnimator:
    """
    Visualize robot trajectory animation with obstacles
    """
    
    def __init__(self, csv_file, output_file='trajectory_animation.mp4', 
                 fps=30, speed_multiplier=1.0, use_3d=False, danger_zones=None):
        """
        Initialize the trajectory animator
        
        Args:
            csv_file: Path to CSV file with trajectory data
            output_file: Output animation file path (.mp4 or .gif)
            fps: Frames per second for the animation
            speed_multiplier: Speed up (>1) or slow down (<1) the animation
            use_3d: Whether to use 3D visualization
            danger_zones: Array of danger zone rectangles [x_min, x_max, y_min, y_max] (optional)
        """
        self.csv_file = csv_file
        self.output_file = output_file
        self.fps = fps
        self.speed_multiplier = speed_multiplier
        self.use_3d = use_3d
        
        # Define spatial limits
        self.box_limit = np.array([-2.2, 2.2, -1.8, 1.6])
        
        # Define theoretical start and goal positions
        self.start_position = np.array([-1.5, -0.7, 0.8])
        self.goal_position = np.array([1.0, 1.0, 0.8])
        
        # Define rectangular obstacles [x_min, x_max, y_min, y_max]
        self.obstacles = np.array([
            [-1.93, -0.89, -0.13, 0.13],
            [-0.13, 0.13, 0.61, 1.13],
            [0.87, 1.65, 0, 0.26],
            [0.50, 0.76, -1.02, -0.50],
        ])
        
        # Define danger zones (if provided)
        self.danger_zones = danger_zones
        
        # Load trajectory data
        self.load_data()
        
    def load_data(self):
        """Load trajectory data from CSV file"""
        data = np.loadtxt(self.csv_file, delimiter=',')
        
        # Extract time and position data
        self.time = data[0, :]  # First row: time (s)
        self.x = data[1, :]     # Second row: x position (m)
        self.y = data[2, :]     # Third row: y position (m)
        self.z = data[3, :]     # Fourth row: z position (m)
        
        # Print trajectory information
        print(f"Loaded trajectory with {len(self.time)} data points")
        print(f"Time range: {self.time[0]:.3f}s to {self.time[-1]:.3f}s")
        print(f"Duration: {self.time[-1] - self.time[0]:.3f}s")
        print(f"\nTheoretical start position: {self.start_position}")
        print(f"Actual start position: [{self.x[0]:.3f}, {self.y[0]:.3f}, {self.z[0]:.3f}]")
        print(f"\nTheoretical goal position: {self.goal_position}")
        print(f"Actual end position: [{self.x[-1]:.3f}, {self.y[-1]:.3f}, {self.z[-1]:.3f}]")
        
    def setup_plot(self):
        """Setup the matplotlib figure and axes"""
        if self.use_3d:
            from mpl_toolkits.mplot3d import Axes3D
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.setup_3d_plot()
        else:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.setup_2d_plot()
            
    def setup_2d_plot(self):
        """Setup 2D plot with obstacles"""
        # Set axis limits
        self.ax.set_xlim(self.box_limit[0], self.box_limit[1])
        self.ax.set_ylim(self.box_limit[2], self.box_limit[3])
        self.ax.set_aspect('equal')
        
        # Labels and title
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_title('Robot Trajectory Animation', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles (gray color)
        for i, obs in enumerate(self.obstacles):
            width = obs[1] - obs[0]
            height = obs[3] - obs[2]
            rect = patches.Rectangle(
                (obs[0], obs[2]), width, height,
                linewidth=2, edgecolor='gray', facecolor='gray', alpha=0.5,
                label='Obstacles' if i == 0 else None
            )
            self.ax.add_patch(rect)
        
        # Draw danger zones (red color) if provided
        if self.danger_zones is not None:
            for i, dz in enumerate(self.danger_zones):
                width = dz[1] - dz[0]
                height = dz[3] - dz[2]
                rect = patches.Rectangle(
                    (dz[0], dz[2]), width, height,
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.3,
                    label='Danger Zone' if i == 0 else None
                )
                self.ax.add_patch(rect)
        
        # Plot theoretical start and goal positions
        self.ax.plot(self.start_position[0], self.start_position[1], 
                    'go', markersize=12, label='Theoretical Start', zorder=5)
        self.ax.plot(self.goal_position[0], self.goal_position[1], 
                    'r*', markersize=15, label='Theoretical Goal', zorder=5)
        
        # Plot full trajectory as reference (light gray)
        self.ax.plot(self.x, self.y, 'lightgray', linewidth=1, 
                    alpha=0.5, label='Full Trajectory', zorder=1)
        
        # Initialize animated elements
        self.trajectory_line, = self.ax.plot([], [], 'b-', linewidth=2, 
                                             label='Current Path', zorder=3)
        self.robot_marker, = self.ax.plot([], [], 'bo', markersize=10, 
                                          label='Robot', zorder=4)
        
        # Time text
        self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      fontsize=12, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax.legend(loc='upper right', fontsize=10)
        
    def setup_3d_plot(self):
        """Setup 3D plot with obstacles"""
        # Set axis limits
        self.ax.set_xlim(self.box_limit[0], self.box_limit[1])
        self.ax.set_ylim(self.box_limit[2], self.box_limit[3])
        z_min, z_max = min(self.z.min(), self.start_position[2]) - 0.2, max(self.z.max(), self.goal_position[2]) + 0.2
        self.ax.set_zlim(z_min, z_max)
        
        # Labels and title
        self.ax.set_xlabel('X Position (m)', fontsize=10)
        self.ax.set_ylabel('Y Position (m)', fontsize=10)
        self.ax.set_zlabel('Z Position (m)', fontsize=10)
        self.ax.set_title('Robot Trajectory Animation (3D)', fontsize=14, fontweight='bold')
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Draw obstacles (gray color, as vertical columns)
        for i, obs in enumerate(self.obstacles):
            # Create vertices for a vertical rectangular prism
            x_coords = [obs[0], obs[1], obs[1], obs[0], obs[0]]
            y_coords = [obs[2], obs[2], obs[3], obs[3], obs[2]]
            
            # Draw bottom and top
            self.ax.plot(x_coords, y_coords, [z_min]*5, 'gray', alpha=0.5)
            self.ax.plot(x_coords, y_coords, [z_max]*5, 'gray', alpha=0.5)
            
            # Draw vertical edges
            for j in range(4):
                self.ax.plot([x_coords[j], x_coords[j]], 
                           [y_coords[j], y_coords[j]], 
                           [z_min, z_max], 'gray', alpha=0.5)
            
            # Fill sides
            verts = []
            # Front face
            verts.append([[obs[0], obs[2], z_min], [obs[1], obs[2], z_min], 
                         [obs[1], obs[2], z_max], [obs[0], obs[2], z_max]])
            # Back face
            verts.append([[obs[0], obs[3], z_min], [obs[1], obs[3], z_min], 
                         [obs[1], obs[3], z_max], [obs[0], obs[3], z_max]])
            # Left face
            verts.append([[obs[0], obs[2], z_min], [obs[0], obs[3], z_min], 
                         [obs[0], obs[3], z_max], [obs[0], obs[2], z_max]])
            # Right face
            verts.append([[obs[1], obs[2], z_min], [obs[1], obs[3], z_min], 
                         [obs[1], obs[3], z_max], [obs[1], obs[2], z_max]])
            
            poly = Poly3DCollection(verts, alpha=0.3, facecolor='gray', edgecolor='gray',
                                   label='Obstacles' if i == 0 else None)
            self.ax.add_collection3d(poly)
        
        # Draw danger zones (red color, as vertical columns) if provided
        if self.danger_zones is not None:
            for i, dz in enumerate(self.danger_zones):
                x_coords = [dz[0], dz[1], dz[1], dz[0], dz[0]]
                y_coords = [dz[2], dz[2], dz[3], dz[3], dz[2]]
                
                # Draw bottom and top
                self.ax.plot(x_coords, y_coords, [z_min]*5, 'r-', alpha=0.5)
                self.ax.plot(x_coords, y_coords, [z_max]*5, 'r-', alpha=0.5)
                
                # Draw vertical edges
                for j in range(4):
                    self.ax.plot([x_coords[j], x_coords[j]], 
                               [y_coords[j], y_coords[j]], 
                               [z_min, z_max], 'r-', alpha=0.5)
                
                # Fill sides
                verts = []
                verts.append([[dz[0], dz[2], z_min], [dz[1], dz[2], z_min], 
                             [dz[1], dz[2], z_max], [dz[0], dz[2], z_max]])
                verts.append([[dz[0], dz[3], z_min], [dz[1], dz[3], z_min], 
                             [dz[1], dz[3], z_max], [dz[0], dz[3], z_max]])
                verts.append([[dz[0], dz[2], z_min], [dz[0], dz[3], z_min], 
                             [dz[0], dz[3], z_max], [dz[0], dz[2], z_max]])
                verts.append([[dz[1], dz[2], z_min], [dz[1], dz[3], z_min], 
                             [dz[1], dz[3], z_max], [dz[1], dz[2], z_max]])
                
                poly = Poly3DCollection(verts, alpha=0.2, facecolor='red', edgecolor='red',
                                       label='Danger Zone' if i == 0 else None)
                self.ax.add_collection3d(poly)
        
        # Plot start and goal positions
        self.ax.plot([self.start_position[0]], [self.start_position[1]], [self.start_position[2]], 
                    'go', markersize=10, label='Theoretical Start')
        self.ax.plot([self.goal_position[0]], [self.goal_position[1]], [self.goal_position[2]], 
                    'r*', markersize=12, label='Theoretical Goal')
        
        # Plot full trajectory
        self.ax.plot(self.x, self.y, self.z, 'lightgray', linewidth=1, 
                    alpha=0.5, label='Full Trajectory')
        
        # Initialize animated elements
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Current Path')
        self.robot_marker, = self.ax.plot([], [], [], 'bo', markersize=8, label='Robot')
        
        # Time text
        self.time_text = self.ax.text2D(0.02, 0.98, '', transform=self.ax.transAxes,
                                        fontsize=12, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax.legend(loc='upper right', fontsize=9)
        
    def init_animation(self):
        """Initialize animation"""
        if self.use_3d:
            self.trajectory_line.set_data([], [])
            self.trajectory_line.set_3d_properties([])
            self.robot_marker.set_data([], [])
            self.robot_marker.set_3d_properties([])
        else:
            self.trajectory_line.set_data([], [])
            self.robot_marker.set_data([], [])
        
        self.time_text.set_text('')
        return self.trajectory_line, self.robot_marker, self.time_text
    
    def animate(self, frame):
        """Animation update function"""
        # Calculate which data point to show based on frame and speed multiplier
        total_frames = int(self.fps * (self.time[-1] - self.time[0]) / self.speed_multiplier)
        progress = frame / total_frames
        
        # Find corresponding data index
        current_time = self.time[0] + progress * (self.time[-1] - self.time[0])
        idx = np.searchsorted(self.time, current_time)
        idx = min(idx, len(self.time) - 1)
        
        # Update trajectory line
        if self.use_3d:
            self.trajectory_line.set_data(self.x[:idx+1], self.y[:idx+1])
            self.trajectory_line.set_3d_properties(self.z[:idx+1])
            self.robot_marker.set_data([self.x[idx]], [self.y[idx]])
            self.robot_marker.set_3d_properties([self.z[idx]])
        else:
            self.trajectory_line.set_data(self.x[:idx+1], self.y[:idx+1])
            self.robot_marker.set_data([self.x[idx]], [self.y[idx]])
        
        # Update time text
        self.time_text.set_text(f'Time: {current_time:.2f}s')
        
        return self.trajectory_line, self.robot_marker, self.time_text
    
    def create_animation(self):
        """Create and save the animation"""
        self.setup_plot()
        
        # Calculate total number of frames
        duration = self.time[-1] - self.time[0]
        total_frames = int(self.fps * duration / self.speed_multiplier)
        
        print(f"\nCreating animation with {total_frames} frames at {self.fps} fps")
        print(f"Speed multiplier: {self.speed_multiplier}x")
        
        # Create animation
        anim = FuncAnimation(
            self.fig, 
            self.animate, 
            init_func=self.init_animation,
            frames=total_frames, 
            interval=1000/self.fps,  # Interval in milliseconds
            blit=True,
            repeat=True
        )
        
        # Save animation
        print(f"Saving animation to {self.output_file}...")
        
        if self.output_file.endswith('.gif'):
            writer = PillowWriter(fps=self.fps)
            anim.save(self.output_file, writer=writer)
        else:
            # Try to use FFMpegWriter for mp4
            try:
                writer = FFMpegWriter(fps=self.fps, bitrate=1800)
                anim.save(self.output_file, writer=writer)
            except Exception as e:
                print(f"Warning: FFMpeg not available ({e}). Trying PillowWriter for GIF instead...")
                output_gif = self.output_file.replace('.mp4', '.gif')
                writer = PillowWriter(fps=self.fps)
                anim.save(output_gif, writer=writer)
                print(f"Saved as GIF: {output_gif}")
        
        print("Animation saved successfully!")
        plt.close()


def main():
    """
    Main function - Configure your settings here for VSCode direct execution
    """
    # ===== CONFIGURATION SECTION - EDIT THESE PARAMETERS =====
    
    # Path to your CSV file
    csv_file = 'traj/out/cf_02_20260209154102(WithIntervention).csv'  # Change this to your CSV file path
    
    # Output settings
    output_file = 'traj/out/cf_02_20260209154102(WithIntervention).mp4'  # Output file (.mp4 or .gif)
    fps = 30                                   # Frames per second
    speed_multiplier = 1.0                     # Animation speed (1.0 = normal, 2.0 = 2x faster)
    use_3d = False                             # Set to True for 3D visualization
    
    # Danger zones [x_min, x_max, y_min, y_max] (set to None to disable)
    # You can add multiple danger zones just like obstacles
    danger_zones = np.array([
        [-0.5, 0.5, -0.5, 0.5],      # First danger zone
        # [1.2, 1.8, 0.8, 1.3],      # Uncomment to add more danger zones
    ])
    # Or set to None if no danger zones needed:
    # danger_zones = None
    
    # ===== END OF CONFIGURATION SECTION =====
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        print("Please make sure the file exists in the current directory or provide the full path.")
        return
    
    # Create animator and generate animation
    print("Starting animation generation...")
    print(f"Input file: {csv_file}")
    print(f"Output file: {output_file}")
    if danger_zones is not None:
        print(f"Number of danger zones: {len(danger_zones)}")
        for i, dz in enumerate(danger_zones):
            print(f"  Zone {i+1}: X[{dz[0]}, {dz[1]}], Y[{dz[2]}, {dz[3]}]")
    print("-" * 50)
    
    animator = TrajectoryAnimator(
        csv_file=csv_file,
        output_file=output_file,
        fps=fps,
        speed_multiplier=speed_multiplier,
        use_3d=use_3d,
        danger_zones=danger_zones
    )
    animator.create_animation()
    
    print("-" * 50)
    print(f"✓ Animation saved to: {output_file}")


if __name__ == '__main__':
    main()