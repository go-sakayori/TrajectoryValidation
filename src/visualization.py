import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import os


class TrajectoryVisualizer:
    def __init__(self, output_dir="./plots"):
        """
        Initialize trajectory visualizer.
        
        Args:
            output_dir: Directory to save plot files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_trajectory_comparison(self, planned_points, ground_truth_points, 
                                 trajectory_id, save=True, show=False):
        """
        Plot comparison between planned and ground truth trajectories.
        
        Args:
            planned_points: List of planned trajectory points
            ground_truth_points: List of ground truth trajectory points  
            trajectory_id: ID/index of the trajectory
            save: Whether to save the plot
            show: Whether to display the plot
        """
        if not planned_points or not ground_truth_points:
            print(f"Skipping trajectory {trajectory_id}: insufficient data")
            return
        
        # Extract positions
        planned_x = [p.pose.position.x for p in planned_points]
        planned_y = [p.pose.position.y for p in planned_points]
        
        gt_x = [p.pose.position.x for p in ground_truth_points]
        gt_y = [p.pose.position.y for p in ground_truth_points]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Trajectory {trajectory_id} - Planned vs Ground Truth', fontsize=16)
        
        # 1. XY Position comparison
        ax1.plot(planned_x, planned_y, 'b-', label='Planned', linewidth=2, alpha=0.8)
        ax1.plot(gt_x, gt_y, 'r--', label='Ground Truth', linewidth=2, alpha=0.8)
        ax1.plot(planned_x[0], planned_y[0], 'go', markersize=8, label='Start')
        ax1.plot(planned_x[-1], planned_y[-1], 'ro', markersize=8, label='End')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Trajectory Path (XY)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. Position error over time
        if len(planned_points) == len(ground_truth_points):
            position_error = [np.sqrt((px - gx)**2 + (py - gy)**2) 
                            for px, py, gx, gy in zip(planned_x, planned_y, gt_x, gt_y)]
            time_points = [i * 0.1 for i in range(len(position_error))]  # Assuming 0.1s intervals
            
            ax2.plot(time_points, position_error, 'g-', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Position Error (m)')
            ax2.set_title('Position Error Over Time')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_error = np.mean(position_error)
            max_error = np.max(position_error)
            ax2.axhline(y=mean_error, color='orange', linestyle='--', 
                       label=f'Mean: {mean_error:.3f}m')
            ax2.axhline(y=max_error, color='red', linestyle='--', 
                       label=f'Max: {max_error:.3f}m')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Different trajectory lengths\nCannot compute error', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Position Error (N/A)')
        
        # 3. Velocity comparison
        planned_vel = [p.longitudinal_velocity_mps for p in planned_points]
        gt_vel = [p.longitudinal_velocity_mps for p in ground_truth_points]
        
        time_planned = [i * 0.1 for i in range(len(planned_vel))]
        time_gt = [i * 0.1 for i in range(len(gt_vel))]
        
        ax3.plot(time_planned, planned_vel, 'b-', label='Planned', linewidth=2)
        ax3.plot(time_gt, gt_vel, 'r--', label='Ground Truth', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Longitudinal Velocity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics summary
        ax4.axis('off')
        stats_text = f"""
Trajectory Statistics:
━━━━━━━━━━━━━━━━━━━━━━

Planned Points: {len(planned_points)}
Ground Truth Points: {len(ground_truth_points)}

Planned Trajectory:
  Start: ({planned_x[0]:.2f}, {planned_y[0]:.2f})
  End: ({planned_x[-1]:.2f}, {planned_y[-1]:.2f})
  Length: {self._calculate_path_length(planned_x, planned_y):.2f}m

Ground Truth Trajectory:
  Start: ({gt_x[0]:.2f}, {gt_y[0]:.2f})
  End: ({gt_x[-1]:.2f}, {gt_y[-1]:.2f})
  Length: {self._calculate_path_length(gt_x, gt_y):.2f}m
"""
        
        if len(planned_points) == len(ground_truth_points):
            position_error = [np.sqrt((px - gx)**2 + (py - gy)**2) 
                            for px, py, gx, gy in zip(planned_x, planned_y, gt_x, gt_y)]
            stats_text += f"""
Position Error:
  Mean: {np.mean(position_error):.3f}m
  Max: {np.max(position_error):.3f}m
  RMS: {np.sqrt(np.mean(np.square(position_error))):.3f}m
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save:
            filename = f"trajectory_{trajectory_id:03d}_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _calculate_path_length(self, x_points, y_points):
        """Calculate total path length from list of x,y coordinates."""
        if len(x_points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(x_points)):
            dx = x_points[i] - x_points[i-1]
            dy = y_points[i] - y_points[i-1]
            total_length += np.sqrt(dx*dx + dy*dy)
        
        return total_length
    
    def plot_trajectory_overview(self, all_planned_trajectories, all_ground_truth_trajectories, 
                               save=True, show=False):
        """
        Plot overview of all trajectories.
        
        Args:
            all_planned_trajectories: List of (planned_points, trajectory_id) tuples
            all_ground_truth_trajectories: List of (gt_points, trajectory_id) tuples
            save: Whether to save the plot
            show: Whether to display the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('All Trajectories Overview', fontsize=16)
        
        # Plot all planned trajectories
        for planned_points, traj_id in all_planned_trajectories[:10]:  # Limit to first 10 for clarity
            if planned_points:
                x_vals = [p.pose.position.x for p in planned_points]
                y_vals = [p.pose.position.y for p in planned_points]
                ax1.plot(x_vals, y_vals, alpha=0.6, linewidth=1)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Planned Trajectories (First 10)')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot all ground truth trajectories
        for gt_points, traj_id in all_ground_truth_trajectories[:10]:  # Limit to first 10 for clarity
            if gt_points:
                x_vals = [p.pose.position.x for p in gt_points]
                y_vals = [p.pose.position.y for p in gt_points]
                ax2.plot(x_vals, y_vals, alpha=0.6, linewidth=1)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Ground Truth Trajectories (First 10)')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, "trajectories_overview.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved overview plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_animated_trajectory_comparison(self, planned_points, ground_truth_points, 
                                            trajectory_id, save=True, duration=5.0):
        """
        Create animated trajectory comparison showing trajectories being drawn over time.
        
        Args:
            planned_points: List of planned trajectory points
            ground_truth_points: List of ground truth trajectory points  
            trajectory_id: ID/index of the trajectory
            save: Whether to save the animation
            duration: Animation duration in seconds
        """
        if not planned_points or not ground_truth_points:
            print(f"Skipping animated trajectory {trajectory_id}: insufficient data")
            return
        
        # Extract positions
        planned_x = [p.pose.position.x for p in planned_points]
        planned_y = [p.pose.position.y for p in planned_points]
        
        gt_x = [p.pose.position.x for p in ground_truth_points]
        gt_y = [p.pose.position.y for p in ground_truth_points]
        
        # Create figure and subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Animated Trajectory {trajectory_id} - Planned vs Ground Truth', fontsize=16)
        
        # Set up plot limits and styling
        all_x = planned_x + gt_x
        all_y = planned_y + gt_y
        x_margin = (max(all_x) - min(all_x)) * 0.1
        y_margin = (max(all_y) - min(all_y)) * 0.1
        
        ax1.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax1.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Trajectory Path (XY) - Animated')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Initialize empty line objects for animation
        planned_line, = ax1.plot([], [], 'b-', label='Planned', linewidth=3, alpha=0.8)
        gt_line, = ax1.plot([], [], 'r--', label='Ground Truth', linewidth=3, alpha=0.8)
        planned_point, = ax1.plot([], [], 'bo', markersize=8, alpha=0.9)
        gt_point, = ax1.plot([], [], 'ro', markersize=8, alpha=0.9)
        
        ax1.legend()
        
        # Set up velocity plot
        planned_vel = [p.longitudinal_velocity_mps for p in planned_points]
        gt_vel = [p.longitudinal_velocity_mps for p in ground_truth_points]
        
        max_vel = max(max(planned_vel) if planned_vel else [0], max(gt_vel) if gt_vel else [0])
        min_vel = min(min(planned_vel) if planned_vel else [0], min(gt_vel) if gt_vel else [0])
        vel_margin = (max_vel - min_vel) * 0.1
        
        ax2.set_xlim(0, max(len(planned_vel), len(gt_vel)) * 0.1)
        ax2.set_ylim(min_vel - vel_margin, max_vel + vel_margin)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Longitudinal Velocity - Animated')
        ax2.grid(True, alpha=0.3)
        
        planned_vel_line, = ax2.plot([], [], 'b-', label='Planned', linewidth=2)
        gt_vel_line, = ax2.plot([], [], 'r--', label='Ground Truth', linewidth=2)
        planned_vel_point, = ax2.plot([], [], 'bo', markersize=6)
        gt_vel_point, = ax2.plot([], [], 'ro', markersize=6)
        
        ax2.legend()
        
        # Set up error plot
        if len(planned_points) == len(ground_truth_points):
            position_errors = [np.sqrt((px - gx)**2 + (py - gy)**2) 
                             for px, py, gx, gy in zip(planned_x, planned_y, gt_x, gt_y)]
            max_error = max(position_errors)
            
            ax3.set_xlim(0, len(position_errors) * 0.1)
            ax3.set_ylim(0, max_error * 1.1)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Position Error (m)')
            ax3.set_title('Position Error Over Time - Animated')
            ax3.grid(True, alpha=0.3)
            
            error_line, = ax3.plot([], [], 'g-', linewidth=2)
            error_point, = ax3.plot([], [], 'go', markersize=6)
            
            # Add mean error line
            mean_error = np.mean(position_errors)
            ax3.axhline(y=mean_error, color='orange', linestyle='--', alpha=0.7,
                       label=f'Mean: {mean_error:.3f}m')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Different trajectory lengths\nCannot compute error', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Position Error (N/A)')
        
        # Statistics panel
        ax4.axis('off')
        stats_text = ax4.text(0.05, 0.95, '', transform=ax4.transAxes, fontsize=10,
                             verticalalignment='top', fontfamily='monospace')
        
        # Animation function
        def animate(frame):
            # Calculate current progress (0 to 1)
            progress = frame / (frames - 1) if frames > 1 else 1.0
            
            # Calculate how many points to show
            planned_idx = int(progress * len(planned_points))
            gt_idx = int(progress * len(ground_truth_points))
            
            # Update trajectory lines
            if planned_idx > 0:
                planned_line.set_data(planned_x[:planned_idx], planned_y[:planned_idx])
                planned_point.set_data([planned_x[planned_idx-1]], [planned_y[planned_idx-1]])
            
            if gt_idx > 0:
                gt_line.set_data(gt_x[:gt_idx], gt_y[:gt_idx])
                gt_point.set_data([gt_x[gt_idx-1]], [gt_y[gt_idx-1]])
            
            # Update velocity plots
            if planned_idx > 0:
                time_planned = [i * 0.1 for i in range(planned_idx)]
                planned_vel_line.set_data(time_planned, planned_vel[:planned_idx])
                planned_vel_point.set_data([time_planned[-1]], [planned_vel[planned_idx-1]])
            
            if gt_idx > 0:
                time_gt = [i * 0.1 for i in range(gt_idx)]
                gt_vel_line.set_data(time_gt, gt_vel[:gt_idx])
                gt_vel_point.set_data([time_gt[-1]], [gt_vel[gt_idx-1]])
            
            # Update error plot
            if len(planned_points) == len(ground_truth_points) and min(planned_idx, gt_idx) > 0:
                error_idx = min(planned_idx, gt_idx)
                time_error = [i * 0.1 for i in range(error_idx)]
                error_line.set_data(time_error, position_errors[:error_idx])
                error_point.set_data([time_error[-1]], [position_errors[error_idx-1]])
            
            # Update statistics
            current_stats = f"""
Animation Progress: {progress*100:.1f}%
Points Shown: {min(planned_idx, gt_idx)}/{min(len(planned_points), len(ground_truth_points))}

Trajectory Statistics:
━━━━━━━━━━━━━━━━━━━━━━

Planned Points: {len(planned_points)}
Ground Truth Points: {len(ground_truth_points)}

Current Position:
  Planned: ({planned_x[planned_idx-1]:.2f}, {planned_y[planned_idx-1]:.2f}) if planned_idx > 0 else N/A
  Ground Truth: ({gt_x[gt_idx-1]:.2f}, {gt_y[gt_idx-1]:.2f}) if gt_idx > 0 else N/A
"""
            if len(planned_points) == len(ground_truth_points) and min(planned_idx, gt_idx) > 0:
                current_error = position_errors[min(planned_idx, gt_idx)-1]
                current_stats += f"\nCurrent Error: {current_error:.3f}m"
            
            stats_text.set_text(current_stats)
            
            return [planned_line, gt_line, planned_point, gt_point, 
                    planned_vel_line, gt_vel_line, planned_vel_point, gt_vel_point] + \
                   ([error_line, error_point] if len(planned_points) == len(ground_truth_points) else []) + \
                   [stats_text]
        
        # Create animation
        frames = 100  # Number of animation frames
        interval = (duration * 1000) / frames  # Interval in milliseconds
        
        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, 
                           blit=False, repeat=True)
        
        plt.tight_layout()
        
        if save:
            filename = f"trajectory_{trajectory_id:03d}_animated.gif"
            filepath = os.path.join(self.output_dir, filename)
            print(f"Creating animated visualization: {filepath}")
            print("This may take a moment...")
            
            # Save as GIF
            writer = PillowWriter(fps=frames/duration)
            anim.save(filepath, writer=writer, dpi=100)
            print(f"Saved animated plot: {filepath}")
        
        plt.show()
        return anim
    
    def create_animated_trajectory_overview(self, all_planned_trajectories, all_ground_truth_trajectories,
                                          save=True, duration=8.0, max_trajectories=10):
        """
        Create animated overview showing multiple trajectories being drawn sequentially.
        
        Args:
            all_planned_trajectories: List of (planned_points, trajectory_id) tuples
            all_ground_truth_trajectories: List of (gt_points, trajectory_id) tuples
            save: Whether to save the animation
            duration: Animation duration in seconds
            max_trajectories: Maximum number of trajectories to animate
        """
        # Limit trajectories for performance
        planned_trajs = all_planned_trajectories[:max_trajectories]
        gt_trajs = all_ground_truth_trajectories[:max_trajectories]
        
        if not planned_trajs or not gt_trajs:
            print("No trajectories to animate")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Animated Trajectories Overview (First {len(planned_trajs)})', fontsize=16)
        
        # Calculate plot limits
        all_x, all_y = [], []
        for planned_points, _ in planned_trajs:
            if planned_points:
                all_x.extend([p.pose.position.x for p in planned_points])
                all_y.extend([p.pose.position.y for p in planned_points])
        
        for gt_points, _ in gt_trajs:
            if gt_points:
                all_x.extend([p.pose.position.x for p in gt_points])
                all_y.extend([p.pose.position.y for p in gt_points])
        
        if all_x and all_y:
            x_margin = (max(all_x) - min(all_x)) * 0.1
            y_margin = (max(all_y) - min(all_y)) * 0.1
            
            for ax in [ax1, ax2]:
                ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
                ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Planned Trajectories')
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Ground Truth Trajectories')
        
        # Initialize line objects for each trajectory
        planned_lines = []
        gt_lines = []
        colors = plt.cm.tab10(np.linspace(0, 1, len(planned_trajs)))
        
        for i, color in enumerate(colors):
            planned_line, = ax1.plot([], [], color=color, alpha=0.7, linewidth=2)
            gt_line, = ax2.plot([], [], color=color, alpha=0.7, linewidth=2)
            planned_lines.append(planned_line)
            gt_lines.append(gt_line)
        
        # Animation function
        def animate(frame):
            frames_per_trajectory = 20
            current_traj = frame // frames_per_trajectory
            frame_in_traj = frame % frames_per_trajectory
            
            if current_traj >= len(planned_trajs):
                current_traj = len(planned_trajs) - 1
                frame_in_traj = frames_per_trajectory - 1
            
            # Show all previous trajectories completely
            for i in range(current_traj):
                if i < len(planned_trajs):
                    planned_points, _ = planned_trajs[i]
                    if planned_points:
                        x_vals = [p.pose.position.x for p in planned_points]
                        y_vals = [p.pose.position.y for p in planned_points]
                        planned_lines[i].set_data(x_vals, y_vals)
                
                if i < len(gt_trajs):
                    gt_points, _ = gt_trajs[i]
                    if gt_points:
                        x_vals = [p.pose.position.x for p in gt_points]
                        y_vals = [p.pose.position.y for p in gt_points]
                        gt_lines[i].set_data(x_vals, y_vals)
            
            # Animate current trajectory
            if current_traj < len(planned_trajs):
                planned_points, _ = planned_trajs[current_traj]
                if planned_points:
                    progress = frame_in_traj / (frames_per_trajectory - 1)
                    num_points = int(progress * len(planned_points))
                    if num_points > 0:
                        x_vals = [p.pose.position.x for p in planned_points[:num_points]]
                        y_vals = [p.pose.position.y for p in planned_points[:num_points]]
                        planned_lines[current_traj].set_data(x_vals, y_vals)
            
            if current_traj < len(gt_trajs):
                gt_points, _ = gt_trajs[current_traj]
                if gt_points:
                    progress = frame_in_traj / (frames_per_trajectory - 1)
                    num_points = int(progress * len(gt_points))
                    if num_points > 0:
                        x_vals = [p.pose.position.x for p in gt_points[:num_points]]
                        y_vals = [p.pose.position.y for p in gt_points[:num_points]]
                        gt_lines[current_traj].set_data(x_vals, y_vals)
            
            # Update title with progress
            fig.suptitle(f'Animated Trajectories Overview - Drawing Trajectory {current_traj + 1}/{len(planned_trajs)}', 
                        fontsize=16)
            
            return planned_lines + gt_lines
        
        # Create animation
        total_frames = len(planned_trajs) * 20
        interval = (duration * 1000) / total_frames
        
        anim = FuncAnimation(fig, animate, frames=total_frames, interval=interval,
                           blit=False, repeat=True)
        
        plt.tight_layout()
        
        if save:
            filename = "trajectories_overview_animated.gif"
            filepath = os.path.join(self.output_dir, filename)
            print(f"Creating animated overview: {filepath}")
            print("This may take a moment...")
            
            writer = PillowWriter(fps=total_frames/duration)
            anim.save(filepath, writer=writer, dpi=100)
            print(f"Saved animated overview: {filepath}")
        
        plt.show()
        return anim