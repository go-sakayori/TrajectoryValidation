import argparse
import yaml
from rosbag.reader import RosbagDataLoader
from visualization import TrajectoryVisualizer

def load_topics_from_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        topics = data
    else:
        topics = data.get('topics', [])
    return topics

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from rosbag and save results to another rosbag.")
    parser.add_argument('--input', '-i', required=True, help='Path to input rosbag file')
    parser.add_argument('--output', '-o', required=True, help='Path to output rosbag file')
    parser.add_argument('--topics-yaml', '-t', required=False, default="config/topic_list.yaml", help='Path to YAML file containing topic names')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualization plots')
    parser.add_argument('--animate', '-a', action='store_true', help='Generate animated visualization plots')
    parser.add_argument('--plot-dir', '-p', default='./plots', help='Directory to save plots')
    args = parser.parse_args()

    input_bag = args.input
    output_bag = args.output
    topics = load_topics_from_yaml(args.topics_yaml)

    # Read rosbag and store messages using RosbagDataLoader
    loader = RosbagDataLoader(input_bag, topics)
    
    # Generate ground truth trajectories
    trajectory_messages = loader.get_messages('/diffusion_planner/trajectories')
    print(f"\nGenerating ground truth for {len(trajectory_messages)} planned trajectories...")
    
    # Store trajectory data for visualization
    planned_trajectories = []
    ground_truth_trajectories = []
    
    for i, (traj_msg, traj_timestamp) in enumerate(trajectory_messages):
        ground_truth_points = loader.generate_ground_truth(traj_msg, traj_timestamp)
        if ground_truth_points:
            print(f"Trajectory {i+1}: Generated {len(ground_truth_points)} ground truth points")
            
            # Store for visualization
            if traj_msg.trajectories:
                planned_points = traj_msg.trajectories[0].points
                planned_trajectories.append((planned_points, i+1))
                ground_truth_trajectories.append((ground_truth_points, i+1))
        else:
            print(f"Trajectory {i+1}: No ground truth points generated (outside time range)")
    
    # Generate visualizations if requested
    if args.visualize and planned_trajectories:
        print(f"\nGenerating visualizations...")
        visualizer = TrajectoryVisualizer(args.plot_dir)
        
        # Create overview plot
        visualizer.plot_trajectory_overview(planned_trajectories, ground_truth_trajectories)
        
        # Create detailed plots for first few trajectories
        num_detailed_plots = min(5, len(planned_trajectories))
        print(f"Creating detailed plots for first {num_detailed_plots} trajectories...")
        
        for i in range(num_detailed_plots):
            planned_points, traj_id = planned_trajectories[i]
            gt_points, _ = ground_truth_trajectories[i]
            visualizer.plot_trajectory_comparison(planned_points, gt_points, traj_id)
        
        print(f"Visualization complete! Plots saved to: {args.plot_dir}")
    
    # Generate animations if requested
    if args.animate and planned_trajectories:
        print(f"\nGenerating animated visualizations...")
        visualizer = TrajectoryVisualizer(args.plot_dir)
        
        # Create animated overview
        print("Creating animated overview...")
        visualizer.create_animated_trajectory_overview(planned_trajectories, ground_truth_trajectories)
        
        # Create animated detailed plots for first few trajectories
        num_animated_plots = min(3, len(planned_trajectories))  # Limit to 3 for performance
        print(f"Creating animated plots for first {num_animated_plots} trajectories...")
        
        for i in range(num_animated_plots):
            planned_points, traj_id = planned_trajectories[i]
            gt_points, _ = ground_truth_trajectories[i]
            visualizer.create_animated_trajectory_comparison(planned_points, gt_points, traj_id)
        
        print(f"Animation complete! Animated plots saved to: {args.plot_dir}")

if __name__ == "__main__":
    main()