#!/usr/bin/env python3
"""
Quick test script for animation functionality
"""
import sys
sys.path.append('./src')

from rosbag.reader import RosbagDataLoader
from visualization import TrajectoryVisualizer
import yaml

def load_topics_from_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        topics = data
    else:
        topics = data.get('topics', [])
    return topics

def main():
    input_bag = "/home/gosakayori/.webauto/simulation/data/data_20250616200226_2474253659/log/scenario_test_runner/scenario/scenario/scenario_0.mcap"
    topics = load_topics_from_yaml("config/topic_list.yaml")
    
    # Read rosbag and get first trajectory only
    loader = RosbagDataLoader(input_bag, topics)
    trajectory_messages = loader.get_messages('/diffusion_planner/trajectories')
    
    if trajectory_messages:
        # Get just the first trajectory for testing
        traj_msg, traj_timestamp = trajectory_messages[0]
        ground_truth_points = loader.generate_ground_truth(traj_msg, traj_timestamp)
        
        if ground_truth_points and traj_msg.trajectories:
            planned_points = traj_msg.trajectories[0].points
            
            print(f"Testing animation with {len(planned_points)} planned points and {len(ground_truth_points)} ground truth points")
            
            # Create visualizer and test animation
            visualizer = TrajectoryVisualizer("./plots")
            
            print("Creating animated trajectory comparison...")
            visualizer.create_animated_trajectory_comparison(
                planned_points, ground_truth_points, 1, 
                save=True, duration=3.0  # Shorter duration for testing
            )
            
            print("Animation test complete!")
        else:
            print("No valid trajectory data found")
    else:
        print("No trajectory messages found")

if __name__ == "__main__":
    main()