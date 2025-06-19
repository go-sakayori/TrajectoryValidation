import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import os
import numpy as np
from scipy.interpolate import interp1d
from autoware_planning_msgs.msg import TrajectoryPoint


class RosbagDataLoader:
    def __init__(self, bag_path, topics):
        """
        Load messages from specified topics in the rosbag and store them internally.

        Args:
            bag_path (str): Path to the rosbag file.
            topics (list): List of topic names to read.
        """
        self.bag_path = bag_path
        self.topics = topics
        self.data = {topic: [] for topic in topics}
        self._load_data()
        self._setup_ground_truth_interpolators()

    def _get_storage_id(self, bag_path):
        """Determine storage ID based on file extension."""
        _, ext = os.path.splitext(bag_path)
        if ext == '.db3':
            return 'sqlite3'
        elif ext == '.mcap':
            return 'mcap'
        else:
            # Default to sqlite3 for compatibility
            return 'sqlite3'

    def _load_data(self):
        reader = rosbag2_py.SequentialReader()
        storage_id = self._get_storage_id(self.bag_path)
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)
        
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            if topic in self.topics:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                self.data[topic].append((msg, timestamp))

    def get_messages(self, topic):
        """
        Get all messages for a specific topic.

        Args:
            topic (str): Topic name.

        Returns:
            list: List of (msg, t) tuples for the topic.
        """
        return self.data.get(topic, [])
    
    def _setup_ground_truth_interpolators(self):
        """Load kinematic state data for ground truth generation."""
        kinematic_messages = self.data.get('/localization/kinematic_state', [])
        if not kinematic_messages:
            self.has_ground_truth = False
            return
        
        timestamps = []
        positions = []
        orientations = []
        velocities = []
        
        for msg, timestamp in kinematic_messages:
            # Convert ROS timestamp to seconds
            t = timestamp / 1e9  # nanoseconds to seconds
            timestamps.append(t)
            
            # Extract position
            pose = msg.pose.pose
            positions.append([pose.position.x, pose.position.y, pose.position.z])
            
            # Extract orientation (quaternion)
            orientations.append([
                pose.orientation.x, pose.orientation.y, 
                pose.orientation.z, pose.orientation.w
            ])
            
            # Extract velocity
            twist = msg.twist.twist
            velocities.append([
                twist.linear.x, twist.linear.y, twist.linear.z,
                twist.angular.x, twist.angular.y, twist.angular.z
            ])
        
        self.timestamps = np.array(timestamps)
        self.positions = np.array(positions)
        self.orientations = np.array(orientations)
        self.velocities = np.array(velocities)
        self.time_range = (self.timestamps.min(), self.timestamps.max())
        self.has_ground_truth = True

    def _get_nearest_state(self, query_time):
        """Return position, orientation, velocity at the nearest timestamp."""
        idx = int(np.abs(self.timestamps - query_time).argmin())
        return (self.positions[idx], self.orientations[idx], self.velocities[idx])
        
    def generate_ground_truth(self, trajectory_msg, trajectory_timestamp):
        """
        Generate ground truth trajectory points corresponding to a planned trajectory.
        
        Args:
            trajectory_msg: The planned trajectory message
            trajectory_timestamp: Timestamp when the trajectory was published (nanoseconds)
        
        Returns:
            List of ground truth trajectory points
        """
        if not self.has_ground_truth or not trajectory_msg.trajectories:
            return []
        
        planned_traj = trajectory_msg.trajectories[0]  # Use first trajectory
        ground_truth_points = []
        
        # Convert trajectory timestamp to seconds
        base_time = trajectory_timestamp / 1e9
        
        # Try to find the best execution start time by matching the first planned point
        # to the closest actual vehicle position
        execution_start_time = self._find_execution_start_time(planned_traj, base_time)
        
        # Optional debug output (can be enabled for troubleshooting)
        debug_enabled = False
        if debug_enabled:
            print(f"[DEBUG] Trajectory timestamp: {base_time:.3f}s")
            print(f"[DEBUG] Estimated execution start: {execution_start_time:.3f}s")
            print(f"[DEBUG] Kinematic data range: {self.time_range[0]:.3f}s to {self.time_range[1]:.3f}s")
            print(f"[DEBUG] Planned trajectory has {len(planned_traj.points)} points")
        
        point_times = []
        
        # Check if trajectory points have valid time_from_start values
        has_valid_timestamps = any(
            (point.time_from_start.sec + point.time_from_start.nanosec / 1e9) > 0.001 
            for point in planned_traj.points
        )
        
        if not has_valid_timestamps:
            if debug_enabled:
                print(f"[WARNING] Trajectory points have invalid timestamps, using estimated timing")
            # Estimate timing based on trajectory length and assumed vehicle speed
            # Typical trajectory planning uses 0.1s intervals
            estimated_interval = 0.1  # seconds
        
        for i, point in enumerate(planned_traj.points):
            if has_valid_timestamps:
                # Use original timestamps from trajectory
                point_time = execution_start_time + point.time_from_start.sec + point.time_from_start.nanosec / 1e9
            else:
                # Use estimated timing: assume trajectory execution starts from execution_start_time
                point_time = execution_start_time + i * estimated_interval
            
            point_times.append(point_time)
            
            # Debug first few points
            if debug_enabled and i < 3:
                time_offset = point.time_from_start.sec + point.time_from_start.nanosec / 1e9
                print(f"[DEBUG] Point {i}: time_from_start={time_offset:.3f}s, absolute_time={point_time:.3f}s, estimated={not has_valid_timestamps}")
            
            # Skip if outside our kinematic data range
            if point_time < self.time_range[0] or point_time > self.time_range[1]:
                if debug_enabled and i < 3:  # Debug first few points
                    print(f"[DEBUG] Point {i} skipped: time {point_time:.3f}s outside range [{self.time_range[0]:.3f}, {self.time_range[1]:.3f}]")
                continue
            
            # Get ground truth state from nearest localization sample
            gt_position, gt_orientation, gt_velocity = self._get_nearest_state(point_time)
            
            # Normalize quaternion
            quat_norm = np.linalg.norm(gt_orientation)
            if quat_norm > 0:
                gt_orientation = gt_orientation / quat_norm
            
            # Create ground truth trajectory point
            gt_point = TrajectoryPoint()
            gt_point.pose.position.x = float(gt_position[0])
            gt_point.pose.position.y = float(gt_position[1])
            gt_point.pose.position.z = float(gt_position[2])
            
            gt_point.pose.orientation.x = float(gt_orientation[0])
            gt_point.pose.orientation.y = float(gt_orientation[1])
            gt_point.pose.orientation.z = float(gt_orientation[2])
            gt_point.pose.orientation.w = float(gt_orientation[3])
            
            gt_point.longitudinal_velocity_mps = float(gt_velocity[0])
            gt_point.lateral_velocity_mps = float(gt_velocity[1])
            
            gt_point.time_from_start = point.time_from_start
            
            ground_truth_points.append(gt_point)
        
        # Debug output for ground truth generation
        if debug_enabled and ground_truth_points:
            print(f"[DEBUG] Generated {len(ground_truth_points)} ground truth points")
            first_gt = ground_truth_points[0]
            last_gt = ground_truth_points[-1]
            print(f"[DEBUG] First GT point: ({first_gt.pose.position.x:.2f}, {first_gt.pose.position.y:.2f})")
            print(f"[DEBUG] Last GT point: ({last_gt.pose.position.x:.2f}, {last_gt.pose.position.y:.2f})")
            
            # Check if all points are the same (indicating a problem)
            positions = [(p.pose.position.x, p.pose.position.y) for p in ground_truth_points]
            unique_positions = set(positions)
            if len(unique_positions) == 1:
                print(f"[WARNING] All ground truth points have the same position! This indicates a timing issue.")
                print(f"[DEBUG] Point times range: {min(point_times):.3f}s to {max(point_times):.3f}s")
        
        return ground_truth_points
    
    def _find_execution_start_time(self, planned_traj, base_time):
        """
        Find the best estimate for when trajectory execution actually started
        by matching planned positions to actual vehicle positions.
        """
        if not planned_traj.points:
            return base_time
        
        # Get the first planned position
        first_planned_pos = planned_traj.points[0].pose.position
        planned_x, planned_y = first_planned_pos.x, first_planned_pos.y
        
        # Search for the time when the vehicle was closest to this position
        # Look in a window around the trajectory timestamp
        search_start = max(self.time_range[0], base_time - 5.0)  # 5 seconds before
        search_end = min(self.time_range[1], base_time + 10.0)   # 10 seconds after
        
        # # Get recorded positions within the search window
        mask = (self.timestamps >= search_start) & (self.timestamps <= search_end)
        if not np.any(mask):
            print("[WARNING] No localization data in search window, using original timestamp")
            return base_time
        search_times = self.timestamps[mask]
        search_positions = self.positions[mask]
        
        # Find the time with minimum distance to planned start position
        distances = np.sqrt((search_positions[:, 0] - planned_x)**2 + 
                           (search_positions[:, 1] - planned_y)**2)
        min_dist_idx = np.argmin(distances)
        best_start_time = search_times[min_dist_idx]
        min_distance = distances[min_dist_idx]
        
        # Always log execution start time since it's important for validation
        print(f"Found execution start time: {best_start_time:.3f}s (distance to planned start: {min_distance:.2f}m)")
        
        # If the minimum distance is very large, fall back to the original timestamp
        if min_distance > 50.0:  # 50 meters tolerance
            print(f"[WARNING] Large position mismatch ({min_distance:.2f}m), using original timestamp")
            return base_time
        
        return best_start_time

    def print_point_intervals(self, trajectory_msg):
        """
        Print the time interval (in seconds) between consecutive points in a trajectory message.
        Assumes each point has a 'time_from_start' attribute (e.g., in trajectory_msgs/JointTrajectoryPoint).
        """
        try:
            if not hasattr(trajectory_msg, 'points') or len(trajectory_msg.points) == 0:
                print("[DEBUG] trajectory_msg.points is empty or not found.")
                return
            times = [p.time_from_start.to_sec() for p in trajectory_msg.points]
            print(f"[DEBUG] times: {times}")
            intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
            print(f"[DEBUG] intervals: {intervals}")
            for i, dt in enumerate(intervals):
                print(f"Interval between point {i} and {i+1}: {dt:.6f} seconds")
        except Exception as e:
            print(f"[ERROR] Exception in print_point_intervals: {e}")