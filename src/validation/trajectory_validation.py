"""Tools for validating Autoware trajectories from rosbag files."""

from __future__ import annotations

import argparse
import dataclasses
from typing import Iterable, List, Optional, Tuple

from .utils import (
    aggregate_to_dict,
    compute_ade,
    compute_curvature,
    compute_fde,
    compute_heading_change,
    compute_lateral_offset,
    compute_safety_score,
    metrics_to_dict,
)

# rosbag2_py is only available inside a ROS environment. Import lazily.
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from autoware_auto_planning_msgs.msg import Trajectory
except Exception:  # pragma: no cover - optional dependency
    SequentialReader = None  # type: ignore
    StorageOptions = None
    ConverterOptions = None
    deserialize_message = None
    Trajectory = None


@dataclasses.dataclass
class TrajectoryMetrics:
    timestamp: float
    ade: Optional[float]
    fde: Optional[float]
    velocity: float
    acceleration: float
    jerk: float
    heading_change: float
    curvature: float
    lateral_offset: float
    safety_score: float


@dataclasses.dataclass
class AggregateStats:
    count: int = 0
    ade_mean: float = 0.0
    ade_max: float = 0.0
    fde_mean: float = 0.0
    fde_max: float = 0.0
    velocity_mean: float = 0.0
    velocity_max: float = 0.0
    acceleration_mean: float = 0.0
    acceleration_max: float = 0.0
    jerk_mean: float = 0.0
    jerk_max: float = 0.0
    heading_change_mean: float = 0.0
    heading_change_max: float = 0.0
    curvature_mean: float = 0.0
    curvature_max: float = 0.0
    lateral_offset_mean: float = 0.0
    lateral_offset_max: float = 0.0
    safety_score_mean: float = 0.0
    safety_score_min: float = 1.0

    def update(self, m: TrajectoryMetrics) -> None:
        self.count += 1
        if m.ade is not None:
            self.ade_mean += m.ade
            self.ade_max = max(self.ade_max, m.ade)
        if m.fde is not None:
            self.fde_mean += m.fde
            self.fde_max = max(self.fde_max, m.fde)
        self.velocity_mean += m.velocity
        self.velocity_max = max(self.velocity_max, m.velocity)
        self.acceleration_mean += m.acceleration
        self.acceleration_max = max(self.acceleration_max, m.acceleration)
        self.jerk_mean += m.jerk
        self.jerk_max = max(self.jerk_max, m.jerk)
        self.heading_change_mean += m.heading_change
        self.heading_change_max = max(self.heading_change_max, m.heading_change)
        self.curvature_mean += m.curvature
        self.curvature_max = max(self.curvature_max, m.curvature)
        self.lateral_offset_mean += m.lateral_offset
        self.lateral_offset_max = max(self.lateral_offset_max, m.lateral_offset)
        self.safety_score_mean += m.safety_score
        self.safety_score_min = min(self.safety_score_min, m.safety_score)

    def finalize(self) -> None:
        if self.count == 0:
            return
        self.ade_mean /= max(1, self.count)
        self.fde_mean /= max(1, self.count)
        self.velocity_mean /= self.count
        self.acceleration_mean /= self.count
        self.jerk_mean /= self.count
        self.heading_change_mean /= self.count
        self.curvature_mean /= self.count
        self.lateral_offset_mean /= self.count
        self.safety_score_mean /= self.count



# ROS bag handling -----------------------------------------------------------


def read_trajectories(path: str, topic: str = "/planning/trajectory") -> Iterable[Trajectory]:
    if SequentialReader is None:
        raise RuntimeError("rosbag2_py is not available")

    storage_options = StorageOptions(uri=path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")

    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    topics = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics}
    if topic not in topic_types:
        available = ", ".join(topic_types.keys())
        raise RuntimeError(f"Topic {topic} not in bag. Available: {available}")

    while reader.has_next():
        topic_name, data, t = reader.read_next()
        if topic_name != topic:
            continue
        msg = deserialize_message(data, Trajectory)
        yield msg


# Metrics computation --------------------------------------------------------


def process_bag(path: str, ground_truth: Optional[List[Tuple[float, float]]] = None, topic: str = "/planning/trajectory") -> Tuple[List[TrajectoryMetrics], AggregateStats]:
    metrics: List[TrajectoryMetrics] = []
    agg = AggregateStats()

    for msg in read_trajectories(path, topic):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.points]
        velocities = [p.longitudinal_velocity_mps for p in msg.points]
        accelerations = [p.acceleration_mps2 for p in msg.points]
        jerks = [p.jerk_mps3 for p in msg.points]

        ade_val = None
        fde_val = None
        if ground_truth and len(ground_truth) >= len(points):
            gt_slice = ground_truth[:len(points)]
            ade_val = compute_ade(points, gt_slice)
            fde_val = compute_fde(points, gt_slice)

        vel = sum(velocities) / len(velocities) if velocities else 0.0
        acc = sum(accelerations) / len(accelerations) if accelerations else 0.0
        jerk = sum(jerks) / len(jerks) if jerks else 0.0

        heading_change = compute_heading_change(points)
        curvature = compute_curvature(points)
        lateral_offset = compute_lateral_offset(points, ground_truth) if ground_truth else 0.0
        safety_score = compute_safety_score(acc, jerk, curvature, lateral_offset)

        m = TrajectoryMetrics(
            timestamp=ts,
            ade=ade_val,
            fde=fde_val,
            velocity=vel,
            acceleration=acc,
            jerk=jerk,
            heading_change=heading_change,
            curvature=curvature,
            lateral_offset=lateral_offset,
            safety_score=safety_score,
        )
        metrics.append(m)
        agg.update(m)

    agg.finalize()
    return metrics, agg

# Command line interface -----------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Validate trajectories from rosbag files")
    parser.add_argument("bag", help="Path to input bag directory")
    parser.add_argument("--topic", default="/planning/trajectory", help="Trajectory topic name")
    parser.add_argument("--ground-truth", help="Optional path to ground truth CSV with x,y per line")
    args = parser.parse_args(argv)

    gt_points: Optional[List[Tuple[float, float]]] = None
    if args.ground_truth:
        gt_points = []
        with open(args.ground_truth, "r", encoding="utf-8") as f:
            for line in f:
                x_str, y_str = line.strip().split(',')
                gt_points.append((float(x_str), float(y_str)))

    metrics, agg = process_bag(args.bag, ground_truth=gt_points, topic=args.topic)

    for m in metrics:
        print(metrics_to_dict(m))
    print("Aggregate:")
    print(aggregate_to_dict(agg))


if __name__ == "__main__":
    main()

