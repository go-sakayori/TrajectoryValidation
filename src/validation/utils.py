"""Helper functions for trajectory validation."""

from __future__ import annotations

import dataclasses
import math
from typing import List, Tuple


def compute_ade(points: List[Tuple[float, float]], gt: List[Tuple[float, float]]) -> float:
    """Average displacement error."""
    if not points or not gt or len(points) != len(gt):
        return 0.0
    dists = [math.hypot(px - gx, py - gy) for (px, py), (gx, gy) in zip(points, gt)]
    return sum(dists) / len(dists)


def compute_fde(points: List[Tuple[float, float]], gt: List[Tuple[float, float]]) -> float:
    """Final displacement error."""
    if not points or not gt:
        return 0.0
    px, py = points[-1]
    gx, gy = gt[-1]
    return math.hypot(px - gx, py - gy)


def compute_heading_change(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    headings = [math.atan2(y2 - y1, x2 - x1) for (x1, y1), (x2, y2) in zip(points[:-1], points[1:])]
    diffs = [abs(h2 - h1) for h1, h2 in zip(headings[:-1], headings[1:])]
    return sum(diffs)


def compute_curvature(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    curvatures = []
    for i in range(1, len(points) - 1):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        x3, y3 = points[i + 1]
        k1 = 0.5 * ((x1**2 + y1**2) - (x2**2 + y2**2))
        k2 = 0.5 * ((x2**2 + y2**2) - (x3**2 + y3**2))
        denom = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
        if abs(denom) < 1e-6:
            continue
        cx = (k1 * (y2 - y3) - k2 * (y1 - y2)) / denom
        cy = (k2 * (x1 - x2) - k1 * (x2 - x3)) / denom
        r = math.sqrt((x2 - cx)**2 + (y2 - cy)**2)
        if r != 0:
            curvatures.append(1.0 / r)
    if not curvatures:
        return 0.0
    return sum(curvatures) / len(curvatures)


def compute_lateral_offset(points: List[Tuple[float, float]], centerline: List[Tuple[float, float]]) -> float:
    if not points or not centerline:
        return 0.0
    offsets = [math.hypot(x - cx, y - cy) for (x, y), (cx, cy) in zip(points, centerline)]
    return sum(offsets) / len(offsets)


def compute_safety_score(acceleration: float, jerk: float, curvature: float, lateral_offset: float) -> float:
    """Compute a simple safety score in [0, 1]."""
    penalty = abs(acceleration) + abs(jerk) + abs(curvature) + abs(lateral_offset)
    return 1.0 / (1.0 + penalty)


def metrics_to_dict(m: dataclasses.dataclass) -> dict:
    return dataclasses.asdict(m)


def aggregate_to_dict(a: dataclasses.dataclass) -> dict:
    return dataclasses.asdict(a)


