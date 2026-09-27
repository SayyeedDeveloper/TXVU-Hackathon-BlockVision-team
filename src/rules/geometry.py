"""src/rules/geometry.py — small pixel-space geometry helpers shared by rule modules."""
from __future__ import annotations

import math


def point_in_polygon(pt: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test. `polygon` need not be explicitly closed."""
    if len(polygon) < 3:
        return False
    x, y = pt
    inside = False
    n = len(polygon)
    x1, y1 = polygon[-1]
    for i in range(n):
        x2, y2 = polygon[i]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def side_of_line(pt: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Signed cross product; sign tells which side of the p1->p2 line `pt` is on."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def point_segment_distance(pt: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
    px, py = pt
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len2))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy)


def segments_intersect(p1, p2, p3, p4) -> bool:
    """Standard orientation-based segment intersection test."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)

    def on_segment(a, b, c):
        return min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and \
               min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9

    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, p3):
        return True
    if o2 == 0 and on_segment(p1, p2, p4):
        return True
    if o3 == 0 and on_segment(p3, p4, p1):
        return True
    if o4 == 0 and on_segment(p3, p4, p2):
        return True
    return False


def unit_vector(v: tuple[float, float]) -> tuple[float, float]:
    n = math.hypot(v[0], v[1])
    return (0.0, 0.0) if n < 1e-9 else (v[0] / n, v[1] / n)


def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Angle in degrees [0, 180] between two vectors; 0 if either is ~zero."""
    u1, u2 = unit_vector(v1), unit_vector(v2)
    if u1 == (0.0, 0.0) or u2 == (0.0, 0.0):
        return 0.0
    dot = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
    return math.degrees(math.acos(dot))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ground_point(cx: float, cy: float, h: float) -> tuple[float, float]:
    """Bounding-box bottom-center: (cx, cy + h/2). A better ground-contact
    proxy than the raw box center for TALL, NARROW boxes -- specifically
    pedestrians -- viewed from an elevated, oblique camera. The box center
    sits at roughly chest height, which projects well above where a
    standing/walking person's feet actually touch the ground/crosswalk
    surface, causing false "off the crossing"/"off the road" results even
    when they're visibly standing on it (confirmed with real coordinates:
    e.g. a box-center of (1485, 1076) with h=158 tested outside a real
    crossing polygon, while its ground_point (1485, 1155) tested inside the
    same polygon).

    Deliberately NOT used for vehicles: a car/bus's box is low-profile
    relative to its footprint at this camera angle, so its box-center is
    already a fine ground proxy -- applying this would shift a vehicle's
    tested point toward its rear bumper for no benefit. Callers must apply
    this only to pedestrian points, not blanket all TrackPoints through it.
    """
    return (cx, cy + h / 2)
