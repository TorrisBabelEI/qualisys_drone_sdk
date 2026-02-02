"""Boundary helpers for lab limits."""
from typing import Tuple

LabLimit = Tuple[float, float]


def clamp_xy(x: float, y: float, lab_xlim: LabLimit, lab_ylim: LabLimit):
    """Clamp x,y to provided lab limits (xmin, xmax), (ymin, ymax)."""
    xmin, xmax = lab_xlim
    ymin, ymax = lab_ylim
    x_clamped = min(max(x, xmin), xmax)
    y_clamped = min(max(y, ymin), ymax)
    return x_clamped, y_clamped
