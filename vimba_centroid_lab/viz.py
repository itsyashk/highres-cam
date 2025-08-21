"""viz.py – visualization helpers for centroid overlays and zoom grid."""
from __future__ import annotations

import cv2
import numpy as np


def overlay_centroids(
    img: np.ndarray,
    baseline: tuple[float, float] | None,
    refined: tuple[float, float] | None,
    *,
    color_baseline=(0, 0, 255),
    color_refined=(0, 255, 0),
) -> np.ndarray:
    """Return BGR image with crosses at given centroids."""
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()

    if baseline and not np.isnan(baseline[0]):
        cv2.drawMarker(
            out,
            (int(round(baseline[0])), int(round(baseline[1]))),
            color_baseline,
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=1,
        )
    if refined and not np.isnan(refined[0]):
        cv2.drawMarker(
            out,
            (int(round(refined[0])), int(round(refined[1]))),
            color_refined,
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=1,
        )
    return out


def render_zoom_roi(
    roi: np.ndarray,
    *,
    scale: int = 16,
    bicubic: bool = False,
    grid_color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Return magnified ROI with pixel grid overlay.

    Parameters
    ----------
    roi : np.ndarray
        Raw Mono8 ROI (H×W).
    scale : int
        Magnification factor.
    bicubic : bool
        Use bicubic interpolation if True, nearest-neighbor otherwise.
    grid_color : tuple[int, int, int]
        BGR color for grid lines.
    """
    interp = cv2.INTER_CUBIC if bicubic else cv2.INTER_NEAREST
    out = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=interp)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    h, w = roi.shape
    for y in range(1, h):
        y_ = y * scale
        cv2.line(out_bgr, (0, y_), (w * scale, y_), grid_color, 1)
    for x in range(1, w):
        x_ = x * scale
        cv2.line(out_bgr, (x_, 0), (x_, h * scale), grid_color, 1)
    return out_bgr 