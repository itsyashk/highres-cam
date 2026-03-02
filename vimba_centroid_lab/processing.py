"""processing.py – image processing algorithms for vimba_centroid_lab

Contains:
    detect_blobs         – bright object detection via threshold + CC.
    baseline_centroid    – centroid of binary mask (core/non-black) as baseline.
    subpixel_centroid    – sub-pixel centroid via isophote interpolation + circle fit.

All functions operate on single-channel uint8 numpy arrays (Mono8 images).
"""
from __future__ import annotations

from typing import Tuple, List, Dict

import numpy as np
import cv2

from scipy import optimize  # for future robust fits


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Vectorised bilinear sampling at floating-point coordinates."""
    h, w = img.shape
    xs = np.clip(xs, 0, w - 2)
    ys = np.clip(ys, 0, h - 2)
    x0 = xs.astype(int)
    y0 = ys.astype(int)
    dx = xs - x0
    dy = ys - y0

    val = (
        (1 - dx) * (1 - dy) * img[y0, x0]
        + dx * (1 - dy) * img[y0, x0 + 1]
        + (1 - dx) * dy * img[y0 + 1, x0]
        + dx * dy * img[y0 + 1, x0 + 1]
    )
    return val


def centroid_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.nan, np.nan
    return float(xs.mean()), float(ys.mean())


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def detect_blobs(
    img: np.ndarray,
    *,
    max_candidates: int = 3,
    threshold_percentile: float = 99.5,
) -> List[Dict]:
    """Detect bright blobs using global percentile threshold and CC analysis."""
    thr = np.percentile(img, threshold_percentile)
    _, bin_img = cv2.threshold(img, int(thr), 255, cv2.THRESH_BINARY)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    blobs: List[Dict] = []
    for i in range(1, num):  # skip background 0
        x, y, w, h, area = stats[i]
        if area < 20:
            continue  # very small – likely noise
        cx, cy = cents[i]
        mask = labels == i
        mean_int = img[mask].mean()
        score = mean_int * area
        blobs.append(dict(mask=mask, bbox=(x, y, w, h), centroid=(cx, cy), area=area, score=score))

    blobs.sort(key=lambda d: d["score"], reverse=True)
    return blobs[:max_candidates]


def baseline_centroid(
    img: np.ndarray,
    mode: str = "core",
    thr_core: int = 240,
    thr_nonblack: int = 20,
) -> Tuple[float, float]:
    """Baseline centroid by binary thresholding."""
    if mode == "core":
        _, mask = cv2.threshold(img, thr_core, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(img, thr_nonblack, 255, cv2.THRESH_BINARY)
    return centroid_from_mask(mask)


def subpixel_centroid(
    img: np.ndarray,
    coarse_center: Tuple[float, float],
    *,
    initial_radius_px: float | None = None,
    num_rays: int = 180,
) -> Tuple[Tuple[float, float], float, np.ndarray]:
    """Sub-pixel centroid via isophote interpolation and circle fit."""
    cx0, cy0 = coarse_center
    if np.isnan(cx0):
        return (np.nan, np.nan), np.nan, np.empty((0, 2))

    # Handle both grayscale and color images
    if len(img.shape) == 3:
        h, w = img.shape[:2]
    else:
        h, w = img.shape
    # Rough radius estimate if none provided (fallback to 1/4 of min dim)
    if initial_radius_px is None:
        # Use low threshold to include blurred edge
        inside_mask = img > 50
        ys, xs = np.nonzero(inside_mask)
        if xs.size > 50:
            initial_radius_px = np.percentile(np.hypot(xs - cx0, ys - cy0), 80)
        else:
            initial_radius_px = min(h, w) / 4

    # Intensities for inside/background to compute I50
    yy = np.arange(h, dtype=np.float64)[:, np.newaxis]
    xx = np.arange(w, dtype=np.float64)[np.newaxis, :]
    dist = np.hypot(xx - cx0, yy - cy0)
    inside_vals = img[(dist < initial_radius_px * 0.5) & (img < 250)]
    bg_vals = img[(dist > initial_radius_px * 1.2)]
    I_in = float(np.median(inside_vals) if inside_vals.size else 255)
    I_bg = float(np.median(bg_vals) if bg_vals.size else 0)
    I50 = 0.5 * (I_in + I_bg)

    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    edge_pts: List[Tuple[float, float]] = []

    max_r = min(h, w) / 2  # sample out to half image diagonal to ensure crossing
    samples = int(max_r * 4)  # 0.25 px steps
    r_vals = np.linspace(0, max_r, samples)

    img_f32 = img.astype(np.float32)

    for cos_ang, sin_ang in zip(cos_angles, sin_angles):
        xs = cx0 + r_vals * cos_ang
        ys = cy0 + r_vals * sin_ang
        vals = _bilinear_sample(img_f32, xs, ys)
        above = vals > I50
        transitions = np.where(above[:-1] & ~above[1:])[0]
        if transitions.size == 0:
            continue
        # pick transition nearest expected radius
        k = transitions[np.argmin(np.abs(r_vals[transitions] - initial_radius_px))]
        v0, v1 = vals[k], vals[k + 1]
        r0, r1 = r_vals[k], r_vals[k + 1]
        if v0 == v1:
            r_edge = r0
        else:
            frac = (v0 - I50) / (v0 - v1)
            r_edge = r0 + frac * (r1 - r0)
        xe = cx0 + r_edge * cos_ang
        ye = cy0 + r_edge * sin_ang
        edge_pts.append((xe, ye))

    edge_pts = np.asarray(edge_pts, dtype=np.float64)
    if edge_pts.shape[0] < 3:
        return (np.nan, np.nan), np.nan, edge_pts

    # Pratt fit for better robustness
    xm = edge_pts[:, 0].mean()
    ym = edge_pts[:, 1].mean()
    x = edge_pts[:, 0] - xm
    y = edge_pts[:, 1] - ym
    z = x**2 + y**2

    z_mean = z.mean()
    zx_mean = (z * x).mean()
    zy_mean = (z * y).mean()

    B = np.array([[ (x * x).mean(), (x * y).mean()],
                  [ (x * y).mean(), (y * y).mean()]])
    C = np.array([zx_mean, zy_mean]) * 0.5

    try:
        a, b = np.linalg.solve(B, C)
        xc = xm + a
        yc = ym + b
        radius = np.sqrt(a * a + b * b + z_mean)
    except np.linalg.LinAlgError:
        # fallback to simple mean
        xc = xm
        yc = ym
        radius = float(np.mean(np.hypot(x, y)))

    # Non-linear least squares refinement (optional)
    try:
        from scipy.optimize import least_squares  # noqa: WPS433

        def residuals(p):
            xc2, yc2, r2 = p
            return np.sqrt((edge_pts[:, 0] - xc2) ** 2 + (edge_pts[:, 1] - yc2) ** 2) - r2

        res = least_squares(residuals, x0=[xc, yc, radius], method='lm', max_nfev=20)
        xc, yc, radius = res.x
    except Exception:
        # scipy not available or fit failed – keep algebraic result
        pass

    return (float(xc), float(yc)), float(radius), edge_pts 