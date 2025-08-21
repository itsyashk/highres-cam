"""Synthetic unit test verifying sub-pixel centroid accuracy."""
import numpy as np

from vimba_centroid_lab.processing import baseline_centroid, subpixel_centroid


def _synthetic_disk(h: int = 512, w: int = 512, cx: float = 256.4, cy: float = 200.7, sigma: float = 3.0, radius: float = 100):
    yy, xx = np.indices((h, w))
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    img = 255 * np.exp(-(np.sqrt(r2) - radius) ** 2 / (2 * sigma ** 2))
    img += np.random.normal(0, 1.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_subpixel_precision():
    img = _synthetic_disk()
    base = baseline_centroid(img, mode="nonblack", thr_nonblack=20)
    refined, radius, pts = subpixel_centroid(img, base, num_rays=360)

    err_base = np.hypot(base[0] - 256.4, base[1] - 200.7)
    err_refined = np.hypot(refined[0] - 256.4, refined[1] - 200.7)

    assert err_refined < 0.02, f"Sub-pixel error too high: {err_refined:.4f} px"
    assert err_refined < err_base, "Sub-pixel method should beat baseline centroid" 