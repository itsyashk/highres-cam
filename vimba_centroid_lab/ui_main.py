"""ui_main.py – PySide6 GUI for Vimba Centroid Lab."""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from queue import Queue
from typing import Any

from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QCheckBox,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from .camera_vimba import CameraController
from .processing import (
    detect_blobs,
    baseline_centroid,
    subpixel_centroid,
)
from .viz import overlay_centroids, render_zoom_roi


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.setWindowTitle("Vimba Centroid Lab")
        self.config = config

        # UI elements ---------------------------------------------------
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.start_btn = QPushButton("Start")
        self.capture_btn = QPushButton("Capture Series")

        # Sliders -------------------------------------------------------
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(100, 20000)  # microseconds
        self.exposure_slider.setValue(5000)
        self.exposure_slider.valueChanged.connect(lambda v: self.cam.set_exposure(float(v)))

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 240)  # 0–24 dB *10
        self.gain_slider.setValue(0)
        self.gain_slider.valueChanged.connect(lambda v: self.cam.set_gain(v / 10))

        # Capture series length
        self.spin_series_len = QSpinBox()
        self.spin_series_len.setRange(1, 1000)
        self.spin_series_len.setValue(self.config.get("capture_series_length", 50))

        # Pixel size & calibration
        self.edit_pixel_size = QDoubleSpinBox()
        self.edit_pixel_size.setDecimals(5)
        self.edit_pixel_size.setRange(0.001, 10)
        self.edit_pixel_size.setValue(self.config.get("pixel_size_mm_per_px", 0.6))

        self.edit_known_diam = QDoubleSpinBox()
        self.edit_known_diam.setDecimals(3)
        self.edit_known_diam.setRange(0.1, 100)
        self.edit_known_diam.setValue(self.config.get("known_diameter_mm", 12.7))

        self.btn_calibrate = QPushButton("Calibrate Scale")
        self.btn_calibrate.clicked.connect(self._calibrate_scale)

        # Zoom/bicubic toggle
        self.zoom_label = QLabel(alignment=Qt.AlignCenter)
        self.chk_bicubic = QCheckBox("Smooth (bicubic)")
        self.chk_bicubic.setChecked(False)

        # Stats HUD
        self.label_stats = QLabel("Δ px: -- | μm: -- | σ(px): -- | σ(μm): --")

        # Matplotlib figure for live plot
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)

        # Left column: live image and controls
        layout_left = QVBoxLayout()
        layout_left.addWidget(self.image_label)

        ctrl_exp = QHBoxLayout()
        ctrl_exp.addWidget(QLabel("Exposure (µs)"))
        ctrl_exp.addWidget(self.exposure_slider)
        layout_left.addLayout(ctrl_exp)

        ctrl_gain = QHBoxLayout()
        ctrl_gain.addWidget(QLabel("Gain (dB)"))
        ctrl_gain.addWidget(self.gain_slider)
        layout_left.addLayout(ctrl_gain)

        layout_left.addWidget(self.start_btn)

        # Capture series controls
        cap_layout = QHBoxLayout()
        cap_layout.addWidget(self.capture_btn)
        cap_layout.addWidget(QLabel("K:"))
        cap_layout.addWidget(self.spin_series_len)
        layout_left.addLayout(cap_layout)

        # Calibration controls
        calib_layout = QHBoxLayout()
        calib_layout.addWidget(QLabel("px size (mm)"))
        calib_layout.addWidget(self.edit_pixel_size)
        calib_layout.addWidget(QLabel("Ø mm"))
        calib_layout.addWidget(self.edit_known_diam)
        calib_layout.addWidget(self.btn_calibrate)
        layout_left.addLayout(calib_layout)

        layout_left.addWidget(self.label_stats)

        # Right side: zoom ROI, bicubic toggle, plot
        layout_zoom = QVBoxLayout()
        layout_zoom.addWidget(self.zoom_label)
        layout_zoom.addWidget(self.chk_bicubic)
        layout_zoom.addWidget(self.canvas)

        layout_main = QHBoxLayout()
        layout_main.addLayout(layout_left, stretch=3)
        layout_main.addLayout(layout_zoom, stretch=2)

        container = QWidget()
        container.setLayout(layout_main)
        self.setCentralWidget(container)

        # Camera + timer -------------------------------------------------
        self.frame_queue: Queue[np.ndarray] = Queue(maxsize=3)
        self.cam = CameraController(self.frame_queue)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._pump_frames)

        # Signals --------------------------------------------------------
        self.start_btn.clicked.connect(self._toggle_stream)
        self.capture_btn.clicked.connect(self._toggle_capture_series)
        self.image_label.mousePressEvent = self._on_click  # type: ignore

        self.latest_frame: np.ndarray | None = None
        self.selected_blob_center: tuple[float, float] | None = None

        # Data storage for plotting ------------------------------------
        self.series_data: list[dict] = []
        self._series_length_target = 0
        self._pixel_size_mm = self.edit_pixel_size.value()

    # ------------------------------------------------------------------
    # Camera handling
    # ------------------------------------------------------------------
    def _toggle_stream(self):
        if self.timer.isActive():
            self.cam.stop()
            self.timer.stop()
            self.start_btn.setText("Start")
        else:
            self.cam.start()
            self.timer.start(30)
            self.start_btn.setText("Stop")

    def _pump_frames(self):
        if self.frame_queue.empty():
            return
        frame = self.frame_queue.get()
        self.latest_frame = frame

        disp = frame
        if self.selected_blob_center is not None:
            baseline = baseline_centroid(
                frame,
                mode="core",
                thr_core=self.config.get("threshold_core", 240),
                thr_nonblack=self.config.get("threshold_non_black", 20),
            )
            refined, radius, _ = subpixel_centroid(
                frame,
                baseline,
                num_rays=self.config.get("number_of_rays", 180),
            )
            disp = overlay_centroids(frame, baseline, refined)

            # Zoom pane -------------------------------------------------
            scale = self.config.get("zoom_scale", 16)
            roi_size = self.config.get("roi_size_px", 40)
            x0, y0 = map(int, refined if not np.isnan(refined[0]) else baseline)
            y0 = np.clip(y0, roi_size // 2, frame.shape[0] - roi_size // 2 - 1)
            x0 = np.clip(x0, roi_size // 2, frame.shape[1] - roi_size // 2 - 1)
            roi = frame[y0 - roi_size // 2 : y0 + roi_size // 2, x0 - roi_size // 2 : x0 + roi_size // 2]
            zoom_img = render_zoom_roi(roi, scale=scale, bicubic=self.chk_bicubic.isChecked())
            qimg_zoom = QImage(zoom_img.data, zoom_img.shape[1], zoom_img.shape[0], zoom_img.strides[0], QImage.Format_BGR888)
            self.zoom_label.setPixmap(QPixmap.fromImage(qimg_zoom))

            # Update capture series ------------------------------------
            if self._series_length_target > 0:
                delta_px = float(np.hypot(refined[0] - baseline[0], refined[1] - baseline[1]))
                self.series_data.append(
                    dict(
                        frame=len(self.series_data),
                        baseline_x=baseline[0],
                        baseline_y=baseline[1],
                        refined_x=refined[0],
                        refined_y=refined[1],
                        radius_px=radius,
                        diameter_px=2 * radius,
                        delta_px=delta_px,
                    )
                )
                if len(self.series_data) >= self._series_length_target:
                    self._finalize_capture_series()

            # Update live plot ----------------------------------------
            self._update_plot()

        # Convert to QImage/Pixmap -------------------------------------
        if disp.ndim == 2:
            h, w = disp.shape
            qimg = QImage(disp.data, w, h, disp.strides[0], QImage.Format_Grayscale8)
        else:
            h, w, _ = disp.shape
            qimg = QImage(disp.data, w, h, disp.strides[0], QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------
    def _on_click(self, event):
        if self.latest_frame is None:
            return
        # For now simply store that a blob is selected – more sophisticated selection later.
        self.selected_blob_center = (event.pos().x(), event.pos().y())

    # ------------------------------------------------------------------
    # Capture series helpers
    # ------------------------------------------------------------------
    def _toggle_capture_series(self):
        if self._series_length_target == 0:
            self._series_length_target = self.spin_series_len.value()
            self.series_data.clear()
            self.label_stats.setText("Capturing series...")
        else:
            # cancel
            self._series_length_target = 0

    def _finalize_capture_series(self):
        self._series_length_target = 0
        if not self.series_data:
            return
        import csv, datetime

        output_dir = Path(self.config.get("output_dir", "output"))
        output_dir.mkdir(exist_ok=True, parents=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"series_{ts}.csv"
        keys = self.series_data[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.series_data)

        # Compute stats
        deltas = np.array([d["delta_px"] for d in self.series_data])
        std_px = float(deltas.std(ddof=1)) if deltas.size > 1 else float("nan")
        std_um = std_px * self._pixel_size_mm * 1000
        self.label_stats.setText(
            f"Δ px avg: {deltas.mean():.3f} | σ(px): {std_px:.4f} | σ(µm): {std_um:.3f}"
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _calibrate_scale(self):
        if not self.series_data:
            return
        last = self.series_data[-1]
        measured_diam_px = last.get("diameter_px")
        if not measured_diam_px or np.isnan(measured_diam_px):
            return
        known_mm = self.edit_known_diam.value()
        self._pixel_size_mm = known_mm / measured_diam_px
        self.edit_pixel_size.setValue(self._pixel_size_mm)
        # Update stats display to reflect new units
        self._finalize_capture_series()

    # ------------------------------------------------------------------
    # Plot updating
    # ------------------------------------------------------------------
    def _update_plot(self):
        self.ax.cla()
        if self.series_data:
            xs = [d["frame"] for d in self.series_data]
            ys = [d["delta_px"] for d in self.series_data]
            self.ax.plot(xs, ys, "o", markersize=4)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Δ centroid (px)")
        self.ax.set_title("Baseline vs Sub-pixel difference")
        self.ax.grid(True, linestyle=":", linewidth=0.5)
        self.canvas.draw_idle()

    def _save_current_frame(self):
        if self.latest_frame is None:
            return
        default_dir = Path(self.config.get("output_dir", "output"))
        default_dir.mkdir(exist_ok=True, parents=True)
        fname, _ = QFileDialog.getSaveFileName(self, "Save image", str(default_dir / "frame.png"), "PNG (*.png)")
        if fname:
            cv2.imwrite(fname, self.latest_frame) 