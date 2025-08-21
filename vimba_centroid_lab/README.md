# Vimba Centroid Lab

Evaluate centroid precision of bright fiducials using an Allied Vision camera.

## 1. Hardware & SDK

1. Install Allied Vision **Vimba SDK** for Linux.
   ```bash
   chmod +x Vimba_X.Y.Z_linux.tgz.run
   sudo ./Vimba_X.Y.Z_linux.tgz.run
   ```
2. Add udev rules (installer offers to do this). Then **unplug / re-plug** the camera or reboot.
3. Verify the Python binding:
   ```bash
   python3 - <<'PY'
   from vimba import Vimba
   with Vimba.get_instance() as v:
       print("Vimba version:", v.get_version())
   PY
   ```

## 2. Clone & set-up the project

```bash
# get the source
git clone <repo> highres-cam && cd highres-cam
# create & activate Python virtual-env
python3 -m venv .venv && source .venv/bin/activate
# install packages (VimbaPython is already on the system)
pip install -r vimba_centroid_lab/requirements.txt
```

## 3. Run the application

```bash
python -m vimba_centroid_lab
```

If no Allied Vision camera is detected, a **synthetic mock camera** appears so you can still explore the UI.

## 4. UI Reference

* **Start/Stop** – begin or halt live acquisition.
* **Exposure (µs) / Gain (dB)** – sliders change camera parameters on the fly.
* **Capture Series (K)** – set length, press once → collect K frames, auto-export CSV & show σ.
* **Pixel size / Ø mm + Calibrate Scale** – enter the real diameter, click calibrate to update µm/px.
* **Zoom pane** – click a blob in the live view; ROI magnifies ×16 with pixel grid. Toggle *Smooth* for bicubic.
* **Plot** – live graph of |baseline – sub-pixel| per frame in pixels.

## 5. CSV Fields

| column        | meaning                             |
|---------------|-------------------------------------|
| frame         | frame index in the series           |
| baseline_x/y  | centroid from threshold mask        |
| refined_x/y   | sub-pixel centroid (circle fit)     |
| radius_px     | fitted radius (px)                  |
| diameter_px   | 2·radius                            |
| delta_px      | distance between the two centroids  |

The HUD computes mean Δ and σ in pixels and µm using the current pixel size.

## 6. Tests

Run `pytest -q` – it generates a synthetic blurred disk and checks that the sub-pixel estimator recovers the true center within ≤ 0.02 px and beats the baseline centroid.

## 7. Troubleshooting

* **Camera not found** – ensure the interface (GigE/USB) is up, the IP (e.g. `10.64.5.85`) is reachable, and you have permissions.
* **`ImportError: vimba`** – run the Vimba installer; the Python wheels are not on PyPI.
* **No Qt platform plugin** – install `libxcb-xinerama0` (Ubuntu: `sudo apt install libxcb-xinerama0`). 