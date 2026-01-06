# Vehicle Detection (Speed & Color)

This repository contains a simple computer-vision project for detecting vehicles in video, estimating their speed (in terms of motion per frame), and classifying their color using a K-Nearest Neighbors (KNN) model.

The code is written in Python using OpenCV, NumPy, and scikit-learn. The scripts are currently configured for Google Colab / Google Drive paths, but can also be adapted to run locally.

---

## 📂 Project Structure

- `combine.py`  
  Main script that combines vehicle detection, color classification, and basic speed estimation, and writes an output video.

- `Forvedio.py`  
  Alternative video-based script for vehicle detection, color classification, and speed estimation using bounding-box center movement.

- `Forimage.py`  
  Experimental script for vehicle detection and color classification on video frames (designed for use in Google Colab with `cv2_imshow`).

All scripts use background subtraction (`cv2.createBackgroundSubtractorMOG2`) to detect moving objects and a simple KNN classifier trained on a small set of RGB color samples.

---

## 🧰 Requirements

- Python 3.x
- Recommended to use a virtual environment
- Python packages (listed in `requirements.txt`):
  - `opencv-python`
  - `numpy`
  - `scikit-learn`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

or, if you prefer to install manually:

```bash
pip install opencv-python numpy scikit-learn
```

> **Note:** In `combine.py` and `Forimage.py`, `google.colab.patches` is imported for `cv2_imshow`. This works in Google Colab. For local execution, you can replace `cv2_imshow` with `cv2.imshow` and remove the Colab import.

---

## 🚀 Getting Started (Local)

1. **Clone the repository**

```bash
git clone https://github.com/dumbdead221/vehicledetection.git
cd vehicledetection
```

2. **(Optional) Create and activate a virtual environment**

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Update video input/output paths**

`Forimage.py` still uses hard-coded Google Drive / Colab paths like:

```python
cap = cv2.VideoCapture('/content/drive/MyDrive/Projectmini/Black.mp4')
...
out = cv2.VideoWriter('/content/drive/MyDrive/Projectmini/Modelcheck_output.avi', ...)
```

For local use, change these to point to your local files, for example:

```python
cap = cv2.VideoCapture('input_videos/my_video.mp4')
out = cv2.VideoWriter('output_videos/output.avi', fourcc, output_fps, (output_width, output_height))
```

`combine.py` and `Forvedio.py` instead take input and output paths from command-line arguments (see the usage examples below). Make sure any referenced folders exist.

---

## ▶️ Running the Scripts

### 1. `combine.py`

This script:
- Reads a video.
- Detects moving vehicles using background subtraction.
- Classifies the dominant color of each detected vehicle using KNN.
- Computes a simple speed metric based on the number of detected vehicles per frame.
- Overlays color labels on the video and writes the processed frames to an output video.

Run (with command-line arguments):

```bash
python combine.py --input path/to/input.mp4 --output path/to/output.avi
```

Options:
- `--input` / `--video`: path to input video file (required)
- `--output`: output video path (default: `output.avi`)
- `--no-display`: disable the live display window

### 2. `Forvedio.py`

This script focuses on video-based detection and uses the movement of the bounding box center between frames to estimate speed (in units per frame). It also overlays the average speed text on each frame.

Run (with command-line arguments):

```bash
python Forvedio.py --input path/to/input.mp4 --output path/to/output_forvedio.avi
```

Options:
- `--input` / `--video`: path to input video file (required)
- `--output`: output video path (default: `output_forvedio.avi`)
- `--no-display`: disable the live display window

### 3. `Forimage.py`

This script demonstrates detection and color classification per frame and was written for Google Colab, using `cv2_imshow` to display frames.

- In **Google Colab**: you can run it as-is (after mounting Drive and ensuring the video path is correct).
- For **local use**: replace `cv2_imshow(frame)` with `cv2.imshow('Frame', frame)` and remove the `google.colab.patches` import.

Run:

```bash
python Forimage.py
```

Press `q` to exit the display window if you adapt it to use `cv2.imshow`.

---

## 🧮 How Speed Is Estimated

The project uses simple heuristics rather than calibrated real-world speed:

- `combine.py` counts detected vehicles per frame and computes an "average speed" in terms of detections per frame.
- `Forvedio.py` tracks bounding boxes and computes the Euclidean distance between the center of the previous and current bounding box:
  - This distance (in pixels per frame) is used as a speed-like value.
  - The script overlays an "Average Speed" value in units per frame (labeled as m/s in the overlay, but it is not calibrated to real-world meters).

To convert this to real-world speed, you would need camera calibration and a known scale (pixels-per-meter) and frame rate.

---

## 🎨 Color Detection Approach

The color classifier is a simple KNN model trained on a tiny synthetic dataset:

- Predefined color labels: `['red', 'green', 'blue', 'white', 'black']`
- Corresponding RGB values: `[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]`

For each detected vehicle region:

1. The region is resized (e.g., to 100×100).
2. The mean color over all pixels is computed.
3. KNN predicts the closest label based on that mean RGB value.

This is a very basic approach and mainly works for clear, saturated colors in decent lighting.

---

## 🚧 Limitations & Notes

- Paths are currently hard-coded for Google Colab / Google Drive; they should be adjusted for your environment.
- Speed is not calibrated to real-world units; it is based on frame-to-frame motion.
- Color classification uses a very small synthetic dataset and may struggle with:
  - Poor lighting
  - Shadows
  - Similar shades (e.g., gray vs. white)
- No multi-object ID tracking is implemented; detections are treated frame by frame or with a very simple previous-bbox comparison.

---

## 🗺️ Possible Future Improvements

- Replace hard-coded paths with command-line arguments.
- Add proper object tracking (e.g., with SORT / Deep SORT).
- Calibrate the camera and convert pixel motion to real-world speed (km/h or m/s).
- Use a more robust color classification model with a larger dataset.
- Provide a configuration file for thresholds and parameters.

---

## 🤝 Contributing

Feel free to fork this repository and experiment with improvements. Suggestions and pull requests to clean up paths, add configuration, or improve detection/estimation are welcome.

---

## 📄 License

No explicit license has been provided yet. By default, this means all rights are reserved by the author unless a license is added.