# PA-SQGN — Vision-Based Queue Length Estimation

<p align="center">
  <img src="outputs/queue_output_preview.png" alt="PA-SQGN Demo" width="720"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/YOLOv8-Pose-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/LangChain-Gemini-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

> **PA-SQGN** (Pose-Aware Spatiotemporal Queue Graph Network) is a real-time, vision-based queue length estimation system developed as an M.Tech Computer Vision project. It uses pose keypoints, graph neural reasoning, and temporal smoothing to count queue members from a video feed — with no sensors or infrastructure required.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Novel Contributions](#novel-contributions)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Run the main pipeline](#1-run-the-main-pipeline)
  - [Run the Streamlit dashboard](#2-run-the-streamlit-dashboard)
  - [Run ablation study](#3-run-ablation-study)
  - [Evaluate metrics](#4-evaluate-metrics)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Ablation Study](#ablation-study)
- [Requirements](#requirements)
- [Author](#author)

---

## Overview

Traditional queue management relies on physical sensors, floor markings, or manual counting. PA-SQGN solves this with a pure computer vision pipeline:

1. Detects people using **YOLOv8n-Pose** (bounding boxes + 17 keypoints)
2. Tracks them across frames using **ByteTrack**
3. Builds a **pose-aware graph** where edge weights encode spatial proximity, facing direction, and velocity alignment
4. Extracts the queue spine using **MST longest-path**
5. Smooths membership using an **EMA temporal confidence model**
6. Classifies the scene as `queue`, `crowd`, or `uncertain`
7. Generates **natural language advisories** via LangChain + Gemini

---

## Key Features

- Real-time processing at 25 FPS on standard hardware
- No sensors, floor markings, or infrastructure required
- Works on any fixed-camera video (MP4, AVI, MOV)
- Streamlit web dashboard with live charts and video playback
- LLM-powered advisory messages for venue staff
- Full ablation study comparing 3 configurations
- CSV metrics logging + annotated output video

---

## System Architecture

```
Streamlit Video Upload (960×540)
        ↓
YOLOv8n-Pose — bounding boxes + 17 COCO keypoints
        ↓
ByteTrack — stable track_id per person
        ↓
Pose Feature Extraction — pos (cx,cy), angle θ, velocity (dx,dy)
        ↓
Pose-Aware Graph — edge weight = spatial × orientation × velocity  ← NOVEL
        ↓
MST Longest Path — queue spine, sorted front-to-back
        ↓
Temporal Smoother (EMA) — per-person confidence score              ← NOVEL
        ↓
Scene Classifier — queue / crowd / uncertain (5 graph features)    ← NOVEL
        ↓
    ┌───────────────────────┐
    ↓                       ↓
MetricsLogger         QueueAdvisoryAgent
CSV + Video           LangChain + Gemini
    ↓                       ↓
Streamlit Dashboard (4 tabs)
```

---

## Novel Contributions

| Contribution | Description |
|---|---|
| **3-term edge weight** | `w = exp(−d/s) × cos(Δθ) × vel_coherence` — no prior queue paper uses orientation × velocity together |
| **EMA confidence smoother** | Per-person float score with asymmetric rise (0.35) vs decay (0.25), replacing crude frame counters |
| **5-feature scene classifier** | angle_variance, aspect_ratio, avg_edge_weight, density, linearity_score |
| **LLM advisory layer** | LangChain + Gemini 2.0 Flash with rule-based fallback |

---

## Project Structure

```
vision_queue_length_estimation/
├── app.py                        # Streamlit dashboard (4 tabs)
├── main_pose.py                  # Main pipeline — Config C (full PA-SQGN)
├── main.py                       # Basic entry point
├── requirements.txt
├── .env                          # API keys (not committed)
├── yolov8n-pose.pt               # YOLOv8 pose model weights
├── yolov8n.pt                    # YOLOv8 base model weights
│
├── queue_detection/
│   ├── pose_graph.py             # Graph construction + MST longest path
│   ├── hybrid_queue.py           # Config A — PCA + kNN + MST baseline
│   ├── projection_queue.py       # PCA-based direction and metrics
│   ├── temporal_smoother.py      # EMA confidence smoother
│   ├── crowd_classifier.py       # 5-feature scene classifier
│   ├── clustering.py
│   ├── graph_queue.py
│   ├── mst_queue.py
│   ├── queue_detector.py
│   └── spatial_analysis.py
│
├── scripts/
│   ├── run_hybrid.py             # Ablation Config A
│   ├── run_pose_only.py          # Ablation Config B
│   ├── run_ablation.py           # Run all 3 configs + comparison table
│   └── evaluate_metrics.py       # MAE, RMSE, DeltaStd, confusion matrix
│
├── tracking/
│   └── bytetrack.py
│
├── utils/
│   ├── langchain_alerts.py       # QueueAdvisoryAgent (LangChain + Gemini)
│   ├── metrics_logger.py         # CSV + video writer
│   ├── video_utils.py
│   └── visualization.py
│
├── data/
│   ├── videos/                   # Input videos (not committed)
│   └── ground_truth.csv          # Manual annotations for evaluation
│
├── outputs/                      # Generated outputs (not committed)
│   ├── queue_log.csv
│   ├── queue_output.mp4
│   ├── ablation_A.csv
│   ├── ablation_B.csv
│   ├── queue_over_time.png
│   └── confusion_matrix.png
│
├── configs/
├── models/
├── notebooks/
└── test_day1.py ... test_day4.py
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sagnik120/vision_queue_length_estimation.git
cd vision_queue_length_estimation
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key (optional — for LLM advisories)

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

If you skip this, the system runs fine with rule-based fallback advisories.

### 5. Download model weights

The YOLOv8 weights are downloaded automatically on first run via the `ultralytics` package. If you already have `yolov8n-pose.pt` and `yolov8n.pt`, place them in the project root.

---

## Usage

### 1. Run the main pipeline

Processes a video and generates annotated output + CSV metrics:

```bash
python main_pose.py
```

By default it reads from `data/videos/test_queue.mp4`. To change the input video, edit this line in `main_pose.py`:

```python
VIDEO_PATH = "data/videos/your_video.mp4"
```

Press `Q` to stop processing early. Output files appear in `outputs/`.

---

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`. The dashboard has 4 tabs:

| Tab | Contents |
|---|---|
| Results Dashboard | Live charts, queue size over time, scene label distribution, advisory log, output video |
| Ablation Study | Config A vs B vs C comparison table, stability charts, confusion matrix |
| Architecture & Novelty | Pipeline stage table, novel contributions, comparison vs 5 prior papers |
| Live Demo | Upload any video, run the full pipeline, view results instantly |

---

### 3. Run ablation study

Runs all 3 configurations and prints a comparison table:

```bash
# Config A — hybrid geometric baseline (no pose)
python scripts/run_hybrid.py

# Config B — pose graph only (no smoother/classifier)
python scripts/run_pose_only.py

# Config C — full PA-SQGN (runs automatically via main_pose.py)
python main_pose.py

# Print comparison table
python scripts/run_ablation.py
```

---

### 4. Evaluate metrics

Requires a filled `data/ground_truth.csv`. Generate the template first:

```bash
python scripts/evaluate_metrics.py
```

If `ground_truth.csv` does not exist, the script creates a template. Fill in `gt_size` (manual count) and `gt_label` (`queue` / `crowd` / `uncertain`) for each frame, then re-run.

Outputs: MAE, RMSE, NMAE%, DeltaStd jitter, per-scene breakdown, confusion matrix PNG.

---

## Configuration

Key parameters you can change in `main_pose.py`:

| Parameter | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `data/videos/test_queue.mp4` | Input video path |
| `OUTPUT_VIDEO` | `outputs/queue_output.mp4` | Annotated output video |
| `OUTPUT_CSV` | `outputs/queue_log.csv` | Metrics CSV path |
| `SMOOTH_WINDOW` | `5` | Rolling average window for queue size |
| `MAX_TRACK_HISTORY` | `5` | Frames of position history per person |

Key parameters in `queue_detection/temporal_smoother.py`:

| Parameter | Default | Description |
|---|---|---|
| `RISE_RATE` | `0.35` | How fast confidence rises when in queue |
| `DECAY_RATE` | `0.25` | How fast confidence falls when not in queue |
| `STABLE_THRESHOLD` | `0.55` | Minimum score to count as queue member |
| `FORGET_THRESHOLD` | `0.05` | Score below which person is removed from memory |

---

## Output Files

| File | Description |
|---|---|
| `outputs/queue_log.csv` | Frame-by-frame metrics: queue size, length, scene label, confidence, advisory |
| `outputs/queue_output.mp4` | Annotated video with overlays: bounding boxes, queue spine, heatmap, HUD |
| `outputs/ablation_A.csv` | Config A (hybrid baseline) metrics |
| `outputs/ablation_B.csv` | Config B (pose only) metrics |
| `outputs/queue_over_time.png` | Ablation comparison plot |
| `outputs/confusion_matrix.png` | Scene classifier confusion matrix |

---

## Ablation Study

Three configurations are compared:

| Config | Components | Key difference |
|---|---|---|
| A — Hybrid baseline | YOLOv8 + PCA + kNN + MST | No pose orientation, no smoother |
| B — Pose graph only | YOLOv8-Pose + 3-term graph + MST | Adds orientation and velocity, no smoother |
| C — Full PA-SQGN | All of B + EMA smoother + CrowdClassifier + LLM | Complete system |

Expected result: Config C reduces DeltaStd jitter vs Config A, with similar or better MAE. Config B and C agree on queue membership, confirming pose orientation is the key filtering mechanism.

---

## Requirements

```
ultralytics>=8.0.0
supervision>=0.16.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
networkx>=3.0
scikit-learn>=1.3.0
streamlit>=1.28.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
langchain>=0.1.0
langchain-google-genai>=0.0.6
python-dotenv>=1.0.0
torch>=2.0.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Author

**Sagnik** — M.Tech Computer Vision  
GitHub: [@Sagnik120](https://github.com/Sagnik120)  
Repository: [vision_queue_length_estimation](https://github.com/Sagnik120/vision_queue_length_estimation)
