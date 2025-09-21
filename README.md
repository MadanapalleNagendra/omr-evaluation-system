# OMR Evaluation System

A robust, production-ready Optical Mark Recognition (OMR) system for automated evaluation of answer sheets. This project features overlays, detection, authentication, batch processing, and a modern Streamlit UI with results visualization.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Activate the Virtual Environment](#3-activate-the-virtual-environment)
  - [4. Install Dependencies](#4-install-dependencies)
  - [5. Run the Application](#5-run-the-application)
- [Usage Guide](#usage-guide)
- [Authentication](#authentication)
- [OMR Processing Pipeline](#omr-processing-pipeline)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Logs & Debugging](#logs--debugging)
- [Extending the System](#extending-the-system)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Overview
This OMR system automates the evaluation of answer sheets using computer vision and machine learning. It supports batch processing, overlays, user authentication, and provides a modern, user-friendly web interface for uploading, processing, and visualizing results.

## Features
- Secure login page (username/password)
- Batch upload and processing of OMR sheets
- Automatic detection and classification of marked bubbles
- Overlay and debug image generation
- Results visualization (charts, stats)
- Modern, professional UI (light theme)
- Robust error handling and logging
- Configurable via YAML

## Project Structure
```
app.py                  # Main Streamlit app (UI, login, OMR, visualization)
requirements.txt        # Python dependencies
README.md               # Project documentation (this file)
configs/
  config.yaml           # OMR and app configuration
scripts/
  classify_marks.py     # Bubble detection/classification logic
  detect_and_warp.py    # Image preprocessing and warping
  extract_grid.py       # Grid extraction logic
  preprocess.py         # Image preprocessing utilities
  run_batch.py          # Batch OMR processing pipeline
  score.py              # Scoring logic
  utils.py              # Utility functions (logging, debug, etc.)
data/
  Set A/                # Example OMR images (Set A)
  Set B/                # Example OMR images (Set B)
logs/                   # Log files and debug images
omr_env/                # Python virtual environment (optional)
omr_env311/             # Python virtual environment (optional)
Key (Set A and B).xlsx  # Answer key (example)
```

### Structure Diagram

```
[app.py] -- main UI/login/visualization
  |
  |-- [scripts/]
  |     |-- classify_marks.py
  |     |-- detect_and_warp.py
  |     |-- extract_grid.py
  |     |-- preprocess.py
  |     |-- run_batch.py
  |     |-- score.py
  |     |-- utils.py
  |
  |-- [configs/]
  |     |-- config.yaml
  |
  |-- [data/]
  |     |-- Set A/ (images)
  |     |-- Set B/ (images)
  |
  |-- [logs/]
  |
  |-- [requirements.txt]
  |-- [README.md]
  |-- [Key (Set A and B).xlsx]
```

---

## Setup Instructions

### 1. Clone the Repository
Clone or download the project to your local machine.

```
git clone <repo-url>
cd <project-folder>
```

### 2. Create a Virtual Environment
Create a Python virtual environment (recommended for isolation):

**Windows (PowerShell):**
```
python -m venv omr_env
```

**Linux/Mac:**
```
python3 -m venv omr_env
```

### 3. Activate the Virtual Environment

**Windows (PowerShell):**
```
.\omr_env\Scripts\Activate.ps1
```

**Linux/Mac:**
```
source omr_env/bin/activate
```

### 4. Install Dependencies
Install all required Python packages:
```
pip install -r requirements.txt
```

### 5. Run the Application
Start the Streamlit app:
```
streamlit run app.py
```

---

## Usage Guide
1. Open the app in your browser (Streamlit will provide a local URL).
2. Log in with your credentials (default: admin/admin123).
3. Upload OMR images or select a batch folder.
4. Process the images and review overlays/debug images.
5. View results and visualizations.
6. Download/export results as needed.

## Authentication
- The app requires login before accessing OMR features.
- User credentials can be managed in the code or via a users.json file (future feature).

## OMR Processing Pipeline
- Images are preprocessed, deskewed, and bubbles detected/classified.
- Overlays and debug images are generated for review.
- Results are scored and visualized.

## Visualization
- After processing, results are shown as tables and interactive charts (per-question stats, overall scores, etc.).
- Visualizations use Streamlit and Plotly for clarity and interactivity.

## Configuration
- All main settings are in `configs/config.yaml` (paths, thresholds, debug mode, etc.).
- Edit this file to adjust OMR parameters or app behavior.

## Logs & Debugging
- All logs and debug images are saved in the `logs/` folder.
- Errors and warnings are shown in the app and logged for troubleshooting.

## Extending the System
- Add new models or detection logic in `scripts/classify_marks.py` or `scripts/detect_and_warp.py`.
- Update the UI in `app.py` for new features.
- Add new configuration options in `configs/config.yaml`.

## Troubleshooting
- If you see errors, check the logs/ folder and Streamlit error messages.
- Ensure your virtual environment is activated and all dependencies are installed.
- For image processing errors, check your input image quality and config settings.

## License
This project is for educational and research use. Contact the author for commercial licensing.

---

For questions or support, please contact the project maintainer.
