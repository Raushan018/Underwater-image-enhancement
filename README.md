HEAD
# Underwater-image-enhancement
A digital image processing project to improve the quality of underwater images affected by low contrast, color distortion, and haze.
# Underwater Image Enhancement System using DSP and Multiscale Fusion

This project implements an advanced underwater image enhancement pipeline using Digital Signal Processing (DSP) techniques and Multiscale Fusion (Laplacian Pyramid). It features a responsive web application for easy interaction.

## Features

- **White Balancing**: Red/Blue channel compensation + Grey-World Algorithm.
- **Gamma Correction**: Adjusts global contrast.
- **Sharpening**: Unsharp Masking with Gaussian kernel.
- **Luminance Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **Histogram Linearization**: Global contrast stretching.
- **Multiscale Fusion**: Fuses CLAHE and Linearized outputs using Laplacian Pyramid Fusion.
- **Weight Maps**: Laplacian Contrast, Saliency, Saturation.
- **Object Detection**: AI-based detection of Fish, Corals, Divers, and Rocks using YOLO-World.
- **Metrics**: UCIQE, UIQM (Est), Entropy.

## Project Structure

- `src/`: Core DSP algorithms.
- `app.py`: Flask backend server.
- `static/`, `templates/`: Frontend.
- `report/`: Documentation.

## Installation

### Local Setup

1. **Clone the repository** (if applicable).
2. **Install Python 3.9+**.
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```bash
   python app.py
   ```
5. Open your browser at `http://localhost:5000`.

### VS Code Quick Start

1. Open the project folder in **VS Code**.
2. Open a **New Terminal** (`Ctrl` + `` ` ``).
3. Create a virtual environment (optional but recommended):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Run the app:
   ```powershell
   python app.py
   ```
6. Ctrl+Click the URL `http://127.0.0.1:5000` in the terminal to open.

### Docker

1. **Build the image**:
   ```bash
   docker build -t underwater-enhancement .
   ```
2. **Run container**:
   ```bash
   docker run -p 5000:5000 underwater-enhancement
   ```

## Deployment

### Render/Railway
1. Connect your GitHub repository.
2. Set Build Command: `pip install -r requirements.txt`
3. Set Start Command: `python app.py` (or `gunicorn app:app`)

## Results
The system produces intermediate outputs for every stage and a final fused image, displayed in a comparison slider.

## Report
See `report/` for the detailed project report and diagrams.
429aebd (Initial commit)
