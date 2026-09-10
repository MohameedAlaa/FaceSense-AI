# FaceSense AI

Real-time facial expression recognition using a lightweight Residual CNN, OpenCV face detection, and a feedback-driven continuous improvement pipeline.

## Overview

FaceSense AI is a computer vision project for facial expression classification.

The system detects faces from images or a webcam stream, preprocesses each detected face, and classifies the visible facial expression into one of seven FER2013 classes.

The project also includes a feedback pipeline that allows validated user feedback to be collected and used for future model improvement.

> **Note:** The model performs facial expression classification from visual patterns. It does not determine a person's true internal emotional state.

## Features

- Facial expression classification with a Residual CNN
- Real-time webcam inference
- Multi-face detection
- OpenCV Haar Cascade face detection
- Confidence-based prediction with an `Uncertain` state
- User feedback collection
- Feedback dataset generation
- Controlled model retraining
- Validation-based model comparison
- Safe model promotion with rollback protection
- Reproducible training experiments
- Automated test suite

## Emotion Classes

The model classifies seven facial expression categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Model Performance

### Current Production Model

**ResidualEmotionCNN-Candidate-epoch39**

| Metric | Score |
|---|---:|
| Test Accuracy | 62.84% |
| Test Macro F1 | 61.43% |
| Test Weighted F1 | 62.25% |

Improvement over the previous production model:

| Metric | Improvement |
|---|---:|
| Accuracy | +7.22 points |
| Macro F1 | +10.85 points |

## System Pipeline

```text
Camera / Image
      ↓
Face Detection
      ↓
Face Cropping
      ↓
48×48 Grayscale Preprocessing
      ↓
ResidualEmotionCNN
      ↓
Expression + Confidence
      ↓
User Feedback
      ↓
Validated Feedback Dataset
      ↓
Controlled Retraining
      ↓
Candidate Evaluation
      ↓
Promotion Gate
      ↓
Production Model
```

## Project Structure

```text
FaceSense-AI/
│
├── configs/
│
├── ml/
│   ├── data/
│   ├── detection/
│   ├── evaluation/
│   ├── feedback/
│   ├── inference/
│   ├── models/
│   └── training/
│
├── notebooks/
│   ├── 01_data_pipeline_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_baseline_training_analysis.ipynb
│
├── scripts/
│   └── benchmark_v2.py
│
├── tests/
│
├── .gitignore
└── README.md
```

## Dataset

The project uses FER2013 with seven expression classes.

The original dataset is intentionally excluded from the Git repository.

## Training

The main interactive training workflow is:

```text
notebooks/02_model_training.ipynb
```

The notebook contains:

- experiment configuration
- dataset inspection
- baseline evaluation
- model configuration
- training
- training curves
- test evaluation
- per-class analysis
- confusion matrices
- production vs candidate comparison
- promotion decision
- experiment summary

Core training logic remains inside the reusable Python modules under `ml/`.

## Inference

### Single Image

```powershell
.\.venv\Scripts\python.exe ml/inference/predict.py --image "path/to/image.jpg"
```

### Webcam

```powershell
.\.venv\Scripts\python.exe ml/inference/webcam.py
```

Press:

- `Q` or `ESC` to exit
- `S` to save an annotated snapshot

## Testing

Run the complete test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Current test status:

```text
56 / 56 tests passed
```

## Architecture

The project follows a modular architecture:

```text
Notebook
    ↓
Data Layer
    ↓
Model Layer
    ↓
Training Layer
    ↓
Evaluation Layer
    ↓
Feedback Layer
    ↓
Model Comparison
    ↓
Promotion Gate
```

The notebook is responsible for experiment orchestration, visualization, and analysis, while reusable implementation logic remains inside `ml/`.

## Current Limitations

- FER2013 contains significant class imbalance.
- `Fear` remains the weakest class in the current model.
- Webcam performance depends on CPU and camera conditions.
- Haar Cascade detection can be sensitive to pose and lighting.
- The current feedback dataset is still small and needs real user-labeled samples for meaningful continuous-learning experiments.

## Future Improvements

- Collect a larger validated feedback dataset
- Improve minority-class performance
- Experiment with stronger face detectors
- Explore transfer learning
- Add richer error analysis
- Build a production API and frontend
- Add model monitoring and experiment tracking
