# FaceSense AI

End-to-end facial expression recognition system combining a custom lightweight Residual CNN (PyTorch), a security-hardened REST API (FastAPI), an active user-feedback retraining loop with rollback protection, and an interactive web dashboard (React 19 + Tailwind CSS).

---

> **Responsible AI & Scope Notice**  
> FaceSense AI classifies morphological facial expressions into discrete visual categories based on the FER2013 taxonomy (*Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise*). Visual facial expressions do not have an invariant one-to-one correspondence with internal human emotional states. The project is designed for human-in-the-loop visual analysis and does not perform biometric identification, affective mind-reading, or covert surveillance.

---

## System Overview

FaceSense AI bridges the gap between machine learning experimentation and production-style engineering. Rather than treating modeling as an isolated Jupyter notebook exercise, the project implements a complete lifecycle:

1. **Client Capture:** Webcam frames or image uploads processed in a responsive web interface.
2. **Detection & Normalization:** OpenCV Haar Cascade face detection with input downscaling and CLAHE contrast equalization.
3. **Inference & Uncertainty:** 48×48 grayscale tensor transformation and batched forward-pass classification via `ResidualEmotionCNN` with confidence thresholding and an explicit `Uncertain` state.
4. **Active Feedback:** User-corrected labels collected via a protected API and validated into structured datasets.
5. **Continuous Retraining & Gate:** Controlled retraining pipelines with strict validation-based model comparison, candidate evaluation, and automated rollback guards.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Deep Learning & ML** | PyTorch, torchvision, NumPy, SciPy, scikit-learn |
| **Computer Vision** | OpenCV (`cv2`), Haar Feature-based Cascade Classifiers, CLAHE |
| **Backend & API** | FastAPI, Pydantic v2, Starlette, Uvicorn, Python 3.11 |
| **Database & ORM** | PostgreSQL (optional for production persistence), SQLAlchemy 2.0, Alembic, psycopg |
| **Security & Auth** | Argon2id (`argon2-cffi`), PyJWT, HTTP Bearer tokens, CORS validation, HTTP security headers |
| **Frontend** | React 19, Vite, Tailwind CSS v4, React Router v7, Canvas API |
| **Testing & Quality** | pytest (ML & Backend suites), Node.js native test runner (`node:test`), Oxlint |

---

## End-to-End System Pipeline

```text
       Webcam Stream / Image Upload (React 19 Frontend)
                             │
                             ▼  [JPEG Q=0.85 payload / Multipart Upload]
                FastAPI Security & Upload Gate
       (Rate limiting, MIME verification, 10MB size limit)
                             │
                             ▼
               Face Detection & Preprocessing
       (Haar Cascade, max-dim 640 downscaling, CLAHE, 48×48 crop)
                             │
                             ▼
                  Batched Model Inference
           (ResidualEmotionCNN with Residual Blocks)
                             │
                             ▼
         Confidence Thresholding (Default: 0.50)
             ┌───────────────┴───────────────┐
             ▼                               ▼
  Predicted Expression + Conf.          Uncertain State
             │                               │
             └───────────────┬───────────────┘
                             ▼
                 Client Display & Review
          (Canvas bounding boxes & probability bars)
                             │
                             ▼
                  User Feedback Submission
            (Correct / Incorrect / Corrected Class)
                             │
                             ▼
           Feedback Storage (JSONL + PostgreSQL)
                             │
                             ▼
             Controlled Retraining Pipeline
             (Base FER2013 + Validated Feedback)
                             │
                             ▼
                 Candidate Model Evaluation
         (Test Accuracy, Macro F1, Per-Class F1 Deltas)
                             │
                             ▼
               Automated Promotion Gate
         (Macro F1 delta > 0 & Accuracy delta > 0)
              ├── PASS ──> Promoted to Production (Backup Created)
              └── FAIL ──> Candidate Rejected (Rollback Active)
```

---

## Full-Stack Architecture

```text
FaceSense-AI
├── Frontend (React 19 + Vite)
│   ├── Webcam & Canvas Controller (0.85 JPEG compression, multi-face tracking)
│   ├── Route-based Code Splitting (React.lazy / Suspense)
│   ├── Theme Management (Dynamic OS and explicit dark/light mode)
│   └── Independent Multi-Face Feedback Forms
│
├── Backend (FastAPI REST API)
│   ├── In-Memory Sliding-Window Rate Limiting (Eviction at 10,000 keys)
│   ├── Defense-in-Depth Middleware (Strict CORS, OWASP security headers)
│   ├── JWT Authentication & Argon2id Password Hashing
│   └── Prediction & Feedback Service Orchestrators
│
├── ML Core (PyTorch Modules)
│   ├── ResidualEmotionCNN Architecture (skip connections, batch normalization, dropout)
│   ├── Multi-face Batched Inference Engine
│   ├── FaceDetector (Haar Cascade with dynamic downscaling & CLAHE)
│   └── Data Loaders with Configurable Augmentations
│
└── Database & Storage
    ├── PostgreSQL with SQLAlchemy Connection Pooling (5 pool, 10 overflow)
    ├── JSONL Structured Audit Logging (`outputs/feedback/`)
    └── Checkpoint Manager with Versioned Metadata (`ml/models/checkpoints/`)
```

---

## Key Engineering Features

- **Lightweight Residual CNN:** 4-stage residual architecture with convolutional skip connections, batch normalization, and dropout to mitigate degradation while maintaining fast CPU inference.
- **Confidence Thresholding & `Uncertain` State:** Predictions below confidence threshold (default: 0.50) are routed to an explicit `Uncertain` status rather than returning misleading low-confidence classifications.
- **Multi-Face Batched Inference:** Detected face bounding boxes are cropped and batched into a single PyTorch tensor forward pass (`predict_batch`) rather than sequential single-image evaluations.
- **Detector Input Downscaling:** High-resolution frames are proportionally scaled down to a maximum dimension of 640px before cascade evaluation, reducing detection latency while preserving native crop coordinates.
- **Webcam Payload Optimization:** Client canvas captures utilize JPEG quality 0.85 for frame compression, achieving a >60% payload size reduction on 720p streams compared to default Q=0.95 with negligible visual impact on 48×48 crops.
- **Frontend Code Splitting:** Secondary routes (`DashboardPage`, `HistoryPage`, `InsightsPage`, `SettingsPage`, `AdminPage`) are lazily loaded with `React.lazy` and accessible `Suspense` fallbacks to minimize initial bundle size.
- **Defense-in-Depth Security:** 
  - Password hashing via **Argon2id**.
  - One-time **admin bootstrap** key disabled permanently once an administrator exists.
  - Zero-trust startup: server refuses to boot if `JWT_SECRET_KEY` is missing or uses default placeholders.
  - Strict CORS validation: wildcards (`*`) are disallowed whenever credentials are enabled.
  - OWASP security headers injected via custom Starlette middleware (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`, `Strict-Transport-Security`).
- **Safe Promotion Gate:** Retrained candidate models are benchmarked against the active production model across the full test set. Promotion requires positive net deltas on Macro F1 and Accuracy; previous production checkpoints are backed up automatically.

---

## Model Evaluation & Results

The current production model (`ResidualEmotionCNN-Candidate-epoch39`) was evaluated on the FER2013 test set (7,178 samples) and compared directly against the baseline model (`BaselineEmotionCNN`, 25 epochs).

### Baseline vs. Production Model Comparison

| Evaluation Metric | Baseline Model | Current Production Model | Net Improvement | Gate Status |
|---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | 55.63% | **62.84%** | **+7.22%** | **PASSED** |
| **Test Macro F1** | 50.58% | **61.43%** | **+10.85%** | **PASSED** |
| **Test Weighted F1** | 53.69% | **62.25%** | **+8.56%** | **PASSED** |
| **Test Loss** | 1.1653 | **1.0020** | **-0.1633** | **PASSED** |

*All metrics verified from `ml/models/checkpoints/final/model_metadata.json` and `notebooks/02_model_training.ipynb`.*

### Per-Class Performance Breakdown (Test Set)

| Expression Class | Baseline F1 | Production Candidate F1 | Absolute Delta |
|---|:---:|:---:|:---:|
| **Angry** | 46.36% | 55.49% | +9.14% |
| **Disgust** | 44.13% | 67.54% | +23.42% |
| **Fear** | 17.29% | 37.43% | +20.14% |
| **Happy** | 79.22% | 84.18% | +4.96% |
| **Neutral** | 54.05% | 60.25% | +6.20% |
| **Sad** | 41.56% | 48.75% | +7.19% |
| **Surprise** | 71.47% | 76.35% | +4.89% |

- **Strongest Class:** `Happy` (84.18% F1) due to distinct visual geometric features (smiling mouth/eyes) and higher sample support.
- **Weakest Class:** `Fear` (37.43% F1), reflecting visual ambiguity with `Surprise` and `Sad` in low-resolution 48×48 grayscale patches.
- **Training Progression:** Best validation Macro F1 was achieved at **Epoch 39** (61.16% validation Macro F1, 40 total planned epochs, AdamW optimizer with cosine annealing).

---

## Performance Optimizations & Benchmarks

Benchmarking using `scripts/benchmark_v2.py` on real FER2013 data (batch size: 64, CPU 8 threads) measured execution overheads and training step efficiency:

| Profiling Metric | Baseline CNN | Residual CNN (V2) | Architectural Overhead |
|---|:---:|:---:|:---:|
| **Parameter Count** | 1,180,615 | 1,223,751 | +3.7% |
| **Avg Forward Pass** | 38.0 ms | 41.2 ms | +8.4% |
| **Avg Backward Pass** | 85.0 ms | 92.4 ms | +8.7% |
| **Avg Optimizer Step** | 10.0 ms | 10.5 ms | +5.0% |
| **Avg Total Batch Step** | 133.0 ms | 144.1 ms | +8.3% |
| **Estimated Epoch Duration** | ~4.5 min | ~5.1 min | Safe for CPU training |

### Key Optimizations Implemented:
1. **Residual Blocks over Depth Alone:** Gained +10.85% Macro F1 with only +3.7% parameter overhead and ~8% step latency increase.
2. **CLAHE Preprocessing:** Normalizes lighting gradients on webcam captures, reducing false positive detections from bright backgrounds.
3. **JPEG Quality Calibration:** Using Q=0.85 in `CameraCapture.jsx` reduced upload network transfer payloads by >60% compared to uncalibrated Q=0.95.

---

## Security & Production Hardening

- **JWT Authentication:** Stateful token issuance with configurable expiration (`ACCESS_TOKEN_EXPIRE_MINUTES`) and HMAC-SHA256 signing.
- **Argon2id Passwords:** Password hashing implemented via `argon2-cffi` with salt generation and timing-attack-safe verification.
- **Admin Bootstrap Control:** First admin creation is guarded by `ADMIN_BOOTSTRAP_KEY`. Once any user exists in the database, the bootstrap endpoint permanently returns HTTP 400.
- **Sliding-Window Rate Limiting:** Enforces independent per-IP limits (`/predict`: 30/min, `/feedback`: 60/min, `/auth`: 10/min) with bounded cache memory eviction (`RATE_LIMIT_MAX_KEYS=10000`).
- **File Upload Verification:** Maximum 10MB payload size enforcement, MIME-type whitelisting (`image/jpeg`, `image/png`, `image/webp`), and raw byte header inspection via OpenCV decoding.
- **Strict CORS & Headers:** Prevents credential leakage by rejecting wildcard origins when credentials are enabled, combined with anti-sniff and frame denial security headers.

---

## Project Structure

```text
FaceSense-AI/
├── configs/
│   ├── config.yaml                    # Base training & dataloader configuration
│   └── config_10epoch_cosine.yaml      # Fast experiment configuration
│
├── backend/                           # Production FastAPI REST backend
│   ├── alembic/                       # PostgreSQL database migration scripts
│   ├── app/
│   │   ├── api/v1/endpoints/          # Auth, prediction, feedback, model endpoints
│   │   ├── core/                      # Config, security, rate limiting, dependencies
│   │   ├── db/                        # SQLAlchemy session and engine management
│   │   ├── models/                    # User and Feedback ORM models
│   │   ├── schemas/                   # Pydantic v2 validation schemas
│   │   ├── services/                  # Business logic (prediction, feedback, auth)
│   │   └── main.py                    # FastAPI application and middleware
│   └── tests/                         # 134 passing API & security unit tests
│
├── frontend/                          # React 19 + Vite dashboard
│   ├── src/
│   │   ├── components/                # Modular UI & CameraCapture with Canvas overlay
│   │   ├── contexts/                  # AuthContext and ThemeContext
│   │   ├── pages/                     # Analyze, Dashboard, Admin, Insights, Settings
│   │   └── routes/                    # AppRoutes with lazy-loading and AdminRoute guard
│   └── tests/                         # 29 passing Node.js component & logic tests
│
├── ml/                                # Machine learning core
│   ├── data/                          # Dataset loaders & augmentation pipelines
│   ├── detection/                     # FaceDetector (OpenCV Haar + CLAHE + downscaling)
│   ├── evaluation/                    # Metrics calculation, confusion matrices, reports
│   ├── feedback/                      # Feedback dataset builders & validation logic
│   ├── inference/                     # EmotionPredictor & standalone CLI runners
│   ├── models/                        # BaselineEmotionCNN & ResidualEmotionCNN
│   └── training/                      # Trainer, Retrainer, ModelComparator, benchmark
│
├── notebooks/                         # Interactive research & analysis
│   ├── 01_data_pipeline_exploration.ipynb
│   ├── 02_model_training.ipynb        # Primary experiment & model comparison workflow
│   └── 03_baseline_training_analysis.ipynb
│
├── outputs/                           # Experiment artifacts, checkpoints & feedback
├── scripts/                           # benchmark_v2.py profiling runner
├── tests/                             # 96 passing ML pipeline & model unit tests
├── .env.example                       # Environment configuration template
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- PostgreSQL (optional; in-memory/file fallbacks supported for standalone evaluation)

### 1. Repository Setup & Environment
```powershell
# Clone the repository
git clone https://github.com/MohameedAlaa/FaceSense-AI.git
cd FaceSense-AI

# Activate the project virtual environment (pre-configured with all PyTorch & FastAPI dependencies)
.\.venv\Scripts\Activate.ps1

# Configure environment variables
Copy-Item .env.example .env
```

Generate a secure secret key and update `JWT_SECRET_KEY` in `.env`:
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Launch Backend API Server
```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Documentation: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### 3. Launch Frontend Application
```powershell
cd frontend
npm install
npm run dev
```
- Web Application: `http://localhost:5173`

### 4. Standalone CLI Inference
```powershell
# Predict single image
.\.venv\Scripts\python.exe ml/inference/predict.py --image "path/to/image.jpg"

# Run standalone OpenCV webcam stream
.\.venv\Scripts\python.exe ml/inference/webcam.py
# Controls: 'Q' or 'ESC' to exit, 'S' to save snapshot
```

---

## Automated Testing Suite

The repository contains automated unit and integration tests across three distinct test suites:

```powershell
# 1. Machine Learning & Detection Tests (96 tests)
.\.venv\Scripts\python.exe -m pytest tests/ -v

# 2. Backend API, Security & Rate Limiting Tests (134 tests)
.\.venv\Scripts\python.exe -m pytest backend/tests/ -v

# 3. Frontend Architecture, Logic & Payload Tests (29 tests)
cd frontend
node --test tests/*.test.mjs
```

**Current Repository Test Status:**
- **ML & Pipeline Suite:** 96 / 96 passed (100%)
- **Backend & Security Suite:** 134 / 134 passed (100%)
- **Frontend Suite:** 29 / 29 passed (100%)
- **Total:** **259 automated tests passing**

---

## Known Limitations

1. **Dataset Bias & Imbalance:** The model is trained on FER2013, which contains non-uniform class distribution (`Happy` is over-represented, while `Disgust` has only 111 test samples).
2. **Subtle vs. Posed Expressions:** FER2013 predominantly captures pronounced, stereotypical facial expressions. Accuracy on subtle, natural micro-expressions is lower.
3. **Haar Cascade Detector Sensitivity:** The default OpenCV Haar Cascade is sensitive to extreme profile angles, head tilts, and harsh directional lighting compared to larger deep-learning detectors (e.g., RetinaFace).
4. **Human-in-the-Loop Feedback:** The feedback retraining pipeline is intentionally semi-automated; submitted samples require administrative validation before dataset incorporation to prevent label poisoning.
5. **CPU Inference Latency:** While lightweight, multi-face webcam detection and inference latency depends on the host CPU thread count and resolution.

---

## Roadmap

- [ ] **Modern Face Detector Integration:** Add an optional lightweight ONNX-based detector (e.g., Ultra-Light-Fast-Generic-Face-Detector-1MB) alongside Haar Cascades for improved angled face recall.
- [ ] **Class Re-weighting / Focal Loss:** Implement focal loss in the retraining pipeline to further boost minority class recall (`Fear` and `Disgust`).
- [ ] **Docker Containerization:** Add multi-stage `Dockerfile` and `docker-compose.yml` for unified one-command deployment of API, PostgreSQL, and Frontend.
- [ ] **Model Export (ONNX / TorchScript):** Export `ResidualEmotionCNN` to ONNX format for client-side or edge-accelerated runtime inference.
- [ ] **Automated CI/CD Workflows:** Add GitHub Actions workflows to execute the 259-test suite on pull requests.

---

## Visual Showcase & Demo

```text
[ Web Dashboard Interface Placeholder ]
- Live webcam viewport with SVG bounding boxes and real-time expression tags
- Confidence probability breakdown for each detected face
- Independent human-in-the-loop feedback submission modal

[ Model Evaluation Visuals ]
- Confusion matrix comparing baseline vs. candidate epoch 39
- Training and validation Macro F1 progression curves
```

---

## Author & Portfolio Context

Developed by **Mohamed Alaa** as a comprehensive computer vision and machine learning engineering project demonstrating end-to-end model development, API design, security engineering, and responsive frontend integration.
