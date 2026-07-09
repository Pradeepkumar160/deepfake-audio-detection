<div align="center">

# 🎙️ Deepfake Audio Detection System        

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![WebSocket](https://img.shields.io/badge/WebSocket-010101?style=for-the-badge&logo=socketdotio&logoColor=white)](https://websockets.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**Production-ready real-time AI system to detect synthetic and cloned voices using MFCC + deep learning.**

[🚀 Quick Start](#-quick-start) · [🧠 How It Works](#-how-it-works) · [📡 API Reference](#-api-reference) · [🗂️ Project Structure](#️-project-structure)

![Demo](https://img.shields.io/badge/Status-Live%20Demo%20Ready-brightgreen?style=flat-square)
![Model](https://img.shields.io/badge/Model-PyTorch%20DNN-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Detection-Real%20Time-blue?style=flat-square)

</div>

---
   
## 🔍 Overview

A **production-grade deepfake audio detection platform** that identifies AI-generated or cloned voices in real time. Built as part of a cybersecurity + AI engineering portfolio — targeting VoIP fraud prevention, voice authentication, and enterprise voice security.

### ✨ Key Features 

| Feature | Description | 
|---|---|
| 📁 **Upload & Analyze** | Drag-drop any audio file (WAV/MP3/FLAC/OGG/M4A) for instant detection |
| 🎙️ **Live Stream** | Real-time microphone streaming over WebSocket — results every 2 seconds |
| 🤖 **Train Model** | One-click training using your own real/fake audio dataset |
| 📊 **Confidence Score** | Returns label (REAL/FAKE) + confidence % + raw model score |
| ⚡ **REST API** | Full FastAPI backend with Swagger docs at `/docs` |
| 🔌 **WebSocket** | Raw PCM streaming — no FFmpeg dependency, browser-native |

---

## 🧠 How It Works 

```
Microphone / Audio File
        │
        ▼
┌─────────────────────────┐
│  Audio Loading (librosa) │  ← 16,000 Hz sample rate
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Feature Extraction                 │
│  ├── MFCC (40 coefficients)         │  ← Voice timbre & tone
│  └── Mel-Spectrogram (128 bands)    │  ← Frequency patterns
└────────────┬────────────────────────┘
             │  168-dim feature vector
             ▼
┌─────────────────────────────────────┐
│  PyTorch Neural Network             │
│  Linear(168→256) + BatchNorm + ReLU │
│  Linear(256→128) + BatchNorm + ReLU │
│  Linear(128→64)  + BatchNorm + ReLU │
│  Linear(64→1)    + Sigmoid          │
└────────────┬────────────────────────┘
             │  Score: 0.0 → 1.0
             ▼
      REAL (< 0.5) / FAKE (≥ 0.5)
      + Confidence %
```

---

## 🗂️ Project Structure

```
DeepfakeAudioDetection/
│
├── deepfake_audio_detector.py   ← Main application (backend + frontend + AI)
├── generate_samples.py          ← Generate synthetic training data
├── requirements.txt             ← Python dependencies
├── setup.ps1                    ← Windows PowerShell setup script
├── run.ps1                      ← One-click run script
│
├── dataset/                     ← Training data (auto-created)
│   ├── real/                    ← Real voice samples (.wav)
│   └── fake/                    ← AI-generated voice samples (.wav)
│
├── uploads/                     ← Temp storage for uploaded files
├── streams/                     ← Temp storage for streamed chunks
└── deepfake_model.pth           ← Saved model (created after training)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or 3.11
- Windows 10/11 (or Linux/macOS)
- ~2 GB disk space (for PyTorch)

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/Pradeepkumar160/deepfake-audio-detection.git
cd deepfake-audio-detection

# 2. Allow script execution (one-time)
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# 3. Create virtual environment & install dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 4. Generate synthetic training samples
python generate_samples.py

# 5. Start the server
python deepfake_audio_detector.py
```

### Linux / macOS

```bash
git clone https://github.com/Pradeepkumar160/deepfake-audio-detection.git
cd deepfake-audio-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_samples.py
python deepfake_audio_detector.py
```

Then open **http://localhost:8000** in your browser.

---

## 🖥️ Using the Application

### Step 1 — Train the Model
- Navigate to the **Train Model** tab
- Click **Train Model**
- Wait ~30 seconds for the ✅ success message

### Step 2 — Analyze an Audio File
- Go to **Upload & Analyze**
- Drag & drop any `.wav`, `.mp3`, or `.flac` file
- View the detection result instantly

### Step 3 — Live Microphone Detection
- Go to **Live Stream**
- Click **Start Streaming** (allow microphone access)
- Speak — results appear every 2 seconds below the waveform

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard |
| `GET` | `/api/health` | Health check + model status |
| `POST` | `/api/analyze` | Upload & analyze audio file |
| `POST` | `/api/train` | Train the model |
| `WS` | `/ws/audio` | Real-time PCM audio streaming |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: Analyze via curl
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@your_audio.wav"
```

### Example Response
```json
{
  "filename": "your_audio.wav",
  "label": "FAKE",
  "confidence": 87.43,
  "raw_score": 0.8743,
  "timestamp": "2026-05-25T13:49:08.918Z"
}
```

---

## 📦 Dependencies

```
fastapi          — Async web framework
uvicorn          — ASGI server
python-multipart — File upload support
websockets       — WebSocket protocol
numpy            — Numerical computing
torch            — Deep learning (PyTorch)
torchaudio       — Audio processing
librosa          — MFCC & spectrogram extraction
soundfile        — Audio I/O
scikit-learn     — StandardScaler & train/test split
pydantic         — Data validation
```

---

## 🔬 Adding Real Training Data

Replace synthetic samples with real-world datasets for production accuracy:

**Fake Voice Datasets:**
- [ASVspoof 2021](https://www.asvspoof.org/) — industry standard anti-spoofing
- [WaveFake](https://github.com/RUB-SysSec/WaveFake) — GAN-generated speech
- [FakeAVCeleb](https://github.com/hasam0730/FakeAVCeleb) — deepfake AV

**Real Voice Datasets:**
- [LibriSpeech](https://www.openslr.org/12) — 1000h of English audiobooks
- [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — celebrity speech
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) — crowd-sourced

Place files in `dataset/real/` and `dataset/fake/`, then retrain.

---

## 🛣️ Roadmap

- [x] MFCC + Mel-Spectrogram feature extraction
- [x] PyTorch DNN with BatchNorm & Dropout
- [x] REST API (FastAPI)
- [x] WebSocket live streaming (raw PCM)
- [x] One-click training UI
- [ ] CNN + BiLSTM upgrade
- [ ] Wav2Vec2 / HuBERT embeddings
- [ ] GPU inference support
- [ ] Docker deployment
- [ ] Kubernetes autoscaling
- [ ] JWT authentication
- [ ] Real dataset fine-tuning

---

## 🧑‍💻 Author

**Pradeep Kumar**
CS Undergrad | Cybersecurity + AI/ML | Full-Stack Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/07pradeepk)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/Pradeepkumar160)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---


<div align="center">

⭐ **Star this repo if it helped you — it keeps me motivated to build more!** ⭐

</div>
