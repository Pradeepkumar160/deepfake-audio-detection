#!/usr/bin/env python3
"""
DEEPFAKE AUDIO DETECTION SYSTEM - COMPLETE PRODUCTION APPLICATION
Single-file, fully functional - backend + frontend + AI model.

Usage:
    python deepfake_audio_detector.py
Then visit: http://localhost:8000
"""

import os
import struct
import uuid
import shutil
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    APP_NAME    = "Deepfake Audio Detector"
    VERSION     = "1.0.0"
    SAMPLE_RATE = 16000
    MFCC_FEATURES = 40
    N_MELS      = 128
    MODEL_PATH  = "deepfake_model.pth"
    UPLOAD_DIR  = "uploads"
    STREAMS_DIR = "streams"
    DATASET_DIR = "dataset"
    BATCH_SIZE  = 32
    EPOCHS      = 20
    LEARNING_RATE = 0.001
    TEST_SIZE   = 0.2
    DETECTION_THRESHOLD = 0.5

config = Config()

for d in [config.UPLOAD_DIR, config.STREAMS_DIR, config.DATASET_DIR]:
    Path(d).mkdir(exist_ok=True)
    for sub in ["real", "fake"]:
        Path(d, sub).mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# AUDIO PROCESSOR
# ============================================================================
class AudioProcessor:

    @staticmethod
    def load_audio(path: str, sr: int = config.SAMPLE_RATE):
        audio, sr = librosa.load(path, sr=sr)
        return audio, sr

    @staticmethod
    def features_from_array(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config.MFCC_FEATURES)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=config.N_MELS)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_mean = np.mean(spec_db.T, axis=0)
        return np.concatenate([mfcc_mean, spec_mean])

    @staticmethod
    def extract_features(path: str) -> np.ndarray:
        audio, sr = AudioProcessor.load_audio(path)
        return AudioProcessor.features_from_array(audio, sr)

    @staticmethod
    def pcm_bytes_to_array(data: bytes, sr: int = config.SAMPLE_RATE) -> np.ndarray:
        """Convert raw float32 PCM bytes sent from the browser."""
        n_samples = len(data) // 4          # float32 = 4 bytes
        if n_samples == 0:
            raise ValueError("Empty audio chunk")
        audio = np.frombuffer(data, dtype=np.float32).copy()
        # Resample to target SR if browser sent at a different rate
        # (browser sends at 16 kHz already per our JS constraint)
        return audio


# ============================================================================
# MODEL
# ============================================================================
class DeepfakeDetectorModel(nn.Module):
    def __init__(self, input_size: int = 168):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),           nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class ModelManager:
    def __init__(self):
        self.model = DeepfakeDetectorModel()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.scaler = StandardScaler()
        self._fitted = False
        self._load()

    def _load(self):
        if not os.path.exists(config.MODEL_PATH):
            logger.info("No saved model. Using fresh model.")
            return
        try:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                self.model.load_state_dict(ckpt['model_state'])
                if 'scaler_mean' in ckpt:
                    self.scaler.mean_           = ckpt['scaler_mean']
                    self.scaler.scale_          = ckpt['scaler_scale']
                    self.scaler.var_            = self.scaler.scale_ ** 2
                    self.scaler.n_features_in_  = len(self.scaler.mean_)
                    self.scaler.n_samples_seen_ = 100
                    self._fitted = True
            else:
                self.model.load_state_dict(ckpt)
            logger.info("Model loaded.")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")

    def predict(self, features: np.ndarray) -> Dict:
        if self._fitted:
            features = (features - self.scaler.mean_) / (self.scaler.scale_ + 1e-8)
        t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(t).item()
        label = "FAKE" if score > config.DETECTION_THRESHOLD else "REAL"
        confidence = score * 100 if label == "FAKE" else (1 - score) * 100
        return {"label": label, "confidence": round(confidence, 2), "raw_score": round(score, 4)}

    def save(self):
        torch.save({'model_state': self.model.state_dict(),
                    'scaler_mean': self.scaler.mean_,
                    'scaler_scale': self.scaler.scale_}, config.MODEL_PATH)

    def train(self, X_train, y_train):
        self.scaler.fit(X_train)
        self._fitted = True
        Xs = self.scaler.transform(X_train)
        ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                           torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)
        crit = nn.BCELoss()
        opt  = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.model.train()
        for epoch in range(config.EPOCHS):
            loss_sum = 0
            for bx, by in dl:
                opt.zero_grad()
                loss = crit(self.model(bx), by)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
            logger.info(f"Epoch {epoch+1}/{config.EPOCHS} loss={loss_sum/len(dl):.4f}")
        self.model.eval()
        self.save()


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title=config.APP_NAME, version=config.VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model_manager  = ModelManager()
audio_processor = AudioProcessor()


class AnalysisResult(BaseModel):
    filename: str
    label: str
    confidence: float
    raw_score: float
    timestamp: str

class TrainingRequest(BaseModel):
    epochs: int = config.EPOCHS
    batch_size: int = config.BATCH_SIZE
    learning_rate: float = config.LEARNING_RATE


# ============================================================================
# FRONTEND
# ============================================================================
FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepfake Audio Detector</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
     min-height:100vh;display:flex;justify-content:center;
     align-items:flex-start;padding:30px 20px}
.container{background:#fff;border-radius:20px;
           box-shadow:0 20px 60px rgba(0,0,0,.3);
           max-width:860px;width:100%;padding:40px}
.header{text-align:center;margin-bottom:36px}
.header h1{font-size:2.2em;background:linear-gradient(135deg,#667eea,#764ba2);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text;margin-bottom:8px}
.header p{color:#666}
.tabs{display:flex;gap:8px;margin-bottom:28px;border-bottom:2px solid #eee}
.tab-btn{padding:12px 22px;border:none;background:none;cursor:pointer;
         font-size:.95em;color:#666;border-bottom:3px solid transparent;
         margin-bottom:-2px;transition:all .2s}
.tab-btn.active{color:#667eea;border-bottom-color:#667eea;font-weight:600}
.tab-content{display:none}.tab-content.active{display:block}
.upload-area{border:3px dashed #667eea;border-radius:12px;padding:50px 30px;
             text-align:center;cursor:pointer;background:#f8f9ff;transition:all .2s;
             margin-bottom:18px}
.upload-area:hover,.upload-area.over{background:#eef0ff;border-color:#764ba2}
.upload-area .icon{font-size:3em;margin-bottom:12px}
.upload-area p{color:#667eea;font-size:1.05em;margin-bottom:6px}
.upload-area small{color:#999}
#fileInput{display:none}
.btn{display:inline-block;padding:12px 28px;border:none;border-radius:8px;
     font-size:1em;cursor:pointer;font-weight:600;transition:all .2s}
.btn-primary{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(102,126,234,.4)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btn-danger{background:#e53935;color:#fff}.btn-danger:hover{background:#c62828}
.btn-grey{background:#f0f0f0;color:#444}.btn-grey:hover{background:#e0e0e0}
.result-box{border-radius:12px;padding:24px;background:#f8f9ff;
            margin-top:18px;display:none}
.result-box.show{display:block}
.result-row{display:flex;justify-content:space-between;align-items:center;
            padding:10px 0;border-bottom:1px solid #eee}
.result-row:last-child{border-bottom:none}
.result-row .lbl{color:#777;font-weight:600;font-size:.95em}
.result-row .val{color:#333;font-weight:700}
.badge{display:inline-block;padding:6px 18px;border-radius:20px;font-weight:700;font-size:.9em}
.fake{background:#ffebee;color:#c62828}.real{background:#e8f5e9;color:#2e7d32}
.loading{display:none;text-align:center;padding:30px}.loading.show{display:block}
.spinner{border:4px solid #eee;border-top:4px solid #667eea;border-radius:50%;
         width:44px;height:44px;animation:spin .8s linear infinite;margin:0 auto 14px}
@keyframes spin{to{transform:rotate(360deg)}}
.info-box{background:#e3f2fd;border-left:4px solid #2196f3;padding:14px 18px;
          border-radius:4px;margin-bottom:18px;color:#1565c0;font-size:.95em}
.error-box{background:#ffebee;border-left:4px solid #e53935;padding:14px 18px;
           border-radius:4px;margin-top:14px;color:#b71c1c;font-size:.95em}
.success-box{background:#e8f5e9;border-left:4px solid #43a047;padding:14px 18px;
             border-radius:4px;margin-top:14px;color:#1b5e20;font-size:.95em}
.stream-controls{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.stream-status{padding:12px 18px;border-radius:8px;background:#f0f0f0;
               color:#666;margin-bottom:16px;font-weight:600}
.stream-status.connected{background:#e8f5e9;color:#2e7d32}
.stream-status.error{background:#ffebee;color:#c62828}
.waveform-wrap{background:#111;border-radius:10px;padding:16px;margin-bottom:16px}
#waveform{width:100%;height:140px;display:block}
.info-list{margin-left:20px;margin-top:8px;line-height:1.8;color:#444}
h3{margin-top:18px;color:#333}
code{background:#f0f0f0;padding:2px 6px;border-radius:4px;font-size:.9em}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🎙️ Deepfake Audio Detector</h1>
    <p>Advanced AI-powered detection system for synthetic and cloned voices</p>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('upload',this)">📁 Upload &amp; Analyze</button>
    <button class="tab-btn"        onclick="switchTab('stream',this)">🎙️ Live Stream</button>
    <button class="tab-btn"        onclick="switchTab('train',this)">🤖 Train Model</button>
    <button class="tab-btn"        onclick="switchTab('info',this)">ℹ️ Info</button>
  </div>

  <!-- UPLOAD -->
  <div id="upload" class="tab-content active">
    <div class="upload-area" id="dropZone"
         onclick="document.getElementById('fileInput').click()"
         ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
      <div class="icon">📤</div>
      <p>Click or drag an audio file here</p>
      <small>Supported: WAV · MP3 · M4A · OGG · FLAC</small>
    </div>
    <input type="file" id="fileInput" accept="audio/*" onchange="handleFileSelect(event)">
    <div class="loading" id="uploadLoading"><div class="spinner"></div><p>Analyzing audio…</p></div>
    <div class="result-box" id="uploadResult"><div id="resultContent"></div></div>
    <div id="uploadError"></div>
  </div>

  <!-- STREAM -->
  <div id="stream" class="tab-content">
    <div class="info-box">
      <strong>Live Streaming:</strong> Captures mic audio and sends raw PCM to the server every 2 seconds.
      Train the model first for accurate results.
    </div>
    <div class="stream-controls">
      <button class="btn btn-primary" id="streamStartBtn" onclick="startStreaming()">🎙️ Start Streaming</button>
      <button class="btn btn-danger"  id="streamStopBtn"  onclick="stopStreaming()" style="display:none">⏹ Stop</button>
    </div>
    <div class="stream-status" id="streamStatus">Not connected</div>
    <div class="waveform-wrap"><canvas id="waveform"></canvas></div>
    <div class="result-box" id="streamResult">
      <div id="streamResultContent"></div>
    </div>
    <div id="streamError"></div>
  </div>

  <!-- TRAIN -->
  <div id="train" class="tab-content">
    <div class="info-box">
      <strong>Model Training:</strong> Uses files in <code>dataset/real/</code> and <code>dataset/fake/</code>.
      Run <code>python generate_samples.py</code> first if you haven't yet.
    </div>
    <button class="btn btn-primary" id="trainBtn" onclick="trainModel()">🤖 Train Model</button>
    <div class="loading" id="trainLoading"><div class="spinner"></div><p>Training… please wait.</p></div>
    <div id="trainResult"></div>
  </div>

  <!-- INFO -->
  <div id="info" class="tab-content">
    <h3>About</h3>
    <p style="margin-top:10px;color:#444">Production-ready deepfake audio detection using MFCC + deep learning.</p>
    <h3>API Endpoints</h3>
    <ul class="info-list" style="font-family:monospace">
      <li>GET  /api/health</li>
      <li>POST /api/analyze</li>
      <li>POST /api/train</li>
      <li>WS   /ws/audio</li>
      <li>GET  /docs</li>
    </ul>
    <h3>Quick Start</h3>
    <ol class="info-list">
      <li>Run <code>python generate_samples.py</code></li>
      <li>Click <strong>Train Model</strong></li>
      <li>Upload audio or use Live Stream</li>
    </ol>
  </div>
</div>

<script>
// ─── Tab switching ───────────────────────────────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(name).classList.add('active');
  btn.classList.add('active');
}

// ─── Upload & Analyze ────────────────────────────────────────────────────────
function handleDragOver(e) { e.preventDefault(); e.currentTarget.classList.add('over'); }
function handleDragLeave(e){ e.currentTarget.classList.remove('over'); }
function handleDrop(e) {
  e.preventDefault(); e.currentTarget.classList.remove('over');
  if (e.dataTransfer.files[0]) analyzeFile(e.dataTransfer.files[0]);
}
function handleFileSelect(e) { if (e.target.files[0]) analyzeFile(e.target.files[0]); }

async function analyzeFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  document.getElementById('uploadLoading').classList.add('show');
  document.getElementById('uploadResult').classList.remove('show');
  document.getElementById('uploadError').innerHTML = '';
  try {
    const res  = await fetch('/api/analyze', { method:'POST', body:fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    showUploadResult(data);
  } catch(err) {
    document.getElementById('uploadError').innerHTML =
      `<div class="error-box"><strong>Error:</strong> ${err.message}</div>`;
  } finally {
    document.getElementById('uploadLoading').classList.remove('show');
  }
}

function showUploadResult(r) {
  const fake = r.label === 'FAKE';
  document.getElementById('resultContent').innerHTML = `
    <div class="result-row"><span class="lbl">File</span><span class="val">${r.filename}</span></div>
    <div class="result-row">
      <span class="lbl">Detection</span>
      <span class="badge ${fake?'fake':'real'}">${fake?'⚠️ DEEPFAKE':'✅ AUTHENTIC'}</span>
    </div>
    <div class="result-row"><span class="lbl">Confidence</span><span class="val">${r.confidence}%</span></div>
    <div class="result-row"><span class="lbl">Raw Score</span><span class="val">${r.raw_score}</span></div>
    <div class="result-row"><span class="lbl">Time</span><span class="val">${new Date(r.timestamp).toLocaleString()}</span></div>`;
  document.getElementById('uploadResult').classList.add('show');
}

// ─── Live Stream (raw PCM via AudioWorklet) ───────────────────────────────────
let socket = null, audioCtx = null, workletNode = null,
    sourceNode = null, analyserNode = null, animId = null,
    streamActive = false;

const STREAM_SR = 16000;   // must match server SAMPLE_RATE
const CHUNK_SEC = 2;       // send every N seconds
let   pcmBuffer = [];
let   samplesSinceFlush = 0;

async function startStreaming() {
  document.getElementById('streamError').innerHTML = '';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioCtx = new AudioContext({ sampleRate: STREAM_SR });

    // Inline AudioWorklet processor as a Blob URL
    const processorCode = `
      class PCMProcessor extends AudioWorkletProcessor {
        process(inputs) {
          const ch = inputs[0][0];
          if (ch) this.port.postMessage(ch.slice());
          return true;
        }
      }
      registerProcessor('pcm-processor', PCMProcessor);
    `;
    const blob   = new Blob([processorCode], { type: 'application/javascript' });
    const blobURL = URL.createObjectURL(blob);
    await audioCtx.audioWorklet.addModule(blobURL);

    sourceNode  = audioCtx.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioCtx, 'pcm-processor');
    analyserNode = audioCtx.createAnalyser();
    sourceNode.connect(workletNode);
    sourceNode.connect(analyserNode);
    workletNode.connect(audioCtx.destination);

    socket = new WebSocket('ws://' + window.location.host + '/ws/audio');
    socket.binaryType = 'arraybuffer';

    socket.onopen = () => {
      setStatus('🟢 Connected — streaming every 2 seconds', true);
      document.getElementById('streamStartBtn').style.display = 'none';
      document.getElementById('streamStopBtn').style.display  = 'inline-block';
      streamActive = true;
    };

    socket.onmessage = e => {
      try {
        const d = JSON.parse(e.data);
        if (d.status === 'success') showStreamResult(d.result);
        else if (d.status === 'error')
          document.getElementById('streamError').innerHTML =
            `<div class="error-box"><strong>Server:</strong> ${d.message}</div>`;
      } catch {}
    };

    socket.onerror = () => setStatus('❌ WebSocket error', false);
    socket.onclose = () => { setStatus('Disconnected', false); streamActive = false; };

    // Collect PCM samples and flush every CHUNK_SEC seconds
    const samplesPerChunk = STREAM_SR * CHUNK_SEC;
    workletNode.port.onmessage = e => {
      if (!streamActive) return;
      const chunk = e.data;            // Float32Array, 128 samples
      pcmBuffer.push(...chunk);
      samplesSinceFlush += chunk.length;
      if (samplesSinceFlush >= samplesPerChunk) {
        if (socket && socket.readyState === WebSocket.OPEN) {
          const payload = new Float32Array(pcmBuffer.splice(0, samplesPerChunk));
          socket.send(payload.buffer);
          samplesSinceFlush = 0;
        }
      }
    };

    drawWaveform();
  } catch(err) {
    document.getElementById('streamError').innerHTML =
      `<div class="error-box"><strong>Mic Error:</strong> ${err.message}</div>`;
  }
}

function stopStreaming() {
  streamActive = false;
  if (workletNode)  workletNode.disconnect();
  if (sourceNode)   sourceNode.disconnect();
  if (audioCtx)     audioCtx.close();
  if (socket)       socket.close();
  if (animId)       cancelAnimationFrame(animId);
  pcmBuffer = []; samplesSinceFlush = 0;
  setStatus('Disconnected', false);
  document.getElementById('streamStartBtn').style.display = 'inline-block';
  document.getElementById('streamStopBtn').style.display  = 'none';
}

function setStatus(msg, ok) {
  const el = document.getElementById('streamStatus');
  el.textContent = msg;
  el.className = 'stream-status' + (ok ? ' connected' : '');
}

function showStreamResult(r) {
  const fake = r.label === 'FAKE';
  document.getElementById('streamResultContent').innerHTML = `
    <div class="result-row">
      <span class="lbl">Detection</span>
      <span class="badge ${fake?'fake':'real'}">${fake?'⚠️ DEEPFAKE':'✅ AUTHENTIC'}</span>
    </div>
    <div class="result-row"><span class="lbl">Confidence</span><span class="val">${r.confidence}%</span></div>
    <div class="result-row"><span class="lbl">Score</span><span class="val">${r.raw_score}</span></div>
    <div class="result-row"><span class="lbl">Time</span><span class="val">${new Date().toLocaleTimeString()}</span></div>`;
  document.getElementById('streamResult').classList.add('show');
}

function drawWaveform() {
  if (!analyserNode) return;
  const canvas = document.getElementById('waveform');
  const ctx    = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth  || 780;
  canvas.height = canvas.offsetHeight || 140;
  const buf = new Uint8Array(analyserNode.frequencyBinCount);
  function frame() {
    animId = requestAnimationFrame(frame);
    analyserNode.getByteFrequencyData(buf);
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const w = canvas.width / buf.length;
    buf.forEach((v, i) => {
      const h   = (v / 255) * canvas.height;
      const hue = 240 + (v / 255) * 60;
      ctx.fillStyle = `hsl(${hue},80%,55%)`;
      ctx.fillRect(i * w, canvas.height - h, w - 1, h);
    });
  }
  frame();
}

// ─── Train Model ─────────────────────────────────────────────────────────────
async function trainModel() {
  document.getElementById('trainLoading').classList.add('show');
  document.getElementById('trainResult').innerHTML = '';
  document.getElementById('trainBtn').disabled = true;
  try {
    const res  = await fetch('/api/train', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ epochs:20, batch_size:32, learning_rate:0.001 })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    document.getElementById('trainResult').innerHTML = `
      <div class="success-box">
        <strong>✅ Training Completed!</strong><br>
        Samples: ${data.samples_used} &nbsp;|&nbsp;
        Train: ${data.train_samples} &nbsp;|&nbsp;
        Val: ${data.val_samples}
      </div>`;
  } catch(err) {
    document.getElementById('trainResult').innerHTML =
      `<div class="error-box"><strong>Training Error:</strong> ${err.message}</div>`;
  } finally {
    document.getElementById('trainLoading').classList.remove('show');
    document.getElementById('trainBtn').disabled = false;
  }
}
</script>
</body>
</html>"""


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return FRONTEND_HTML

@app.get("/api/health")
async def health():
    return {"status":"healthy","model_trained": model_manager._fitted}

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")
    if not file.filename.lower().endswith(('.wav','.mp3','.m4a','.ogg','.flac')):
        raise HTTPException(400, "Unsupported format")
    path = os.path.join(config.UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(path,"wb") as f:
            shutil.copyfileobj(file.file, f)
        feats  = audio_processor.extract_features(path)
        result = model_manager.predict(feats)
        return AnalysisResult(filename=file.filename, **result,
                              timestamp=datetime.now().isoformat())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))
    finally:
        try: os.remove(path)
        except: pass

@app.post("/api/train")
async def train(req: TrainingRequest):
    X, y = [], []
    for lbl_name, lbl_val in [("real", 0), ("fake", 1)]:
        folder = os.path.join(config.DATASET_DIR, lbl_name)
        if not os.path.exists(folder): continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.wav','.mp3','.m4a','.ogg','.flac')): continue
            try:
                feats = audio_processor.extract_features(os.path.join(folder, fname))
                X.append(feats); y.append(lbl_val)
            except Exception as e:
                logger.warning(f"Skip {fname}: {e}")
    if len(X) < 4:
        raise HTTPException(400, f"Need ≥4 samples, found {len(X)}")
    X = np.array(X); y = np.array(y)
    X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=42)
    model_manager.train(X_tr, y_tr)
    return {"status":"success","samples_used":len(X),
            "train_samples":len(X_tr),"val_samples":len(X_v)}

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected")
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) < 8:
                continue
            try:
                # Browser sends raw float32 PCM
                audio = np.frombuffer(data, dtype=np.float32).copy()
                if audio.size == 0:
                    continue
                # Clip to valid range
                audio = np.clip(audio, -1.0, 1.0)
                feats  = audio_processor.features_from_array(audio, config.SAMPLE_RATE)
                result = model_manager.predict(feats)
                await ws.send_json({"status":"success","result":result,
                                    "timestamp":datetime.now().isoformat()})
            except Exception as e:
                logger.error(f"WS processing error: {e}")
                await ws.send_json({"status":"error","message":str(e)})
    except WebSocketDisconnect:
        logger.info("WS disconnected")
    except Exception as e:
        logger.error(f"WS fatal: {e}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   DEEPFAKE AUDIO DETECTION SYSTEM  v1.0                         ║
║   Dashboard : http://localhost:8000                             ║
║   API Docs  : http://localhost:8000/docs                        ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
