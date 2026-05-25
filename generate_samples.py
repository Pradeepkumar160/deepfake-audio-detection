"""
Generate synthetic training samples for the Deepfake Audio Detector.
Run this once before training: python generate_samples.py
"""
import os
import numpy as np
import soundfile as sf
from pathlib import Path

def generate_sample(path, duration=2.0, sr=16000, freq=440.0, label="real"):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if label == "real":
        # Natural-sounding sine with harmonics and gentle vibrato
        vibrato = 1 + 0.01 * np.sin(2 * np.pi * 5 * t)
        audio = (0.5 * np.sin(2 * np.pi * freq * vibrato * t)
                 + 0.2 * np.sin(2 * np.pi * freq * 2 * vibrato * t)
                 + 0.1 * np.sin(2 * np.pi * freq * 3 * vibrato * t)
                 + 0.02 * np.random.normal(0, 1, len(t)))
    else:
        # Synthetic / robot-like: heavy FM modulation + noise
        modulator = np.sin(2 * np.pi * 5 * t)
        carrier_freq = freq + 80 * modulator
        audio = (0.4 * np.sin(2 * np.pi * carrier_freq * t)
                 + 0.3 * np.random.normal(0, 1, len(t))
                 + 0.15 * np.sign(np.sin(2 * np.pi * freq * t)))

    # Normalize to [-0.9, 0.9]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    sf.write(path, audio.astype(np.float32), sr)

def main():
    os.makedirs("dataset/real", exist_ok=True)
    os.makedirs("dataset/fake", exist_ok=True)

    n = 15  # samples per class
    print(f"Generating {n} real samples...")
    for i in range(n):
        path = f"dataset/real/sample_{i:02d}.wav"
        generate_sample(path, freq=220 + i * 20, label="real")
        print(f"  Wrote {path}")

    print(f"Generating {n} fake samples...")
    for i in range(n):
        path = f"dataset/fake/sample_{i:02d}.wav"
        generate_sample(path, freq=220 + i * 20, label="fake")
        print(f"  Wrote {path}")

    print(f"\nDone! Generated {n*2} audio files.")
    print("Now run: python deepfake_audio_detector.py")
    print("Then click 'Train Model' in the browser at http://localhost:8000")

if __name__ == "__main__":
    main()
