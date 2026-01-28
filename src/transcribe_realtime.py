import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import os
from dotenv import load_dotenv
import time

# --- Load HF token ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# --- Configuration ---
FS = 16000
BUFFER_SECONDS = 12
STEP_SECONDS = 1
MODEL_SIZE = "tiny.en"
DEVICE = "cpu"
THRESHOLD = 0.01
LOG_FILE = "logs/transcripts.txt"

os.makedirs("logs", exist_ok=True)

# --- Load model ---
print("Loading model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")
print("Model loaded. Starting streaming transcription...\n")

# --- Initialize buffer and counters ---
buffer = np.zeros(BUFFER_SECONDS * FS, dtype=np.float32)
last_text_end = 0.0
total_audio_time = 0.0  # total time since start

def is_speech(audio_chunk, threshold=THRESHOLD):
    return np.sqrt(np.mean(audio_chunk**2)) > threshold

try:
    while True:
        # Record STEP_SECONDS of audio
        audio = sd.rec(int(STEP_SECONDS * FS), samplerate=FS, channels=1)
        sd.wait()
        audio = np.squeeze(audio).astype(np.float32)

        # Only process if new audio contains speech
        if not is_speech(audio):
            total_audio_time += STEP_SECONDS
            continue

        # Shift buffer and append new audio
        buffer = np.roll(buffer, -len(audio))
        buffer[-len(audio):] = audio

        # Transcribe buffer
        segments, info = model.transcribe(buffer, beam_size=5, language='en')

        output_lines = []
        for s in segments:
            # Adjust segment times relative to total_audio_time
            seg_start = total_audio_time - BUFFER_SECONDS + s.start
            seg_end = total_audio_time - BUFFER_SECONDS + s.end

            if seg_end <= last_text_end:
                continue  # already printed

            line = f"[{seg_start:.2f}s → {seg_end:.2f}s] {s.text}"
            print(line)
            output_lines.append(line)
            last_text_end = seg_end

        # Save new lines to log
        if output_lines:
            with open(LOG_FILE, "a") as f:
                f.write("\n".join(output_lines) + "\n")

        total_audio_time += STEP_SECONDS
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping streaming transcription. Goodbye!")
