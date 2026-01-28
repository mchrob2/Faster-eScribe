import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import os
from dotenv import load_dotenv

# --- Load Hugging Face token from .env ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# --- Configuration ---
DURATION = 3        # seconds per chunk
FS = 16000          # sample rate
OVERLAP = 1         # seconds of overlap between chunks
MODEL_SIZE = "tiny.en"
DEVICE = "cpu"
THRESHOLD = 0.01    # silence threshold (RMS)
LOG_FILE = "logs/transcripts.txt"

os.makedirs("logs", exist_ok=True)

# --- Load model ---
print("Loading model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")
print("Model loaded. Starting transcription loop.\n")

# --- Initialize previous audio for overlap ---
prev_audio = np.array([], dtype=np.float32)

def is_speech(audio_chunk, threshold=THRESHOLD):
    """Return True if RMS amplitude exceeds threshold"""
    return np.sqrt(np.mean(audio_chunk**2)) > threshold

try:
    while True:
        print(f"Recording {DURATION}s of audio...")
        audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
        sd.wait()
        audio = np.squeeze(audio).astype(np.float32)

        # prepend previous overlap
        if prev_audio.size > 0:
            audio = np.concatenate([prev_audio, audio])

        if not is_speech(audio):
            print("Silence detected, skipping chunk.")
            # keep last overlap for next chunk
            prev_audio = audio[-int(OVERLAP * FS):]
            continue

        print("Transcribing...")
        segments, info = model.transcribe(audio, beam_size=5)

        output_lines = [f"Detected language: {info.language}"]
        for s in segments:
            line = f"[{s.start:.2f}s → {s.end:.2f}s] {s.text}"
            print(line)
            output_lines.append(line)

        # Save to log file
        with open(LOG_FILE, "a") as f:
            f.write("\n".join(output_lines) + "\n")
        print("-" * 40)

        # Save last OVERLAP seconds for next chunk
        prev_audio = audio[-int(OVERLAP * FS):]

except KeyboardInterrupt:
    print("\nStopping transcription loop. Goodbye!")
