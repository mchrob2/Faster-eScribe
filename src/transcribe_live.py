import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# --- Configuration ---
DURATION = 5        # seconds per chunk
FS = 16000          # sample rate
DEVICE = "cpu"
MODEL_SIZE = "tiny.en"

# --- Load model once ---
print("Loading model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")
print("Model loaded. Starting continuous transcription.\n")

try:
    while True:
        print(f"Recording {DURATION} seconds of audio...")
        audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
        sd.wait()

        audio = np.squeeze(audio).astype(np.float32)

        print("Transcribing...")
        segments, info = model.transcribe(audio, beam_size=5)

        print("Detected language:", info.language)
        for s in segments:
            print(f"[{s.start:.2f}s → {s.end:.2f}s] {s.text}")
        print("-" * 40)  # separator between chunks

except KeyboardInterrupt:
    print("\nStopping transcription loop. Goodbye!")

