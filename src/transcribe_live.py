import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import os
import queue
import threading
from dotenv import load_dotenv
import time

# Environment / HF token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Configuration
FS = 16000                 # Sample rate
DURATION = 4               # Seconds per chunk
OVERLAP = 1                # Seconds of overlap
MODEL_SIZE = "small.en"
DEVICE = "cpu"
THRESHOLD = 0.01           # RMS silence threshold
LOG_FILE = "logs/transcripts.txt"

os.makedirs("logs", exist_ok=True)

# Load model
print("Loading Whisper model...")
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type="int8"
)
print("Model loaded.\n")

# Helpers
def is_speech(audio, threshold=THRESHOLD):
    return np.sqrt(np.mean(audio ** 2)) > threshold

def merge_overlap(text, prev_words, max_overlap=5):
    words = text.strip().split()
    overlap_len = min(len(words), len(prev_words), max_overlap)

    for i in range(overlap_len, 0, -1):
        if prev_words[-i:] == words[:i]:
            return " ".join(words[i:]), words
    return " ".join(words), words


# Shared state
audio_queue = queue.Queue(maxsize=5)
prev_audio = np.array([], dtype=np.float32)
stop_event = threading.Event()
global_time = 0.0          # running offset for timestamps
last_end_time = 0.0        # deduplication
prev_words = []            # for overlap smoothing

# Audio recording thread
def record_loop():
    global prev_audio
    try:
        while not stop_event.is_set():
            audio = sd.rec(
                int(DURATION * FS),
                samplerate=FS,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            audio = np.squeeze(audio)

            # prepend overlap from previous chunk
            if prev_audio.size > 0:
                audio = np.concatenate([prev_audio, audio])

            # save last OVERLAP seconds
            prev_audio = audio[-int(OVERLAP * FS):]

            if not is_speech(audio):
                continue

            try:
                audio_queue.put(audio, timeout=1)
            except queue.Full:
                pass  # drop if transcriber lags
    finally:
        sd.stop()

# Transcription thread
def transcribe_loop():
    global global_time, last_end_time, prev_words
    with open(LOG_FILE, "a") as log:
        while not stop_event.is_set():
            try:
                audio = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            segments, info = model.transcribe(
                audio,
                beam_size=5,
                vad_filter=False
            )

            for s in segments:
                start = s.start + global_time
                end = s.end + global_time
                text = s.text.strip()

                if end <= last_end_time or not text:
                    continue

                # merge overlapping words with previous chunk
                merged_text, prev_words = merge_overlap(text, prev_words)
                if not merged_text:
                    continue

                line = f"[{start:.2f}s → {end:.2f}s] {merged_text}"
                print(line)
                log.write(line + "\n")
                log.flush()

                last_end_time = end

            # increment running timestamp
            global_time += (DURATION - OVERLAP)

            audio_queue.task_done()

# Start threads
rec_thread = threading.Thread(target=record_loop)
tr_thread = threading.Thread(target=transcribe_loop)

rec_thread.start()
tr_thread.start()

print("Live transcription running (Ctrl+C to stop)\n")

# Main loop / graceful shutdown
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping transcription...")
    stop_event.set()

    rec_thread.join()
    tr_thread.join()

    print("Clean shutdown complete. Goodbye!")
