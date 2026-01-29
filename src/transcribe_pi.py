import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import os
import queue
import threading
from dotenv import load_dotenv
import time
from zoneinfo import ZoneInfo
from datetime import datetime

# Date / Time
dt = datetime.now(ZoneInfo("America/Chicago"))
dt = f"{dt:%Y-%m-%d_%H:%M:%S}"
print(dt)

# Environment / HF token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Configuration
FS = 16000                 # Sample rate
DURATION = 3               # Seconds per chunk
OVERLAP = 0.5                # Seconds of overlap
MODEL_SIZE = "tiny.en"
DEVICE = "cpu"
THRESHOLD = 0.01           # RMS silence threshold
LOG_FILE = f"logs/{dt}.txt"
USB_MIC_NAME = "USB Audio"  # part of the name of your USB mic

os.makedirs("logs", exist_ok=True)

# Select USB microphone
def find_usb_mic(name_part=USB_MIC_NAME):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0 and name_part.lower() in dev['name'].lower():
            return i
    return None

mic_index = None
while mic_index is None:
    mic_index = find_usb_mic()
    if mic_index is None:
        print("USB mic not found. Please plug it in...")
        time.sleep(2)
print(f"Using USB mic at index {mic_index}: {sd.query_devices(mic_index)['name']}")

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
audio_queue = queue.Queue(maxsize=2)
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


# Main loop
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping transcription...")
    stop_event.set()

    rec_thread.join()
    tr_thread.join()

    print("Goodbye")
