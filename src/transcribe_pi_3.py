# slow but accurate

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import os
import queue
import threading
import time
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import datetime

# Date / Time
dt = datetime.now(ZoneInfo("America/Chicago"))
dt_str = f"{dt:%Y-%m-%d_%H-%M-%S}"

# Environment / HF token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Configuration (Pi 5 tuned)
FS = 16000
CHUNK_SECONDS = 3.0        # how often we transcribe
MODEL_SIZE = "small.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
USB_MIC_NAME = "USB Audio"
SILENCE_THRESHOLD = 0.003

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/{dt_str}.txt"

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
        print("USB mic not found. Plug it in...")
        time.sleep(2)

print(f"Using mic {mic_index}: {sd.query_devices(mic_index)['name']}")

# Load Whisper model
print("Loading Whisper model...")
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=4,
    num_workers=1
)
print("Model loaded.\n")

# Ring buffer for audio
buffer_len = int((CHUNK_SECONDS + OVERLAP_SECONDS) * FS)
audio_buffer = np.zeros(buffer_len, dtype=np.float32)
buffer_lock = threading.Lock()
write_pos = 0

stop_event = threading.Event()
audio_queue = queue.Queue(maxsize=2)


# Audio callback (non-blocking)
def audio_callback(indata, frames, time_info, status):
    global write_pos
    if status:
        print(status)

    samples = indata[:, 0]
    n = len(samples)

    with buffer_lock:
        end = write_pos + n
        if end < buffer_len:
            audio_buffer[write_pos:end] = samples
        else:
            first = buffer_len - write_pos
            audio_buffer[write_pos:] = samples[:first]
            audio_buffer[:n-first] = samples[first:]
        write_pos = (write_pos + n) % buffer_len

# Audio slicer thread
def slicer_loop():
    last_read = 0
    step = int(CHUNK_SECONDS * FS)

    while not stop_event.is_set():
        time.sleep(CHUNK_SECONDS)

        with buffer_lock:
            start = (write_pos - step) % buffer_len
            end = write_pos

            if start < end:
                chunk = audio_buffer[start:end].copy()
            else:
                chunk = np.concatenate((audio_buffer[start:], audio_buffer[:end])).copy()

        try:
            audio_queue.put(chunk, timeout=0.1)
        except queue.Full:
            # Drop oldest chunk to prevent latency buildup
            try:
                audio_queue.get_nowait()
                audio_queue.put_nowait(chunk)
            except queue.Empty:
                pass

# Transcription thread
def transcribe_loop():
    start_time = time.monotonic()

    with open(LOG_FILE, "a") as log:
        while not stop_event.is_set():
            try:
                audio = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            current_offset = time.monotonic() - start_time

            if np.sqrt(np.mean(audio**2)) < SILENCE_THRESHOLD:
                audio_queue.task_done()
                continue

            segments, info = model.transcribe(
                audio,
                beam_size=1,
                best_of=1,
                temperature=0,
                language="en",
                vad_filter=True,
                vad_parameters = {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500
                },
            )

            for seg in segments:
                start = seg.start + current_offset - CHUNK_SECONDS
                end = seg.end + current_offset - CHUNK_SECONDS
                text = seg.text.strip()

                if not text:
                    continue

                line = f"[{start:6.2f}s → {end:6.2f}s] {text}"
                print(line)
                log.write(line + "\n")
                log.flush()

            audio_queue.task_done()


# Start everything
print("Starting live transcription (Ctrl+C to stop)\n")

stream = sd.InputStream(
    samplerate=FS,
    channels=1,
    dtype="float32",
    device=mic_index,
    callback=audio_callback
)

with stream:
    slicer_thread = threading.Thread(target=slicer_loop, daemon=True)
    trans_thread = threading.Thread(target=transcribe_loop, daemon=True)

    slicer_thread.start()
    trans_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        slicer_thread.join()
        trans_thread.join()

print("Goodbye")
