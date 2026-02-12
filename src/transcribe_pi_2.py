# FAST but innacurate

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
CHUNK_SECONDS = 1.6        # how often we transcribe
OVERLAP_SECONDS = 0.3
MODEL_SIZE = "tiny.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
USB_MIC_NAME = "USB Audio"

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
    compute_type=COMPUTE_TYPE
)
print("Model loaded.\n")

# Ring buffer for audio
buffer_len = int((CHUNK_SECONDS + OVERLAP_SECONDS) * FS)
audio_buffer = np.zeros(buffer_len, dtype=np.float32)
buffer_lock = threading.Lock()
write_pos = 0

stop_event = threading.Event()
audio_queue = queue.Queue(maxsize=3)

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
            start = (write_pos - step - int(OVERLAP_SECONDS * FS)) % buffer_len
            end = write_pos

            if start < end:
                chunk = audio_buffer[start:end].copy()
            else:
                chunk = np.concatenate((audio_buffer[start:], audio_buffer[:end])).copy()

        try:
            audio_queue.put(chunk, timeout=0.5)
        except queue.Full:
            pass

# Transcription thread
def transcribe_loop():
    global_time = 0.0

    with open(LOG_FILE, "a") as log:
        while not stop_event.is_set():
            try:
                audio = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            segments, info = model.transcribe(
                audio,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                language="en"
            )

            for seg in segments:
                start = seg.start + global_time
                end = seg.end + global_time
                text = seg.text.strip()

                if not text:
                    continue

                line = f"[{start:6.2f}s → {end:6.2f}s] {text}"
                print(line)
                log.write(line + "\n")
                log.flush()

            global_time += CHUNK_SECONDS
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
