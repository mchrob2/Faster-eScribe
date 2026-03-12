import numpy as np
from faster_whisper import WhisperModel
import os
import queue
import threading
import time
import serial
from scipy.signal import resample
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import datetime

# Date / Time
dt = datetime.now(ZoneInfo("America/Chicago"))
dt_str = f"{dt:%Y-%m-%d_%H-%M-%S}"

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Configuration
PICO_SAMPLE_RATE = 2000
WHISPER_SAMPLE_RATE = 16000
CHUNK_SECONDS = 3.0
OVERLAP_SECONDS = 1.0      
MODEL_SIZE = "small.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SERIAL_PORT = "/dev/ttyACM0"  # Check your port (ttyACM0 or ttyUSB0)
SILENCE_THRESHOLD = 0.003

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/{dt_str}.txt"

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=4, num_workers=1)
print("Model loaded.\n")

# Ring buffer for audio
buffer_len = int((CHUNK_SECONDS + OVERLAP_SECONDS) * WHISPER_SAMPLE_RATE)
audio_buffer = np.zeros(buffer_len, dtype=np.float32)
buffer_lock = threading.Lock()
write_pos = 0

stop_event = threading.Event()
audio_queue = queue.Queue(maxsize=2)

# --- NEW: Serial Audio Reader ---
def serial_read_loop():
    global write_pos
    accumulation_buffer = []
    ACCUMULATION_TARGET = PICO_SAMPLE_RATE # Process resampling in 1-second chunks

    try:
        ser = serial.Serial(SERIAL_PORT, 115200)
        print(f"Connected to Pico on {SERIAL_PORT}")
    except Exception as e:
        print(f"Failed to connect to serial port: {e}")
        stop_event.set()
        return

    while not stop_event.is_set():
        try:
            line = ser.readline().strip()
            if not line:
                continue
                
            # Decode Hex to raw bytes -> uint16 array
            audio_bytes = bytes.fromhex(line.decode('ascii'))
            samples_uint16 = np.frombuffer(audio_bytes, dtype=np.uint16)
            
            # Convert 0-65535 to float32 [-1.0, 1.0]
            samples_float = (samples_uint16.astype(np.float32) - 32768.0) / 32768.0
            accumulation_buffer.extend(samples_float)
            
            # Resample when we have enough data to avoid chunk boundary artifacts
            if len(accumulation_buffer) >= ACCUMULATION_TARGET:
                chunk_2k = np.array(accumulation_buffer[:ACCUMULATION_TARGET])
                accumulation_buffer = accumulation_buffer[ACCUMULATION_TARGET:]
                
                # Upsample 2kHz to 16kHz
                chunk_16k = resample(chunk_2k, ACCUMULATION_TARGET * (WHISPER_SAMPLE_RATE // PICO_SAMPLE_RATE))
                
                with buffer_lock:
                    n = len(chunk_16k)
                    end = write_pos + n
                    if end < buffer_len:
                        audio_buffer[write_pos:end] = chunk_16k
                    else:
                        first = buffer_len - write_pos
                        audio_buffer[write_pos:] = chunk_16k[:first]
                        audio_buffer[:n-first] = chunk_16k[first:]
                    write_pos = (write_pos + n) % buffer_len

        except ValueError:
            pass # Ignore corrupted hex lines
        except Exception as e:
            print(f"Serial read error: {e}")

# Audio slicer thread
def slicer_loop():
    step = int(CHUNK_SECONDS * WHISPER_SAMPLE_RATE)

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
                audio, beam_size=1, best_of=1, temperature=0, language="en", vad_filter=True,
                vad_parameters={"threshold": 0.5, "min_speech_duration_ms": 250, "min_silence_duration_ms": 500},
            )

            for seg in segments:
                start = seg.start + current_offset - CHUNK_SECONDS
                end = seg.end + current_offset - CHUNK_SECONDS
                text = seg.text.strip()
                if not text: continue

                line = f"[{start:6.2f}s → {end:6.2f}s] {text}"
                print(line)
                log.write(line + "\n")
                log.flush()

            audio_queue.task_done()

print("Starting live RF transcription (Ctrl+C to stop)\n")

serial_thread = threading.Thread(target=serial_read_loop, daemon=True)
slicer_thread = threading.Thread(target=slicer_loop, daemon=True)
trans_thread = threading.Thread(target=transcribe_loop, daemon=True)

serial_thread.start()
slicer_thread.start()
trans_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")
    stop_event.set()
    serial_thread.join()
    slicer_thread.join()
    trans_thread.join()

print("Goodbye")
