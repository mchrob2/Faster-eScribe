"""
Pi5 UDP Audio Receiver + Transcriber (Optimized)
================================================
Receives 16-bit unsigned PCM from the Pico W over UDP,
applies vectorized DC‑blocking and noise gating,
then transcribes with faster-whisper.
"""

import socket
import struct
import numpy as np
from faster_whisper import WhisperModel
import os
import queue
import threading
import time
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import datetime

# ─── Date / Time ──────────────────────────────────────────────────────────────

dt = datetime.now(ZoneInfo("America/Chicago"))
dt_str = f"{dt:%Y-%m-%d_%H-%M-%S}"

# ─── Environment / HF token ───────────────────────────────────────────────────

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# ─── Configuration ────────────────────────────────────────────────────────────

PICO_W_IP       = "192.168.4.1" # Pico W AP gateway
UDP_PORT        = 5005
FS              = 16000         # Hz
CHUNK_SECONDS   = 2.5           
OVERLAP_SECONDS = 0.5
MODEL_SIZE      = "base.en"     # Upgraded to base.en for better accuracy on Pi 5
DEVICE          = "cpu"
COMPUTE_TYPE    = "int8"
NOISE_THRESHOLD = 0.03          # Whisper threshold: increase if you still get static hallucinations

LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/{dt_str}.txt"

# ─── Load Whisper model ───────────────────────────────────────────────────────

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded.\n")

# ─── Ring buffer ──────────────────────────────────────────────────────────────

buffer_len   = int((CHUNK_SECONDS + OVERLAP_SECONDS) * FS)
audio_buffer = np.zeros(buffer_len, dtype=np.float32)
buffer_lock  = threading.Lock()
write_pos    = 0

stop_event = threading.Event()
audio_queue = queue.Queue(maxsize=3)

def _write_to_ring(samples_f32: np.ndarray):
    """Write float32 samples into the circular ring buffer (thread-safe)."""
    global write_pos
    n = len(samples_f32)
    with buffer_lock:
        end = write_pos + n
        if end <= buffer_len:
            audio_buffer[write_pos:end] = samples_f32
        else:
            first = buffer_len - write_pos
            audio_buffer[write_pos:] = samples_f32[:first]
            audio_buffer[:n - first] = samples_f32[first:]
        write_pos = (write_pos + n) % buffer_len

# ─── UDP receive thread ───────────────────────────────────────────────────────

def udp_receive_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)

    # Announce
    hello_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hello_sock.sendto(b"HELLO", (PICO_W_IP, UDP_PORT))
    hello_sock.close()
    print(f"Sent HELLO to Pico W at {PICO_W_IP}")
    print(f"Listening for audio on UDP port {UDP_PORT}...\n")

    last_seq = None
    dropped  = 0
    received = 0

    try:
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                try:
                    hello_sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    hello_sock2.sendto(b"HELLO", (PICO_W_IP, UDP_PORT))
                    hello_sock2.close()
                except OSError:
                    pass
                continue

            if len(data) < 5:
                continue

            seq, _ = struct.unpack_from(">HH", data, 0)
            payload = data[4:]
            payload_samples = len(payload) // 2

            if last_seq is not None:
                gap = (seq - last_seq - 1) & 0xFFFF
                if gap:
                    dropped += gap
                    silence = np.zeros(gap * payload_samples, dtype=np.float32)
                    _write_to_ring(silence)

            last_seq = seq
            received += 1

            # Decode unsigned 16-bit PCM (from Pico's analogbufio) → float32
            samples_u16 = np.frombuffer(payload, dtype='<u2')
            
            # Center and scale to -1.0 -> 1.0
            samples_f32 = (samples_u16.astype(np.float32) - 32768.0) / 32768.0

            # Vectorized DC‑blocking (Instantly removes any hardware DC offset)
            samples_f32 -= np.mean(samples_f32)

            _write_to_ring(samples_f32)

            packets_per_stat = int(5 * FS / payload_samples)
            if received % packets_per_stat == 0:
                pct = 100.0 * dropped / max(received + dropped, 1)
                print(f"[UDP] recv={received}  dropped={dropped} ({pct:.1f}%)")
                dropped = 0
                received = 0
    finally:
        sock.close()

# ─── Audio slicer thread ──────────────────────────────────────────────────────

def slicer_loop():
    step = int(CHUNK_SECONDS * FS)
    while not stop_event.is_set():
        time.sleep(CHUNK_SECONDS)
        with buffer_lock:
            start = (write_pos - step - int(OVERLAP_SECONDS * FS)) % buffer_len
            end   = write_pos

            if start < end:
                chunk = audio_buffer[start:end].copy()
            else:
                chunk = np.concatenate((audio_buffer[start:], audio_buffer[:end])).copy()

        try:
            audio_queue.put(chunk, timeout=0.5)
        except queue.Full:
            pass

# ─── Transcription thread ─────────────────────────────────────────────────────

def transcribe_loop():
    global_time = 0.0

    with open(LOG_FILE, "a") as log:
        while not stop_event.is_set():
            try:
                audio = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            max_amp = np.max(np.abs(audio))
            
            # Noise Gate: Only process if someone is actually talking
            if max_amp > NOISE_THRESHOLD:
                audio = audio / max_amp * 0.5
            else:
                audio = np.zeros_like(audio) # Silence it so Whisper ignores it

            segments, _ = model.transcribe(
                audio,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                language="en"
            )

            for seg in segments:
                start = seg.start + global_time
                end   = seg.end   + global_time
                text  = seg.text.strip()

                if not text:
                    continue

                line = f"[{start:6.2f}s → {end:6.2f}s] {text}"
                print(line)
                log.write(line + "\n")
                log.flush()

            global_time += CHUNK_SECONDS
            audio_queue.task_done()

# ─── Start everything ─────────────────────────────────────────────────────────

print("Starting live transcription (Ctrl+C to stop)\n")

udp_thread    = threading.Thread(target=udp_receive_loop, daemon=True)
slicer_thread = threading.Thread(target=slicer_loop,      daemon=True)
trans_thread  = threading.Thread(target=transcribe_loop,  daemon=True)

udp_thread.start()
slicer_thread.start()
trans_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")
    stop_event.set()
    udp_thread.join(timeout=3)
    slicer_thread.join(timeout=3)
    trans_thread.join(timeout=30)

print("Goodbye")
