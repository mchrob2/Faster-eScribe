"""
Pi5 UDP Audio Receiver + Transcriber
======================================
Receives u-law compressed audio from the Pico W over UDP,
decodes it into the same ring-buffer/slicer/transcriber
pipeline as transcribe_pi_2.py.

Setup:
    pip install faster-whisper numpy python-dotenv
    Connect Pi5 to WiFi:  SSID=LectureAudio / Password=transcribe123
    Then run: python3 transcribe_udp.py
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

PICO_W_IP   = "192.168.4.1"    # Pico W AP gateway (fixed)
UDP_PORT    = 5005
FS          = 16000             # Hz — must match Pico W
CHUNK_SECONDS   = 2.5           # how often we transcribe
OVERLAP_SECONDS = 0.5
MODEL_SIZE  = "tiny.en"         # tiny / base / small — swap to base.en for accuracy
DEVICE      = "cpu"
COMPUTE_TYPE = "int8"

LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/{dt_str}.txt"

# ─── U-law decode table (8-bit ulaw → int16 PCM) ──────────────────────────────

def _build_ulaw_decode_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u    = ~i & 0xFF
        sign = u & 0x80
        exp  = (u >> 4) & 0x07
        mant = u & 0x0F
        val  = ((mant << 3) + 0x84) << max(exp - 1, 0)
        if exp == 0:
            val = (mant << 3) + 0x08
        val -= 0x84
        table[i] = -val if sign else val
    return table

_ULAW_DECODE_TABLE = _build_ulaw_decode_table()

def decode_ulaw(data: bytes) -> np.ndarray:
    """Decode raw u-law bytes → int16 numpy array."""
    return _ULAW_DECODE_TABLE[np.frombuffer(data, dtype=np.uint8)]

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

# ─── UDP receive thread ──────────────────────────────────────────────

def udp_receive_loop():
    """
    Opens a UDP socket, announces ourselves to the Pico W with HELLO,
    then feeds decoded audio into the ring buffer.
    Also tracks dropped packets and fills gaps with silence.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)

    # Tell the Pico W our IP so it starts streaming to us
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
                # Re-announce in case the Pico W restarted
                try:
                    hello_sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    hello_sock2.sendto(b"HELLO", (PICO_W_IP, UDP_PORT))
                    hello_sock2.close()
                except OSError:
                    pass
                continue

            if len(data) < 5:
                continue

            # Header: seq (uint16) + sample_rate (uint16)
            seq, _ = struct.unpack_from(">HH", data, 0)
            payload = data[4:]
            payload_samples = len(payload)

            # Fill gaps with silence to keep ring-buffer timing correct
            if last_seq is not None:
                gap = (seq - last_seq - 1) & 0xFFFF
                if gap:
                    dropped  += gap
                    silence   = np.zeros(gap * payload_samples, dtype=np.float32)
                    _write_to_ring(silence)

            last_seq  = seq
            received += 1

            # Decode u-law → int16 → float32 normalised [-1, 1]
            samples_f32 = decode_ulaw(payload).astype(np.float32) / 32768.0
            _write_to_ring(samples_f32)

            # Periodic stats every ~5 seconds worth of packets
            packets_per_stat = int(5 * FS / payload_samples)
            if received % packets_per_stat == 0:
                pct = 100.0 * dropped / max(received + dropped, 1)
                print(f"[UDP] recv={received}  dropped={dropped} ({pct:.1f}%)")
                dropped  = 0
                received = 0
    finally:
        sock.close()

# ─── Audio slicer thread ────────────────────────────────────────────────────────────────

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

# ─── Transcription thread ───────────────────────────────────────────────────────────────

def transcribe_loop():
    global_time = 0.0

    with open(LOG_FILE, "a") as log:
        while not stop_event.is_set():
            try:
                audio = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

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