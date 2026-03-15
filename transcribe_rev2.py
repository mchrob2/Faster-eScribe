"""
Pi5 Audio Receiver + Transcriber
==================================
- Connects to Pico W's WiFi hotspot "LectureAudio"
- Receives UDP audio stream
- Decodes u-law -> 16-bit PCM
- Buffers audio into chunks (e.g. 5 seconds)
- Transcribes with faster-whisper
- Prints transcription to terminal (extend to save to file etc.)
    Setup on Pi5:
        pip install faster-whisper numpy
    Connect Pi5 to WiFi:
        SSID: LectureAudio
        Password: transcribe123
    Then run:
        python3 receiver.py
"""

import socket
import struct
import numpy as np
import threading
import queue
import time
from faster_whisper import WhisperModel

# ─── Configuration ────────────────────────────────────────────────────────────

PICO_W_IP = "192.168.4.1"   # Pico W AP gateway IP (fixed)
UDP_PORT = 5005             
SAMPLE_RATE = 16000         # Hz - must match Pico W
BUFFER_SECS = 2.5           # seconds of audio per transcription chunk
WHISPER_MODEL = "base.en"      # tiny / base / small / medium (base good for Pi5)
DEVICE = "cpu"              # Pi5 has no CUDA; use "cpu"
COMPUTE_TYPE = "int8"       # fastest on CPU

# ─── U-law decoding (8-bit ulaw -> 16-bit signed PCM) ─────────────────────────

_ULAW_DECODE_TABLE = None

def _build_ulaw_decode_table():
    global _ULAW_DECODE_TABLE
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u = ~i & 0xFF
        sign = u & 0x80
        exp = (u >> 4) & 0x07
        mant = u & 0x0F
        val = ((mant << 3) + 0x84) << max(exp - 1, 0)
        if exp == 0:
            val = (mant << 3) + 0x08
        val -= 0x84
        table[i] = -val if sign else val
    _ULAW_DECODE_TABLE = table
    
_build_ulaw_decode_table()

def decode_ulaw_bytes(data: bytes) -> np.ndarray:
    """Decode bytes of ulaw samples to int16 numpy array."""
    indices = np.frombuffer(data, dtype=np.uint8)
    return _ULAW_DECODE_TABLE[indices]

# ─── Load Whisper model ───────────────────────────────────────────────────────

print(f"Loading faster-whisper model '{WHISPER_MODEL}'...")
model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded.")

# ─── Audio buffer & transcription queue ──────────────────────────────────────

SAMPLES_PER_CHUNK = SAMPLE_RATE * BUFFER_SECS
audio_buffer = []
transcription_queue = queue.Queue()

def transcription_worker():
    """Runs in a separate thread so transcription doesn't block receiving."""
    while True:
        audio_chunk = transcription_queue.get()
        if audio_chunk is None:
            break
        
        # Convert int16 to float32 normalised [-1.0, 1.0] as whisper expects
        audio_f32 = audio_chunk.astype(np.float32) / 32768.0
        
        t0 = time.time()
        segments, info = model.transcribe(
            audio_f32,
            beam_size=5,
            language="en", 		# set to None for auto-detect
            vad_filter=True, 	# skip silent sections
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        
        print(f"\n[{time.strftime('%H:%M:%S')}] Transcription (lang={info.language}):")
        for segment in segments:
            print(f" [{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text.strip()}")
        print(f" (took {time.time()-t0:.1f}s)", flush=True)
        
        transcription_queue.task_done()
        
t = threading.Thread(target=transcription_worker, daemon=True)
t.start()

# ─── UDP socket setup ─────────────────────────────────────────────────────────

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.settimeout(2.0)

print(f"Listening for audio on UDP port {UDP_PORT}...")
print(f"Make sure Pi5 is connected to WiFi: SSID='LectureAudio' / Password='transcribe123'\n"
      
# ─── Send HELLO to Pico W so it learns our IP ─────────────────────────────────
      
hello_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
hello_sock.sendto(b"HELLO", (PICO_W_IP, UDP_PORT))
hello_sock.close()     
print(f"Sent HELLO to Pico W at {PICO_W_IP}")
      
# ─── Packet tracking ──────────────────────────────────────────────────────────
      
last_seq = None
dropped = 0
received = 0
      
# ─── Main receive loop ────────────────────────────────────────────────────────
      
print("Receiving audio... (Ctrl+C to stop)\n")

try:
    while True:
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:      
            # If we have a partial buffer and silence, flush it
            if len(audio_buffer) > SAMPLE_RATE: # at least 1 second
                chunk = np.concatenate(audio_buffer)
                transcription_queue.put(chunk)
                audio_buffer.clear()
                print("[Timeout] Flushed partial buffer to transcription")
            continue

        if len(data) < 5:
            continue # too short, ignore

        # Parse header: seq (2 bytes) + sample_rate (2 bytes)
        seq, sr = struct.unpack_from(">HH", data, 0)
        payload = data[4:]
      
        # Sequence tracking
        if last_seq is not None:
            gap = (seq - last_seq - 1) & 0xFFFF
            if gap > 0:
                dropped += gap
                # Fill dropped packets with silence to maintain timing
                silence = np.zeros(gap * len(payload), dtype=np.int16)
                audio_buffer.append(silence)
        
        last_seq = seq
        received += 1
        
        # Decode ulaw payload
        samples = decode_ulaw_bytes(payload)
        audio_buffer.append(samples)

        # When we have enough audio, push to transcription queue
        total_samples = sum(len(b) for b in audio_buffer)
        if total_samples >= SAMPLES_PER_CHUNK:
            chunk = np.concatenate(audio_buffer)
            audio_buffer.clear()
            transcription_queue.put(chunk)
            
            pct_dropped = 100.0 * dropped / max(received + dropped, 1)
            print(f"[Buffer full] Queued {BUFFER_SECS}s chunk | "
                f"Received: {received} pkts | Dropped: {dropped} ({pct_dropped:.1f}%)")
            dropped = 0
            received = 0
      
except KeyboardInterrupt:
    print("\nStopping...")
    # Flush remaining audio
    if audio_buffer:
        chunk = np.concatenate(audio_buffer)
        transcription_queue.put(chunk)
    transcription_queue.put(None) # signal worker to exit
    t.join(timeout=30)
    print("Done.")
finally:
    sock.close()