"""
Pi5 UDP Audio Receiver + Transcriber — VAD-Triggered Low Latency Version
=========================================================================

Architecture:
    UDP recv thread
        -> per-packet VAD state machine (energy threshold)
            -> speech segments pushed to queue
                -> faster-whisper transcription thread
                    -> print + log

Key improvements vs previous version:
  1. VAD-triggered segmentation replaces fixed 2.5 s slicing.
     Transcription fires ~200 ms after speech ends, not after a full window.
     Typical end-to-end latency: 0.5 - 1.0 s from end of utterance to text.

  2. Pre-roll buffer: captures the ~80 ms of audio before VAD triggers so
     the start of a word is never clipped.

  3. Vectorised DC-block via scipy.signal.lfilter (C-speed IIR) instead of
     a per-sample Python loop -- 100-200x faster.

  4. temperature=0 passed to Whisper for deterministic, slightly faster decoding.

  5. Simpler two-thread design (removed slicer thread -- VAD does its job).

Tuning VAD_THRESHOLD:
  - Too sensitive (fires on background noise): raise it (e.g. 0.015)
  - Missing quiet speech:                      lower it (e.g. 0.004)
  Set PRINT_RMS = True to see live RMS values and dial it in.
"""

import socket
import struct
import os
import queue
import threading
import time
import numpy as np
from faster_whisper import WhisperModel
from scipy import signal
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────

PICO_W_IP    = "192.168.4.1"   # Pico W AP gateway (fixed)
UDP_PORT     = 5005
FS           = 16000            # must match Pico W SAMPLE_RATE
MODEL_SIZE   = "tiny.en"
DEVICE       = "cpu"
COMPUTE_TYPE = "int8"

# VAD tuning
# Each "frame" = one UDP packet = 10 ms of audio (160 samples at 16 kHz).
VAD_THRESHOLD    = 0.008  # RMS energy to distinguish speech from silence.
                           # Raise if background noise causes false starts.
                           # Lower if quiet speech is being missed.
VAD_SPEECH_ONSET = 3      # consecutive loud frames required to enter SPEECH state
                           # (3 frames = 30 ms -- prevents noise spikes)
VAD_SILENCE_END  = 20     # consecutive quiet frames to end a speech segment
                           # (20 frames = 200 ms of trailing silence)
VAD_PRE_ROLL     = 8      # frames of audio kept before speech onset (~80 ms)
                           # ensures the first syllable is never clipped
MIN_CLIP_SEC     = 0.4    # discard segments shorter than this (avoids Whisper hallucinations)
MAX_CLIP_SEC     = 12.0   # force-flush if a segment grows too long

PRINT_RMS = False          # set True temporarily to calibrate VAD_THRESHOLD

# ─── Logging ──────────────────────────────────────────────────────────────────

load_dotenv()
LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
dt_str   = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"{LOG_DIR}/{dt_str}.txt"

# ─── Load Whisper model ───────────────────────────────────────────────────────

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded.\n")

stop_event  = threading.Event()
trans_queue = queue.Queue(maxsize=10)

# ─── DC-block filter (stateful scipy IIR -- runs at C speed) ──────────────────
#
#  Transfer function:  H(z) = (1 - z^-1) / (1 - 0.999 z^-1)
#  Passes everything above ~8 Hz; removes DC offset from the ADC midpoint.
#
_b_dc   = np.array([1.0, -1.0],   dtype=np.float64)
_a_dc   = np.array([1.0, -0.999], dtype=np.float64)
_zi_dc  = signal.lfilter_zi(_b_dc, _a_dc) * 0.0   # initial filter state
_dc_lck = threading.Lock()

def dc_block(x: np.ndarray) -> np.ndarray:
    """Apply DC-blocking high-pass filter, preserving state across packets."""
    global _zi_dc
    with _dc_lck:
        y, _zi_dc = signal.lfilter(_b_dc, _a_dc, x, zi=_zi_dc)
    return y.astype(np.float32)

# ─── Segment flusher ─────────────────────────────────────────────────────────

def _flush_segment(frames: list) -> None:
    """Concatenate frames, normalise, and push to the transcription queue."""
    if not frames:
        return
    audio = np.concatenate(frames)
    dur   = len(audio) / FS
    if dur < MIN_CLIP_SEC:
        return                        # too short -- likely a noise burst
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.5   # normalise to half full-scale
    try:
        trans_queue.put_nowait(audio)
    except queue.Full:
        print("[VAD] transcription queue full -- dropping segment")

# ─── UDP receive + VAD thread ─────────────────────────────────────────────────

def udp_vad_loop() -> None:
    """
    Receives UDP audio packets from the Pico W.
    Runs a per-frame energy VAD state machine and pushes complete speech
    segments to trans_queue for transcription.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)

    def send_hello():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(b"HELLO", (PICO_W_IP, UDP_PORT))

    send_hello()
    print(f"Sent HELLO to Pico W at {PICO_W_IP}:{UDP_PORT}")
    print(f"Listening for audio on UDP port {UDP_PORT}...\n")
    if PRINT_RMS:
        print("[RMS calibration mode ON -- set VAD_THRESHOLD based on values below]")

    # VAD state machine
    state         = "SILENCE"
    speech_count  = 0        # consecutive loud frames
    silence_count = 0        # consecutive quiet frames
    pre_roll      = []       # ring of recent frames (size = VAD_PRE_ROLL)
    current_seg   = []       # frames accumulating for current speech segment

    last_seq   = None
    recv_count = 0
    drop_count = 0

    try:
        while not stop_event.is_set():
            # Receive packet
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                # Re-announce in case the Pico W rebooted
                send_hello()
                continue

            if len(data) < 5:
                continue

            seq, _    = struct.unpack_from(">HH", data, 0)
            payload   = data[4:]
            n_samples = len(payload) // 2   # 2 bytes per int16

            # Dropped packet accounting
            if last_seq is not None:
                gap = (seq - last_seq - 1) & 0xFFFF
                if 0 < gap < 200:
                    drop_count += gap
            last_seq    = seq
            recv_count += 1

            # Decode 16-bit PCM -> float32
            frame_i16 = np.frombuffer(payload, dtype="<i2").copy()
            frame_f32 = frame_i16.astype(np.float32) / 32768.0
            frame_f32 = dc_block(frame_f32)

            # Energy VAD
            rms = float(np.sqrt(np.mean(frame_f32 ** 2)))
            if PRINT_RMS:
                bar = "#" * min(int(rms / 0.001), 60)
                print(f"RMS {rms:.4f}  |{bar}")

            if state == "SILENCE":
                # Maintain pre-roll so we don't clip word onsets
                pre_roll.append(frame_f32)
                if len(pre_roll) > VAD_PRE_ROLL:
                    pre_roll.pop(0)

                if rms > VAD_THRESHOLD:
                    speech_count += 1
                    if speech_count >= VAD_SPEECH_ONSET:
                        # Transition: SILENCE -> SPEECH
                        state         = "SPEECH"
                        current_seg   = list(pre_roll)   # include pre-roll audio
                        pre_roll      = []
                        speech_count  = 0
                        silence_count = 0
                else:
                    speech_count = 0

            else:  # state == "SPEECH"
                current_seg.append(frame_f32)

                if rms < VAD_THRESHOLD:
                    silence_count += 1
                    if silence_count >= VAD_SILENCE_END:
                        # Transition: SPEECH -> SILENCE -- flush segment
                        _flush_segment(current_seg)
                        state         = "SILENCE"
                        current_seg   = []
                        silence_count = 0
                        speech_count  = 0
                else:
                    silence_count = 0

                # Force flush if segment grows too long (continuous speech)
                clip_dur = len(current_seg) * n_samples / FS
                if clip_dur >= MAX_CLIP_SEC:
                    _flush_segment(current_seg)
                    current_seg   = []
                    silence_count = 0

            # Periodic stats
            if recv_count > 0 and recv_count % 1000 == 0:
                pct = 100.0 * drop_count / max(recv_count + drop_count, 1)
                print(f"[UDP] recv={recv_count}  dropped={drop_count} ({pct:.1f}%)")
    finally:
        sock.close()

# ─── Transcription thread ─────────────────────────────────────────────────────

def transcribe_loop() -> None:
    """
    Pulls speech segments from trans_queue and transcribes with faster-whisper.
    Prints each result with a wall-clock timestamp and inference time.
    """
    with open(LOG_FILE, "a") as log:
        while not stop_event.is_set():
            try:
                audio = trans_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0 = time.monotonic()

            segments, _ = model.transcribe(
                audio,
                beam_size=1,
                temperature=0,               # deterministic, slightly faster
                vad_filter=True,             # second-pass VAD inside Whisper
                condition_on_previous_text=False,
                language="en",
            )

            parts = [seg.text.strip() for seg in segments if seg.text.strip()]

            if parts:
                elapsed = time.monotonic() - t0
                text    = " ".join(parts)
                ts      = datetime.now(ZoneInfo("America/Chicago")).strftime("%H:%M:%S")
                line    = f"[{ts}] ({elapsed:.2f}s) {text}"
                print(line)
                log.write(line + "\n")
                log.flush()

            trans_queue.task_done()

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting live transcription -- speak into the mic (Ctrl+C to stop)\n")

    threads = [
        threading.Thread(target=udp_vad_loop,    daemon=True, name="udp-vad"),
        threading.Thread(target=transcribe_loop, daemon=True, name="transcribe"),
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        for t in threads:
            t.join(timeout=5)

    print("Goodbye")
