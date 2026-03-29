"""
Pi5 UDP Audio Receiver + Transcriber — GUI Version
====================================================

Display layout (7" HDMI touchscreen, 800x480 typical):

  ┌─────────────────────────────────────────┐
  │  🎙 Live Transcriber     ● Connected    │  <- status bar
  ├─────────────────────────────────────────┤
  │                                         │
  │   [Final transcript scrolls here]       │  <- scrollable transcript
  │                                         │
  ├─────────────────────────────────────────┤
  │  ⟳ interim text appears here...         │  <- live interim line
  ├─────────────────────────────────────────┤
  │           [ START / STOP ]              │  <- big touch button
  └─────────────────────────────────────────┘

Threading:
  - Main thread     : tkinter event loop
  - udp_vad thread  : receives UDP, runs VAD, pushes to trans_queue
  - transcribe thread: pulls from trans_queue, pushes to gui_queue
  - GUI polling     : root.after(100) drains gui_queue safely on main thread
"""

import socket
import struct
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from faster_whisper import WhisperModel
from scipy import signal
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────

PICO_W_IP    = "192.168.4.1"
UDP_PORT     = 5005
FS           = 16000
MODEL_SIZE   = "tiny.en"
DEVICE       = "cpu"
COMPUTE_TYPE = "int8"

VAD_THRESHOLD    = 0.008
VAD_SPEECH_ONSET = 3
VAD_SILENCE_END  = 20
VAD_PRE_ROLL     = 8
MIN_CLIP_SEC     = 0.4
MAX_CLIP_SEC     = 12.0

INTERIM_INTERVAL_SEC = 1.5

# ─── Logging ──────────────────────────────────────────────────────────────────

load_dotenv()
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
dt_str   = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"{LOG_DIR}/{dt_str}.txt"

# ─── Shared state ─────────────────────────────────────────────────────────────

stop_event      = threading.Event()   # signals threads to exit cleanly
running_event   = threading.Event()   # controls whether transcription is active

# Priority queue: (priority, counter, audio, kind)
trans_queue = queue.PriorityQueue(maxsize=20)
_pq_counter = 0
_pq_lock    = threading.Lock()

# GUI update queue: ("interim", text) | ("final", text) | ("status", text) | ("stats", text)
gui_queue = queue.Queue()

# ─── DC-block filter ──────────────────────────────────────────────────────────

_b_dc   = np.array([1.0, -1.0],   dtype=np.float64)
_a_dc   = np.array([1.0, -0.999], dtype=np.float64)
_zi_dc  = signal.lfilter_zi(_b_dc, _a_dc) * 0.0
_dc_lck = threading.Lock()

def dc_block(x: np.ndarray) -> np.ndarray:
    global _zi_dc
    with _dc_lck:
        y, _zi_dc = signal.lfilter(_b_dc, _a_dc, x, zi=_zi_dc)
    return y.astype(np.float32)

# ─── Segment flusher ──────────────────────────────────────────────────────────

def _flush_segment(frames: list, kind: str = "final") -> None:
    global _pq_counter
    if not frames:
        return
    audio = np.concatenate(frames)
    dur   = len(audio) / FS
    if dur < MIN_CLIP_SEC:
        return
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.5
    priority = 0 if kind == "final" else 1
    with _pq_lock:
        _pq_counter += 1
        counter = _pq_counter
    try:
        trans_queue.put_nowait((priority, counter, audio, kind))
    except queue.Full:
        gui_queue.put(("status", "⚠ Queue full — dropping segment"))

# ─── UDP receive + VAD thread ─────────────────────────────────────────────────

def udp_vad_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)

    def send_hello():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(b"HELLO", (PICO_W_IP, UDP_PORT))
        except OSError:
            pass

    send_hello()
    gui_queue.put(("status", "⟳ Waiting for Pico W..."))

    state             = "SILENCE"
    speech_count      = 0
    silence_count     = 0
    pre_roll          = []
    current_seg       = []
    last_interim_time = 0.0
    last_seq          = None
    recv_count        = 0
    drop_count        = 0
    pico_connected    = False

    try:
        while not stop_event.is_set():
            # If not actively transcribing, drain packets silently to stay connected
            if not running_event.is_set():
                try:
                    sock.recvfrom(4096)
                except socket.timeout:
                    send_hello()
                # Reset VAD state when paused so we start clean on resume
                state         = "SILENCE"
                speech_count  = 0
                silence_count = 0
                pre_roll      = []
                current_seg   = []
                continue

            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                send_hello()
                if pico_connected:
                    pico_connected = False
                    gui_queue.put(("status", "⚠ Pico W not responding..."))
                continue

            if len(data) < 5:
                continue

            if not pico_connected:
                pico_connected = True
                gui_queue.put(("status", "● Connected"))

            seq, _    = struct.unpack_from(">HH", data, 0)
            payload   = data[4:]
            n_samples = len(payload) // 2

            if last_seq is not None:
                gap = (seq - last_seq - 1) & 0xFFFF
                if 0 < gap < 200:
                    drop_count += gap
            last_seq    = seq
            recv_count += 1

            frame_i16 = np.frombuffer(payload, dtype="<i2").copy()
            frame_f32 = frame_i16.astype(np.float32) / 32768.0
            frame_f32 = dc_block(frame_f32)

            rms = float(np.sqrt(np.mean(frame_f32 ** 2)))

            if state == "SILENCE":
                pre_roll.append(frame_f32)
                if len(pre_roll) > VAD_PRE_ROLL:
                    pre_roll.pop(0)
                if rms > VAD_THRESHOLD:
                    speech_count += 1
                    if speech_count >= VAD_SPEECH_ONSET:
                        state             = "SPEECH"
                        current_seg       = list(pre_roll)
                        pre_roll          = []
                        speech_count      = 0
                        silence_count     = 0
                        last_interim_time = time.monotonic()
                else:
                    speech_count = 0

            else:  # SPEECH
                current_seg.append(frame_f32)

                now_t    = time.monotonic()
                clip_dur = len(current_seg) * n_samples / FS
                if (clip_dur >= INTERIM_INTERVAL_SEC and
                        (now_t - last_interim_time) >= INTERIM_INTERVAL_SEC):
                    _flush_segment(list(current_seg), kind="interim")
                    last_interim_time = now_t

                if rms < VAD_THRESHOLD:
                    silence_count += 1
                    if silence_count >= VAD_SILENCE_END:
                        _flush_segment(current_seg, kind="final")
                        state             = "SILENCE"
                        current_seg       = []
                        silence_count     = 0
                        speech_count      = 0
                        last_interim_time = 0.0
                else:
                    silence_count = 0

                clip_dur = len(current_seg) * n_samples / FS
                if clip_dur >= MAX_CLIP_SEC:
                    _flush_segment(current_seg, kind="final")
                    current_seg       = []
                    silence_count     = 0
                    last_interim_time = time.monotonic()

            # Packet stats every 500 packets
            if recv_count > 0 and recv_count % 500 == 0:
                pct = 100.0 * drop_count / max(recv_count + drop_count, 1)
                gui_queue.put(("stats", f"Packets: {recv_count}  Dropped: {drop_count} ({pct:.1f}%)"))

    finally:
        sock.close()

# ─── Transcription thread ─────────────────────────────────────────────────────

def transcribe_loop(model: WhisperModel) -> None:
    with open(LOG_FILE, "a") as log:
        while not stop_event.is_set():
            try:
                priority, _, audio, kind = trans_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Discard queued interims if not running
            if not running_event.is_set():
                trans_queue.task_done()
                continue

            t0 = time.monotonic()
            segments, _ = model.transcribe(
                audio,
                beam_size=1,
                temperature=0,
                vad_filter=True,
                condition_on_previous_text=False,
                language="en",
            )

            parts = [seg.text.strip() for seg in segments if seg.text.strip()]

            if parts:
                elapsed = time.monotonic() - t0
                text    = " ".join(parts)
                ts      = datetime.now(ZoneInfo("America/Chicago")).strftime("%H:%M:%S")

                if kind == "interim":
                    gui_queue.put(("interim", text))
                else:
                    line = f"[{ts}]  {text}"
                    gui_queue.put(("final", line))
                    log.write(f"[{ts}] ({elapsed:.2f}s) {text}\n")
                    log.flush()

            trans_queue.task_done()

# ─── GUI ──────────────────────────────────────────────────────────────────────

class TranscriberApp:
    # Colours
    BG          = "#1a1a2e"   # deep navy
    PANEL_BG    = "#16213e"   # slightly lighter navy
    ACCENT      = "#0f3460"   # button bg (idle)
    ACCENT_STOP = "#8b0000"   # button bg (recording)
    TEXT_MAIN   = "#e0e0e0"   # primary text
    TEXT_DIM    = "#888888"   # interim / secondary
    TEXT_STATUS = "#4ecca3"   # teal status text
    BUTTON_TEXT = "#ffffff"

    def __init__(self, root: tk.Tk, model: WhisperModel):
        self.root  = root
        self.model = model
        self._build_ui()
        self._start_threads()
        self._poll_gui_queue()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        root.title("Live Transcriber")
        root.configure(bg=self.BG)
        root.attributes("-fullscreen", True)       # fill the 7" display
        root.bind("<Escape>", lambda e: self._quit())   # escape to exit (dev)

        f_title  = tkfont.Font(family="DejaVu Sans", size=14, weight="bold")
        f_status = tkfont.Font(family="DejaVu Sans", size=11)
        f_trans  = tkfont.Font(family="DejaVu Sans", size=13)
        f_interim = tkfont.Font(family="DejaVu Sans", size=12, slant="italic")
        f_button = tkfont.Font(family="DejaVu Sans", size=18, weight="bold")
        f_stats  = tkfont.Font(family="DejaVu Sans", size=9)

        # ── Top bar ──────────────────────────────────────────────────────────
        top_bar = tk.Frame(root, bg=self.PANEL_BG, pady=6)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            top_bar, text="🎙  Live Transcriber",
            font=f_title, bg=self.PANEL_BG, fg=self.TEXT_MAIN
        ).pack(side=tk.LEFT, padx=14)

        self.status_label = tk.Label(
            top_bar, text="○ Idle",
            font=f_status, bg=self.PANEL_BG, fg=self.TEXT_DIM
        )
        self.status_label.pack(side=tk.RIGHT, padx=14)

        # ── Transcript area ───────────────────────────────────────────────────
        trans_frame = tk.Frame(root, bg=self.BG)
        trans_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 0))

        scrollbar = tk.Scrollbar(trans_frame, bg=self.PANEL_BG, troughcolor=self.BG)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.transcript = tk.Text(
            trans_frame,
            font=f_trans,
            bg=self.BG, fg=self.TEXT_MAIN,
            insertbackground=self.TEXT_MAIN,
            selectbackground=self.ACCENT,
            relief=tk.FLAT, bd=0,
            wrap=tk.WORD,
            state=tk.DISABLED,
            yscrollcommand=scrollbar.set,
            padx=8, pady=6,
            spacing2=4,          # extra line spacing for readability
        )
        self.transcript.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.transcript.yview)

        # ── Interim line ──────────────────────────────────────────────────────
        interim_frame = tk.Frame(root, bg=self.PANEL_BG, pady=5)
        interim_frame.pack(fill=tk.X, padx=10, pady=(4, 0))

        self.interim_label = tk.Label(
            interim_frame,
            text="",
            font=f_interim,
            bg=self.PANEL_BG, fg=self.TEXT_DIM,
            anchor=tk.W, padx=8,
            wraplength=760,
        )
        self.interim_label.pack(fill=tk.X)

        # ── Stats bar ─────────────────────────────────────────────────────────
        self.stats_label = tk.Label(
            root, text="",
            font=f_stats,
            bg=self.BG, fg=self.TEXT_DIM,
            anchor=tk.W, padx=12,
        )
        self.stats_label.pack(fill=tk.X, pady=(2, 0))

        # ── Start / Stop button ───────────────────────────────────────────────
        self.btn = tk.Button(
            root,
            text="START",
            font=f_button,
            bg=self.ACCENT, fg=self.BUTTON_TEXT,
            activebackground=self.ACCENT_STOP,
            activeforeground=self.BUTTON_TEXT,
            relief=tk.FLAT, bd=0,
            padx=0, pady=18,
            cursor="hand2",
            command=self._toggle,
        )
        self.btn.pack(fill=tk.X, padx=10, pady=10)

    # ── Thread management ─────────────────────────────────────────────────────

    def _start_threads(self):
        self.threads = [
            threading.Thread(target=udp_vad_loop,                    daemon=True, name="udp-vad"),
            threading.Thread(target=transcribe_loop, args=(self.model,), daemon=True, name="transcribe"),
        ]
        for t in self.threads:
            t.start()

    # ── Start / Stop toggle ───────────────────────────────────────────────────

    def _toggle(self):
        if running_event.is_set():
            running_event.clear()
            self.btn.config(text="START", bg=self.ACCENT)
            self.interim_label.config(text="")
            self.status_label.config(text="⏸ Paused", fg=self.TEXT_DIM)
        else:
            running_event.set()
            self.btn.config(text="STOP", bg=self.ACCENT_STOP)
            self.status_label.config(text="⟳ Waiting for audio...", fg=self.TEXT_STATUS)

    # ── GUI queue polling (runs on main thread via after()) ───────────────────

    def _poll_gui_queue(self):
        try:
            while True:
                kind, text = gui_queue.get_nowait()
                if kind == "interim":
                    self._set_interim(text)
                elif kind == "final":
                    self._append_final(text)
                elif kind == "status":
                    self._set_status(text)
                elif kind == "stats":
                    self.stats_label.config(text=text)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_gui_queue)   # poll every 100 ms

    def _set_interim(self, text: str):
        self.interim_label.config(text=f"⟳  {text}")

    def _append_final(self, text: str):
        # Clear interim when final arrives
        self.interim_label.config(text="")
        self.transcript.config(state=tk.NORMAL)
        if self.transcript.get("1.0", tk.END).strip():
            self.transcript.insert(tk.END, "\n")
        self.transcript.insert(tk.END, text)
        self.transcript.config(state=tk.DISABLED)
        self.transcript.see(tk.END)    # auto-scroll to latest

    def _set_status(self, text: str):
        colour = self.TEXT_STATUS if "●" in text else self.TEXT_DIM
        self.status_label.config(text=text, fg=colour)

    # ── Clean exit ────────────────────────────────────────────────────────────

    def _quit(self):
        stop_event.set()
        self.root.after(500, self.root.destroy)

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Whisper model...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model loaded. Launching GUI...\n")

    root = tk.Tk()
    app  = TranscriberApp(root, model)
    root.mainloop()

    # Cleanup after window closes
    stop_event.set()
    print("Goodbye")
