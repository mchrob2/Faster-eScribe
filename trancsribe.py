#!/usr/bin/env python3
"""
Pi5 Two-Stage Live Transcriber for INMP441 Microphone
- tiny.en  → interim "live preview" every ~1.5s while you speak  (fast, rough)
- base.en  → final accurate result after silence detected         (slow, accurate)

Usage:
    python3 transcriber.py

Dependencies:
    pip3 install faster-whisper numpy
"""

import socket
import threading
import queue
import time
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
from faster_whisper import WhisperModel
import sys
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────
HOST    = "0.0.0.0"
PORT    = 5005
RATE    = 16000

VAD_RMS_DEFAULT = 200   # lower = more sensitive; raise if background noise triggers

SILENCE_S      = 0.8   # seconds of quiet before flushing to final transcription
MAX_SEG_S      = 6.0   # force-flush after this many seconds regardless
INTERIM_EVERY  = 1.5   # how often (seconds) to update the live preview while speaking

# ── Models ──────────────────────────────────────────────────────────────────
# tiny.en: ~40MB RAM, ~0.3x realtime on Pi5 → used for fast interim updates
# base.en: ~150MB RAM, ~1x realtime on Pi5  → used for accurate final output
INTERIM_MODEL = "tiny.en"
FINAL_MODEL   = "base.en"

# ── Shared audio state ──────────────────────────────────────────────────────
# audio_buf_lock protects audio_buf which is written by udp_thread
# and read by both interim_thread and the flush path.
audio_buf      = []
audio_buf_lock = threading.Lock()

last_voice       = [0.0]
live_rms         = [0]
packet_count     = [0]
bytes_received   = [0]
last_packet_time = [time.time()]
audio_stats      = {'peak': 0, 'min': 0, 'max': 0, 'avg': 0}
vad_threshold    = [VAD_RMS_DEFAULT]

recording = threading.Event()
connected = threading.Event()

# Two separate queues: one for each model/thread
interim_queue = queue.Queue()   # snapshots of in-progress audio for tiny.en
final_queue   = queue.Queue()   # completed segments for base.en

# Results back to GUI
interim_result = [""]           # latest rough preview text (replaced each update)
final_queue_gui = queue.Queue() # finalized segments to append to transcript

# ── Load both models ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Pi5 Two-Stage Live Transcriber")
print("="*60)
print(f"Loading interim model  '{INTERIM_MODEL}'...")
interim_model = WhisperModel(INTERIM_MODEL, device="cpu", compute_type="int8", num_workers=1)
print(f"✓ Interim model loaded  ({INTERIM_MODEL})")

print(f"Loading final model    '{FINAL_MODEL}'...")
final_model = WhisperModel(FINAL_MODEL, device="cpu", compute_type="int8", num_workers=2)
print(f"✓ Final model loaded    ({FINAL_MODEL})")

print(f"\n✓ VAD threshold : {VAD_RMS_DEFAULT}")
print(f"✓ Silence gap   : {SILENCE_S}s  |  Max segment: {MAX_SEG_S}s")
print(f"✓ Interim update: every {INTERIM_EVERY}s while speaking")
print(f"\nExpected behaviour:")
print(f"  • Gray preview updates every ~{INTERIM_EVERY}s while you speak")
print(f"  • Final accurate line appears ~1-3s after you pause\n")

# ── UDP receiver thread ───────────────────────────────────────────────────────
def udp_thread():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        sock.bind((HOST, PORT))
        sock.settimeout(1.0)
        print(f"✓ Listening on UDP {HOST}:{PORT}")
        print("Waiting for ESP32...\n")

        last_interim_push = 0.0
        packet_errors     = 0

        while True:
            try:
                data, addr = sock.recvfrom(8192)
                packet_count[0]    += 1
                bytes_received[0]  += len(data)
                last_packet_time[0] = time.time()
                connected.set()
                packet_errors = 0

                if packet_count[0] == 1:
                    s = np.frombuffer(data, dtype=np.int16)
                    rms_first = float(np.sqrt(np.mean(s.astype(np.float32)**2)))
                    print(f"✓ First packet from {addr[0]}  |  RMS={rms_first:.1f}  threshold={vad_threshold[0]}")
                    if rms_first < vad_threshold[0]:
                        print(f"  ⚠ RMS below threshold — lower the slider until the bar turns green when you speak\n")
                    else:
                        print(f"  ✓ Audio level looks good\n")

                if not recording.is_set():
                    continue

                samples = np.frombuffer(data, dtype=np.int16).copy()

                audio_stats['peak'] = int(np.abs(samples).max())
                audio_stats['min']  = int(samples.min())
                audio_stats['max']  = int(samples.max())
                audio_stats['avg']  = float(samples.mean())

                rms = int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                live_rms[0] = rms

                now = time.time()

                if rms >= vad_threshold[0]:
                    if not last_voice[0]:
                        print(f"🎤 Voice detected (RMS={rms})")
                    last_voice[0] = now
                    with audio_buf_lock:
                        audio_buf.append(samples)

                    # Push a snapshot to the interim queue periodically
                    if now - last_interim_push >= INTERIM_EVERY:
                        last_interim_push = now
                        with audio_buf_lock:
                            # Limit snapshot to last 6 seconds so tiny.en stays fast
                            max_samples = int(RATE * 6)
                            snap = np.concatenate(audio_buf)[-max_samples:].copy()
                        interim_queue.put(snap)

                elif audio_buf:
                    with audio_buf_lock:
                        audio_buf.append(samples)
                    silence = now - last_voice[0]
                    with audio_buf_lock:
                        total_s = sum(len(a) for a in audio_buf) / RATE

                    if silence >= SILENCE_S or total_s >= MAX_SEG_S:
                        _flush_to_final()
                        last_interim_push = 0.0

            except socket.timeout:
                packet_errors += 1
                if packet_errors > 3:
                    connected.clear()
                if packet_count[0] > 0 and packet_errors > 3 and packet_errors % 10 == 0:
                    print("⚠ No packets for 3+ seconds")

            except Exception as e:
                print(f"❌ UDP error: {e}")

    except Exception as e:
        print(f"❌ UDP thread fatal: {e}")
        sys.exit(1)


def _flush_to_final():
    """Grab current buffer, clear it, send to final transcription."""
    with audio_buf_lock:
        if not audio_buf:
            return
        audio = np.concatenate(audio_buf).astype(np.float32) / 32768.0
        audio_buf.clear()

    last_voice[0] = 0.0
    interim_result[0] = ""   # clear live preview — final result is coming
    duration = len(audio) / RATE
    print(f"📦 Flushing {duration:.2f}s to final queue  (queue depth: {final_queue.qsize()})")
    final_queue.put(audio)


# ── Interim transcription thread (tiny.en) ────────────────────────────────────
def interim_thread():
    """Runs tiny.en on rolling snapshots for the live preview."""
    while True:
        try:
            snap = interim_queue.get()
            if snap is None:
                break

            audio = snap.astype(np.float32) / 32768.0
            dur   = len(audio) / RATE

            segments, _ = interim_model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200, "threshold": 0.4},
                beam_size=1,
                best_of=1,
                temperature=0.0,
                no_speech_threshold=0.7,
            )

            parts = [s.text.strip() for s in segments]
            text  = " ".join(parts)

            if text:
                interim_result[0] = text   # GUI picks this up every 100ms
                print(f"  ↻ Interim ({dur:.1f}s): {text[:70]}...")

        except Exception as e:
            print(f"❌ Interim error: {e}")


# ── Final transcription thread (base.en) ──────────────────────────────────────
def final_transcribe_thread():
    """Runs base.en on completed segments for accurate final output."""
    while True:
        try:
            audio = final_queue.get()
            if audio is None:
                break

            t0  = time.time()
            dur = len(audio) / RATE
            print(f"✏️  Final transcribing {dur:.2f}s...", end=" ", flush=True)

            segments, _ = final_model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 250,
                    "threshold": 0.45,
                    "neg_threshold": 0.15,
                },
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

            parts = []
            for seg in segments:
                parts.append(seg.text.strip())
                print(f"\n  [{seg.start:.1f}->{seg.end:.1f}] {seg.text.strip()}")

            text    = " ".join(parts)
            elapsed = time.time() - t0

            if text:
                ratio = elapsed / max(dur, 0.01)
                print(f"✓ Final ({elapsed:.2f}s, {ratio:.2f}x realtime): {text[:80]}...")
                final_queue_gui.put(text)
                interim_result[0] = ""   # clear preview now that final is in
            else:
                print(f"⏭ No speech ({elapsed:.2f}s)")

        except Exception as e:
            print(f"❌ Final transcription error: {e}")


# ── GUI ────────────────────────────────────────────────────────────────────────
class TranscriberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("INMP441 Live Transcriber")
        self.root.geometry("940x720")
        self.root.configure(bg="#1e1e1e")
        try:
            self.root.iconbitmap("microphone.ico")
        except:
            pass
        self.setup_ui()
        self.update_status()

    def setup_ui(self):
        # Title
        tf = tk.Frame(self.root, bg="#2d2d2d", height=44)
        tf.pack(fill="x")
        tf.pack_propagate(False)
        tk.Label(
            tf,
            text=f"🎙️  INMP441 Live Transcriber  —  {INTERIM_MODEL} preview  /  {FINAL_MODEL} final",
            bg="#2d2d2d", fg="#ffffff", font=("Arial", 12, "bold")
        ).pack(pady=10)

        # Status row
        sf = tk.Frame(self.root, bg="#252525", height=62)
        sf.pack(fill="x", padx=10, pady=(10, 5))
        sf.pack_propagate(False)

        self.conn_label   = tk.Label(sf, text="● Disconnected", font=("Arial", 10, "bold"), fg="#888", bg="#252525")
        self.conn_label.place(x=10, y=10)
        self.packet_label = tk.Label(sf, text="Packets: 0",  font=("Arial", 10), fg="#aaa", bg="#252525")
        self.packet_label.place(x=10, y=34)
        self.bytes_label  = tk.Label(sf, text="Data: 0 KB",  font=("Arial", 10), fg="#aaa", bg="#252525")
        self.bytes_label.place(x=140, y=34)
        self.queue_label  = tk.Label(sf, text="Final queue: 0", font=("Arial", 10), fg="#aaa", bg="#252525")
        self.queue_label.place(x=280, y=34)
        self.rec_label    = tk.Label(sf, text="⏹ Stopped", font=("Arial", 12, "bold"), fg="#888", bg="#252525")
        self.rec_label.place(x=480, y=10)
        self.stats_label  = tk.Label(sf, text="Peak: 0",    font=("Arial", 9),  fg="#aaa", bg="#252525")
        self.stats_label.place(x=480, y=36)

        # RMS meter
        mf = tk.Frame(self.root, bg="#1e1e1e")
        mf.pack(fill="x", padx=10, pady=4)
        tk.Label(mf, text="Level", fg="#aaa", bg="#1e1e1e", font=("Arial", 10)).pack(side="left", padx=(0, 8))
        self.meter = tk.Canvas(mf, height=20, bg="#2d2d2d", highlightthickness=0)
        self.meter.pack(side="left", fill="x", expand=True)
        tk.Label(mf, text="Threshold", fg="#aaa", bg="#1e1e1e", font=("Arial", 10)).pack(side="left", padx=(10, 5))
        self.thr_disp = tk.Label(mf, text=str(VAD_RMS_DEFAULT), fg="#ffa500", bg="#1e1e1e", font=("Arial", 10, "bold"), width=5)
        self.thr_disp.pack(side="right", padx=(0, 5))
        self.slider = ttk.Scale(mf, from_=50, to=3000, orient="horizontal", length=130, command=self.on_threshold_change)
        self.slider.set(VAD_RMS_DEFAULT)
        self.slider.pack(side="right")

        # ── Live preview box ──────────────────────────────────────────────────
        prev_frame = tk.Frame(self.root, bg="#1e1e1e")
        prev_frame.pack(fill="x", padx=10, pady=(6, 2))

        tk.Label(
            prev_frame, text="Live preview  (tiny.en — rough)", fg="#666", bg="#1e1e1e",
            font=("Arial", 9, "italic")
        ).pack(anchor="w")

        self.live_label = tk.Label(
            prev_frame,
            text="",
            bg="#252525", fg="#888888",
            font=("Consolas", 11, "italic"),
            anchor="w", justify="left",
            wraplength=880,
            padx=10, pady=6
        )
        self.live_label.pack(fill="x")

        # ── Final transcript ──────────────────────────────────────────────────
        tr_frame = tk.Frame(self.root, bg="#1e1e1e")
        tr_frame.pack(fill="both", expand=True, padx=10, pady=(4, 4))

        tk.Label(
            tr_frame, text="Transcript  (base.en — accurate)", fg="#aaa", bg="#1e1e1e",
            font=("Arial", 10, "bold")
        ).pack(anchor="w")

        self.transcript = scrolledtext.ScrolledText(
            tr_frame, wrap="word", font=("Consolas", 12),
            bg="#2d2d2d", fg="#e8e8e8", insertbackground="#e8e8e8",
            relief="flat", padx=10, pady=10
        )
        self.transcript.pack(fill="both", expand=True, pady=(4, 0))

        # Buttons
        bf = tk.Frame(self.root, bg="#1e1e1e")
        bf.pack(fill="x", padx=10, pady=8)

        self.record_btn = tk.Button(
            bf, text="▶ Start Recording", command=self.toggle_recording,
            bg="#2e7d32", fg="white", font=("Arial", 12, "bold"),
            relief="flat", padx=20, pady=8, cursor="hand2",
            activebackground="#1b5e20", activeforeground="white"
        )
        self.record_btn.pack(side="left", padx=5)

        for label, cmd in [
            ("🗑 Clear",  self.clear_transcript),
            ("💾 Save",   self.save_transcript),
            ("ℹ️ Info",   self.show_info),
        ]:
            tk.Button(
                bf, text=label, command=cmd,
                bg="#424242", fg="white", font=("Arial", 11),
                relief="flat", padx=15, pady=8, cursor="hand2",
                activebackground="#616161", activeforeground="white"
            ).pack(side="left", padx=5)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def on_threshold_change(self, val):
        v = int(float(val))
        vad_threshold[0] = v
        self.thr_disp.config(text=str(v))

    def toggle_recording(self):
        if recording.is_set():
            recording.clear()
            with audio_buf_lock:
                audio_buf.clear()
            self.record_btn.config(text="▶ Start Recording", bg="#2e7d32")
            self.rec_label.config(text="⏹ Stopped", fg="#888")
            self.live_label.config(text="")
            print("\n⏸ Recording stopped")
        else:
            recording.set()
            self.record_btn.config(text="■ Stop Recording", bg="#c62828")
            self.rec_label.config(text="● Recording", fg="#4caf50")
            print("\n🎙 Recording started")

    def clear_transcript(self):
        self.transcript.config(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.config(state="disabled")
        self.live_label.config(text="")

    def save_transcript(self):
        try:
            fn = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(fn, "w") as f:
                f.write(f"INMP441 Lecture Transcription\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Models: {INTERIM_MODEL} (interim) / {FINAL_MODEL} (final)\n")
                f.write(f"Rate: {RATE} Hz  |  VAD: {vad_threshold[0]}\n")
                f.write("-" * 50 + "\n\n")
                f.write(self.transcript.get("1.0", "end-1c"))
            messagebox.showinfo("Saved", f"Saved to:\n{fn}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")

    def show_info(self):
        messagebox.showinfo("About", f"""INMP441 Two-Stage Live Transcriber

Interim model : {INTERIM_MODEL}  (fast preview, updates every {INTERIM_EVERY}s)
Final model   : {FINAL_MODEL}   (accurate, fires after {SILENCE_S}s silence)
Sample rate   : {RATE} Hz
VAD threshold : {vad_threshold[0]}

What you'll see:
  Gray italic text — rough live preview while you speak
  White text below  — accurate final transcript after each pause

Tuning:
  Nothing shows up     → lower threshold slider
  Too much background  → raise threshold slider
  Final queue > 2      → system falling behind; raise threshold or speak slower
""")

    # ── Status loop ───────────────────────────────────────────────────────────
    def update_status(self):
        try:
            self.conn_label.config(
                text="● Connected" if connected.is_set() else "● Disconnected",
                fg="#4caf50"       if connected.is_set() else "#f44336"
            )
            self.packet_label.config(text=f"Packets: {packet_count[0]}")
            self.bytes_label.config( text=f"Data: {bytes_received[0]/1024:.0f} KB")

            qd = final_queue.qsize()
            self.queue_label.config(text=f"Final queue: {qd}", fg="#f44336" if qd > 2 else "#aaa")

            if audio_stats['peak'] > 0:
                self.stats_label.config(
                    text=f"Peak: {audio_stats['peak']}  Min: {audio_stats['min']}  Max: {audio_stats['max']}"
                )

            # RMS meter
            rms = live_rms[0]
            thr = vad_threshold[0]
            w   = self.meter.winfo_width()
            if w > 1:
                self.meter.delete("all")
                self.meter.create_rectangle(0, 0, w, 20, fill="#2d2d2d", outline="")
                bar_w = min(int(w * rms / 3000), w)
                self.meter.create_rectangle(0, 0, bar_w, 20,
                    fill="#4caf50" if rms >= thr else "#ff9800", outline="")
                tx = max(1, min(int(w * thr / 3000), w - 1))
                self.meter.create_line(tx, 0, tx, 20, fill="#f44336", width=2)
                self.meter.create_text(5,   10, text=str(rms), fill="white", font=("Arial", 8), anchor="w")
                self.meter.create_text(w-5, 10, text=str(thr), fill="white", font=("Arial", 8), anchor="e")

            # Update live preview label
            self.live_label.config(text=interim_result[0])

            # Drain final results into transcript
            new_texts = []
            while not final_queue_gui.empty():
                new_texts.append(final_queue_gui.get_nowait())
            if new_texts:
                self.transcript.config(state="normal")
                for text in new_texts:
                    ts = datetime.now().strftime("%H:%M:%S")
                    self.transcript.insert("end", f"[{ts}] {text}\n\n")
                self.transcript.see("end")
                self.transcript.config(state="disabled")

        except Exception as e:
            print(f"UI error: {e}")

        self.root.after(100, self.update_status)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    try:
        import faster_whisper, numpy
    except ImportError as e:
        print(f"❌ Missing: {e}\n  pip3 install faster-whisper numpy")
        sys.exit(1)

    threading.Thread(target=udp_thread,              daemon=True).start()
    threading.Thread(target=interim_thread,          daemon=True).start()
    threading.Thread(target=final_transcribe_thread, daemon=True).start()

    root = tk.Tk()
    TranscriberGUI(root)

    print("="*60)
    print("System ready — 3 threads running:")
    print("  UDP receiver   → audio capture")
    print(f"  Interim thread → {INTERIM_MODEL} preview every {INTERIM_EVERY}s")
    print(f"  Final thread   → {FINAL_MODEL} accurate result after silence")
    print("\n  1. Connect Pi5 to 'LectureAudio' WiFi")
    print("  2. Click  ▶ Start Recording")
    print("  3. Speak — gray preview updates while you talk,")
    print("     white final text appears after each pause")
    print("="*60 + "\n")

    def on_close():
        interim_queue.put(None)
        final_queue.put(None)
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()