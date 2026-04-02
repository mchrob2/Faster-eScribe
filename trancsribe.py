#!/usr/bin/env python3
"""
Pi5 Two-Stage Live Transcriber for INMP441 Microphone
Optimized layout for 7-inch 800x480 touchscreen

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

VAD_RMS_DEFAULT = 200
SILENCE_S       = 0.8
MAX_SEG_S       = 6.0
INTERIM_EVERY   = 1.5

INTERIM_MODEL = "tiny.en"
FINAL_MODEL   = "base.en"

# ── Shared audio state ──────────────────────────────────────────────────────
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

interim_queue   = queue.Queue()
final_queue     = queue.Queue()
interim_result  = [""]
final_queue_gui = queue.Queue()

# ── Load both models ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Pi5 Two-Stage Live Transcriber  (800x480 layout)")
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
                        print(f"  ⚠ RMS below threshold — lower the slider\n")
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

                    if now - last_interim_push >= INTERIM_EVERY:
                        last_interim_push = now
                        with audio_buf_lock:
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
    with audio_buf_lock:
        if not audio_buf:
            return
        audio = np.concatenate(audio_buf).astype(np.float32) / 32768.0
        audio_buf.clear()

    last_voice[0]     = 0.0
    interim_result[0] = ""
    duration = len(audio) / RATE
    print(f"📦 Flushing {duration:.2f}s to final queue  (depth: {final_queue.qsize()})")
    final_queue.put(audio)


# ── Interim transcription thread (tiny.en) ────────────────────────────────────
def interim_thread():
    while True:
        try:
            snap = interim_queue.get()
            if snap is None:
                break

            audio = snap.astype(np.float32) / 32768.0

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

            text  = " ".join([s.text.strip() for s in segments])
            if text:
                interim_result[0] = text
                print(f"  ↻ Interim: {text[:70]}...")

        except Exception as e:
            print(f"❌ Interim error: {e}")


# ── Final transcription thread (base.en) ──────────────────────────────────────
def final_transcribe_thread():
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
                interim_result[0] = ""
            else:
                print(f"⏭ No speech ({elapsed:.2f}s)")

        except Exception as e:
            print(f"❌ Final transcription error: {e}")


# ── GUI — optimized for 800×480 7-inch touchscreen ───────────────────────────
class TranscriberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lecture Transcriber")
        self.root.geometry("800x480+0+0")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a1a")
        self.root.overrideredirect(True)  # hide top bar

        self.recording_active = False
        self._build_layout()
        self.update_status()

    def _build_layout(self):
        # Top bar (title + connection)
        top = tk.Frame(self.root, bg="#2a2a2a", height=32)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="🎙  Lecture Transcriber", bg="#2a2a2a", fg="#ffffff",
                 font=("Arial", 11, "bold")).pack(side="left", padx=10, pady=4)
        pill_frame = tk.Frame(top, bg="#2a2a2a")
        pill_frame.pack(side="right", padx=8)
        self.conn_label = tk.Label(pill_frame, text="● No signal", bg="#2a2a2a", fg="#f44336",
                                   font=("Arial", 9, "bold"))
        self.conn_label.pack(side="left", padx=6)
        self.rec_label = tk.Label(pill_frame, text="⏹", bg="#2a2a2a", fg="#666",
                                  font=("Arial", 9, "bold"))
        self.rec_label.pack(side="left", padx=4)
        self.queue_label = tk.Label(pill_frame, text="Q:0", bg="#2a2a2a", fg="#666",
                                    font=("Arial", 9))
        self.queue_label.pack(side="left", padx=4)

        # RMS meter + slider
        meter_row = tk.Frame(self.root, bg="#1a1a1a", height=40)
        meter_row.pack(fill="x", padx=6, pady=(4, 2))
        meter_row.pack_propagate(False)
        tk.Label(meter_row, text="Lvl", fg="#666", bg="#1a1a1a", font=("Arial", 8)).pack(side="left", padx=(0,4))
        self.meter = tk.Canvas(meter_row, height=20, bg="#2a2a2a", highlightthickness=0)
        self.meter.pack(side="left", fill="x", expand=True)
        tk.Label(meter_row, text="Thr", fg="#666", bg="#1a1a1a", font=("Arial", 8)).pack(side="left", padx=(8,2))
        self.thr_disp = tk.Label(meter_row, text=str(VAD_RMS_DEFAULT), fg="#ffa500", bg="#1a1a1a",
                                 font=("Arial", 8, "bold"), width=4)
        self.thr_disp.pack(side="right")
        self.slider = ttk.Scale(meter_row, from_=50, to=2000, orient="horizontal", length=100,
                                command=self.on_threshold_change)
        self.slider.set(VAD_RMS_DEFAULT)
        self.slider.pack(side="right", padx=(0,4))

        # Live preview
        prev_row = tk.Frame(self.root, bg="#222222", height=36)
        prev_row.pack(fill="x", padx=6, pady=(2,0))
        prev_row.pack_propagate(False)
        tk.Label(prev_row, text="LIVE", fg="#555", bg="#222222", font=("Arial",7,"bold")).pack(side="left", padx=(6,4))
        self.live_label = tk.Label(prev_row, text="", bg="#222222", fg="#888888",
                                   font=("Consolas",10,"italic"), anchor="w", justify="left")
        self.live_label.pack(side="left", fill="both", expand=True, padx=(0,6))

        # Transcript
        self.transcript = scrolledtext.ScrolledText(self.root, wrap="word",
                                                    font=("Consolas",11),
                                                    bg="#1e1e1e", fg="#e8e8e8",
                                                    insertbackground="#e8e8e8",
                                                    relief="flat", padx=8, pady=6)
        self.transcript.pack(fill="x", padx=6, pady=(4,0))
        self.transcript.config(height=12)

        # Info panel (hidden by default)
        self.info_panel = tk.Frame(self.root, bg="#2b2b2b", height=120)
        self.info_panel.pack(fill="x", padx=6, pady=(0,6))
        self.info_panel.pack_propagate(False)
        self.info_panel_visible = False  # track visibility
        self.info_text = tk.Label(self.info_panel, text="", fg="#e0e0e0",
                                  bg="#2b2b2b", font=("Arial", 10), justify="left", anchor="nw")
        self.info_text.pack(fill="both", expand=True, padx=8, pady=6)
        self.info_panel.pack_forget()  # hide initially

        # Button bar
        btn_bar = tk.Frame(self.root, bg="#1a1a1a", height=52)
        btn_bar.pack(fill="x", padx=6, pady=(21,6))  # lowered 15px
        btn_bar.pack_propagate(False)
        btn_cfg = dict(font=("Arial",11,"bold"), relief="flat", cursor="hand2", height=2)
        self.record_btn = tk.Button(btn_bar, text="▶  Start", command=self.toggle_recording,
                                    bg="#2e7d32", fg="white", activebackground="#1b5e20",
                                    activeforeground="white", **btn_cfg)
        self.record_btn.pack(side="left", fill="x", expand=True, padx=(0,3))
        tk.Button(btn_bar, text="🗑  Clear", command=self.clear_transcript,
                  bg="#37474f", fg="white", activebackground="#263238", activeforeground="white",
                  **btn_cfg).pack(side="left", fill="x", expand=True, padx=3)
        tk.Button(btn_bar, text="💾  Save", command=self.save_transcript,
                  bg="#37474f", fg="white", activebackground="#263238", activeforeground="white",
                  **btn_cfg).pack(side="left", fill="x", expand=True, padx=3)
        tk.Button(btn_bar, text="ℹ  Info", command=self.toggle_info,
                  bg="#37474f", fg="white", activebackground="#263238", activeforeground="white",
                  **btn_cfg).pack(side="left", fill="x", expand=True, padx=(3,0))

    # Toggle info panel visibility
    def toggle_info(self):
        if self.info_panel_visible:
            self.info_panel.pack_forget()
            self.info_panel_visible = False
        else:
            info_text = (
                "Lecture Transcriber v1.2\n"
                "Designed for real-time transcription on Pi5 / ESP32\n\n"
                "• Start / Stop: control recording\n"
                "• Threshold slider: adjust voice level detection\n"
                "• Clear: remove transcript\n"
                "• Save: export transcript\n"
                "• Info: toggle this panel"
            )
            self.info_text.config(text=info_text)
            self.info_panel.pack(fill="x", padx=6, pady=(0,6))
            self.info_panel_visible = True

    def on_threshold_change(self, val):
        v = int(float(val))
        vad_threshold[0] = v
        self.thr_disp.config(text=str(v))

    def toggle_recording(self):
        if recording.is_set():
            recording.clear()
            with audio_buf_lock:
                audio_buf.clear()
            self.record_btn.config(text="▶  Start", bg="#2e7d32", activebackground="#1b5e20")
            self.rec_label.config(text="⏹", fg="#666")
            self.live_label.config(text="")
            print("\n⏸ Recording stopped")
        else:
            recording.set()
            self.record_btn.config(text="■  Stop", bg="#c62828", activebackground="#b71c1c")
            self.rec_label.config(text="● REC", fg="#4caf50")
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
                f.write(f"Models: {INTERIM_MODEL} / {FINAL_MODEL}\n")
                f.write(f"Rate: {RATE} Hz  |  VAD: {vad_threshold[0]}\n")
                f.write("-"*50+"\n\n")
                f.write(self.transcript.get("1.0", "end-1c"))
            messagebox.showinfo("Saved", f"Saved:\n{fn}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")

    def show_info(self):
        messagebox.showinfo("About", (
            f"INMP441 Live Transcriber\n\n"
            f"Preview : {INTERIM_MODEL}  (every {INTERIM_EVERY}s)\n"
            f"Final   : {FINAL_MODEL}   (after {SILENCE_S}s silence)\n"
            f"Rate    : {RATE} Hz\n"
            f"VAD thr : {vad_threshold[0]}\n\n"
            f"Gray text  = rough live preview\n"
            f"White text = accurate final result\n\n"
            f"Tuning:\n"
            f"  Nothing shows  → lower threshold\n"
            f"  Too noisy      → raise threshold\n"
            f"  Q > 2 (red)    → falling behind"
        ))

    def update_status(self):
        try:
            if connected.is_set():
                self.conn_label.config(text="● ESP32", fg="#4caf50")
            else:
                self.conn_label.config(text="● No signal", fg="#f44336")

            qd = final_queue.qsize()
            self.queue_label.config(
                text=f"Q:{qd}",
                fg="#f44336" if qd > 2 else "#4caf50"
            )

            # RMS bar
            self.meter.delete("all")
            width = self.meter.winfo_width()
            if width < 10:
                width = 780
            bar_width = min(width, 780)
            rms = live_rms[0]
            thr = vad_threshold[0]

            # color: green < thr*0.7, yellow < thr, red >= thr
            if rms < thr*0.7:
                c = "#4caf50"
            elif rms < thr:
                c = "#ffa000"
            else:
                c = "#f44336"

            fill_w = min(rms/thr*bar_width, bar_width)
            self.meter.create_rectangle(0,0,fill_w,20,fill=c, width=0)
            self.meter.create_rectangle(fill_w,0,bar_width,20,fill="#444", width=0)

            # overlay threshold line
            thr_x = min(thr/bar_width*bar_width, bar_width)
            self.meter.create_line(thr,0,thr,20,fill="#ffa500", width=2)

            # Live preview text
            self.live_label.config(text=interim_result[0])

            # Apply final queued transcripts
            while not final_queue_gui.empty():
                txt = final_queue_gui.get()
                self.transcript.config(state="normal")
                self.transcript.insert("end", txt + "\n")
                self.transcript.config(state="disabled")
                self.transcript.yview_moveto(1.0)

        except Exception as e:
            print(f"❌ GUI update error: {e}")

        self.root.after(200, self.update_status)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=udp_thread, daemon=True).start()
    threading.Thread(target=interim_thread, daemon=True).start()
    threading.Thread(target=final_transcribe_thread, daemon=True).start()

    root = tk.Tk()
    gui  = TranscriberGUI(root)
    root.mainloop()