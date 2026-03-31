#!/usr/bin/env python3
"""
Pi5 Transcription Receiver for ICS-43434 Microphone
Receives UDP audio from ESP32 and transcribes using faster-whisper

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
import os
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────
HOST        = "0.0.0.0"
PORT        = 5005
RATE        = 16000          # Hz — matches ESP32 I2S rate (Low-Power Mode)
VAD_RMS_DEFAULT = 800        # raw int16 RMS; adjust based on actual levels
SILENCE_S   = 1.5            # seconds of quiet → flush segment
MAX_SEG_S   = 10.0           # force flush after this duration regardless

# ── Shared state ───────────────────────────────────────────────────────────
vad_threshold = [VAD_RMS_DEFAULT]
audio_buf     = []
last_voice    = [0.0]
live_rms      = [0]
packet_count  = [0]
bytes_received = [0]
last_packet_time = [time.time()]
audio_stats = {
   'peak': 0,
   'min': 0,
   'max': 0,
   'avg': 0
}

recording  = threading.Event()
connected  = threading.Event()

seg_queue    = queue.Queue()
result_queue = queue.Queue()

# ── Load Whisper ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Pi5 Transcription Receiver for ICS-43434 Microphone")
print("="*60)
print("Loading Whisper model...")

# Use tiny.en for faster performance, can change to base.en for better accuracy
MODEL_SIZE = "tiny.en"  # Options: tiny.en, base.en, small.en
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print(f"✓ Model '{MODEL_SIZE}' loaded successfully")
print(f"✓ Sample rate: {RATE} Hz")
print(f"✓ VAD threshold: {VAD_RMS_DEFAULT}")
print(f"✓ Silence timeout: {SILENCE_S}s")
print(f"✓ Max segment: {MAX_SEG_S}s\n")

# ── UDP receiver thread ─────────────────────────────────────────────────────
def udp_thread():
   """Receives UDP packets and performs VAD"""
   try:
       sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
       sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
       sock.bind((HOST, PORT))
       sock.settimeout(1.0)
       print(f"✓ Listening on UDP {HOST}:{PORT}")
       print("Waiting for ESP32 audio stream...\n")

       packet_errors = 0

       while True:
           try:
               data, addr = sock.recvfrom(8192)
               packet_count[0] += 1
               bytes_received[0] += len(data)
               last_packet_time[0] = time.time()
               connected.set()
               packet_errors = 0

               # Debug first packet only
               if packet_count[0] == 1:
                   print(f"✓ First packet received from {addr[0]}:{addr[1]}")
                   print(f"  Size: {len(data)} bytes")

                   # Analyze first packet
                   samples = np.frombuffer(data, dtype=np.int16)
                   print(f"  Samples: {len(samples)}")
                   print(f"  Range: {samples.min()} to {samples.max()}")
                   print(f"  RMS: {np.sqrt(np.mean(samples.astype(np.float32)**2)):.2f}")
                   print()

               # Only process if recording is enabled
               if not recording.is_set():
                   continue

               # Convert to int16 samples (already in correct format)
               samples = np.frombuffer(data, dtype=np.int16).copy()

               # Update statistics
               audio_stats['peak'] = np.abs(samples).max()
               audio_stats['min'] = samples.min()
               audio_stats['max'] = samples.max()
               audio_stats['avg'] = samples.mean()

               # Calculate RMS
               rms = int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
               live_rms[0] = rms

               # Voice Activity Detection
               if rms >= vad_threshold[0]:
                   if not last_voice[0]:
                       print(f"🎤 Voice detected (RMS={rms})")
                   last_voice[0] = time.time()
                   audio_buf.append(samples)
               elif audio_buf:
                   # Keep trailing silence for context
                   audio_buf.append(samples)
                   silence = time.time() - last_voice[0]
                   total_s = sum(len(a) for a in audio_buf) / RATE

                   # Flush when silence is long enough or segment is too long
                   if silence >= SILENCE_S or total_s >= MAX_SEG_S:
                       _flush_segment()

           except socket.timeout:
               packet_errors += 1
               if packet_errors > 3:
                   connected.clear()
                   if packet_count[0] > 0 and packet_errors % 10 == 0:
                       print("⚠ No packets received for 3+ seconds")

           except Exception as e:
               print(f"❌ UDP receive error: {e}")

   except Exception as e:
       print(f"❌ UDP thread fatal error: {e}")
       sys.exit(1)

def _flush_segment():
   """Flush audio buffer to transcription queue"""
   if not audio_buf:
       return

   # Concatenate all audio chunks
   audio = np.concatenate(audio_buf).astype(np.float32) / 32768.0
   duration = len(audio) / RATE
   audio_buf.clear()
   last_voice[0] = 0.0

   print(f"📦 Flushing segment: {len(audio)} samples ({duration:.2f}s)")
   seg_queue.put(audio)

# ── Transcription thread ────────────────────────────────────────────────────
def transcribe_thread():
   """Transcribes audio segments using faster-whisper"""
   while True:
       try:
           audio = seg_queue.get()
           if audio is None:
               break

           start_time = time.time()
           print(f"🎤 Transcribing {len(audio)} samples...", end=" ", flush=True)

           # Transcribe with VAD enabled
           segments, info = model.transcribe(
               audio,
               language="en",
               vad_filter=True,
               vad_parameters={
                   "min_silence_duration_ms": 300,
                   "threshold": 0.5,
                   "neg_threshold": 0.2
               },
               beam_size=5,
               best_of=5
           )

           # Collect all segments
           text_parts = []
           for segment in segments:
               text_parts.append(segment.text.strip())
               print(f"\n  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

           text = " ".join(text_parts)
           elapsed = time.time() - start_time

           if text:
               print(f"✓ Transcription ({elapsed:.2f}s): {text[:100]}...")
               result_queue.put(text)
           else:
               print(f"⏭ No speech detected ({elapsed:.2f}s)")

       except Exception as e:
           print(f"❌ Transcription error: {e}")

# ── GUI ─────────────────────────────────────────────────────────────────────
class TranscriberGUI:
   def __init__(self, root):
       self.root = root
       self.root.title("ICS-43434 Microphone Transcriber")
       self.root.geometry("900x650")
       self.root.configure(bg="#1e1e1e")

       # Set icon if available
       try:
           self.root.iconbitmap("microphone.ico")
       except:
           pass

       self.setup_ui()

       # Start periodic updates
       self.update_status()

   def setup_ui(self):
       """Setup all UI elements"""
       # Title bar
       title_frame = tk.Frame(self.root, bg="#2d2d2d", height=40)
       title_frame.pack(fill="x")
       title_frame.pack_propagate(False)

       title_label = tk.Label(
           title_frame,
           text="🎙️ ICS-43434 I2S Microphone Transcriber",
           bg="#2d2d2d",
           fg="#ffffff",
           font=("Arial", 14, "bold")
       )
       title_label.pack(pady=8)

       # Status bar
       status_frame = tk.Frame(self.root, bg="#252525", height=60)
       status_frame.pack(fill="x", padx=10, pady=(10, 5))
       status_frame.pack_propagate(False)

       # Connection status
       self.conn_label = tk.Label(
           status_frame, text="● Disconnected", font=("Arial", 10, "bold"),
           fg="#888", bg="#252525"
       )
       self.conn_label.place(x=10, y=10)

       # Packet counter
       self.packet_label = tk.Label(
           status_frame, text="Packets: 0", font=("Arial", 10),
           fg="#aaa", bg="#252525"
       )
       self.packet_label.place(x=10, y=32)

       # Bytes received
       self.bytes_label = tk.Label(
           status_frame, text="Data: 0 KB", font=("Arial", 10),
           fg="#aaa", bg="#252525"
       )
       self.bytes_label.place(x=150, y=32)

       # Recording status
       self.rec_label = tk.Label(
           status_frame, text="⏹ Stopped", font=("Arial", 12, "bold"),
           fg="#888", bg="#252525"
       )
       self.rec_label.place(x=400, y=10)

       # Audio stats
       self.stats_label = tk.Label(
           status_frame, text="Peak: 0", font=("Arial", 9),
           fg="#aaa", bg="#252525"
       )
       self.stats_label.place(x=400, y=35)

       # RMS Meter frame
       meter_frame = tk.Frame(self.root, bg="#1e1e1e")
       meter_frame.pack(fill="x", padx=10, pady=5)

       # Level label
       tk.Label(
           meter_frame, text="Audio Level", fg="#aaa", bg="#1e1e1e",
           font=("Arial", 10)
       ).pack(side="left", padx=(0, 10))

       # Meter canvas
       self.meter = tk.Canvas(meter_frame, height=20, bg="#2d2d2d",
                              highlightthickness=0, width=400)
       self.meter.pack(side="left", fill="x", expand=True)

       # Threshold slider
       tk.Label(
           meter_frame, text="Threshold", fg="#aaa", bg="#1e1e1e",
           font=("Arial", 10)
       ).pack(side="left", padx=(10, 5))

       self.thr_disp = tk.Label(
           meter_frame, text=str(VAD_RMS_DEFAULT), fg="#ffa500",
           bg="#1e1e1e", font=("Arial", 10, "bold"), width=5
       )
       self.thr_disp.pack(side="right", padx=(0, 5))

       self.slider = ttk.Scale(
           meter_frame, from_=50, to=5000,
           orient="horizontal", length=120,
           command=self.on_threshold_change
       )
       self.slider.set(VAD_RMS_DEFAULT)
       self.slider.pack(side="right")

       # Transcript area
       transcript_frame = tk.Frame(self.root, bg="#1e1e1e")
       transcript_frame.pack(fill="both", expand=True, padx=10, pady=5)

       tk.Label(
           transcript_frame, text="Transcript", fg="#aaa", bg="#1e1e1e",
           font=("Arial", 10, "bold")
       ).pack(anchor="w")

       self.transcript = scrolledtext.ScrolledText(
           transcript_frame, wrap="word",
           font=("Consolas", 12),
           bg="#2d2d2d", fg="#e8e8e8",
           insertbackground="#e8e8e8",
           relief="flat", padx=10, pady=10,
           height=15
       )
       self.transcript.pack(fill="both", expand=True, pady=(5, 0))

       # Buttons
       btn_frame = tk.Frame(self.root, bg="#1e1e1e")
       btn_frame.pack(fill="x", padx=10, pady=10)

       self.record_btn = tk.Button(
           btn_frame, text="▶ Start Recording", command=self.toggle_recording,
           bg="#2e7d32", fg="white", font=("Arial", 12, "bold"),
           relief="flat", padx=20, pady=8, cursor="hand2",
           activebackground="#1b5e20", activeforeground="white"
       )
       self.record_btn.pack(side="left", padx=5)

       tk.Button(
           btn_frame, text="🗑 Clear", command=self.clear_transcript,
           bg="#424242", fg="white", font=("Arial", 11),
           relief="flat", padx=15, pady=8, cursor="hand2",
           activebackground="#616161", activeforeground="white"
       ).pack(side="left", padx=5)

       tk.Button(
           btn_frame, text="💾 Save", command=self.save_transcript,
           bg="#424242", fg="white", font=("Arial", 11),
           relief="flat", padx=15, pady=8, cursor="hand2",
           activebackground="#616161", activeforeground="white"
       ).pack(side="left", padx=5)

       tk.Button(
           btn_frame, text="ℹ️ Info", command=self.show_info,
           bg="#424242", fg="white", font=("Arial", 11),
           relief="flat", padx=15, pady=8, cursor="hand2",
           activebackground="#616161", activeforeground="white"
       ).pack(side="right", padx=5)

   def on_threshold_change(self, val):
       """Handle threshold slider change"""
       v = int(float(val))
       vad_threshold[0] = v
       self.thr_disp.config(text=str(v))

   def toggle_recording(self):
       """Start/stop recording"""
       if recording.is_set():
           recording.clear()
           audio_buf.clear()
           self.record_btn.config(text="▶ Start Recording", bg="#2e7d32")
           self.rec_label.config(text="⏹ Stopped", fg="#888")
           print("\n⏸ Recording stopped")
       else:
           recording.set()
           self.record_btn.config(text="■ Stop Recording", bg="#c62828")
           self.rec_label.config(text="● Recording", fg="#4caf50")
           print("\n🎙 Recording started - Speak into the microphone")

   def clear_transcript(self):
       """Clear transcript text"""
       self.transcript.config(state="normal")
       self.transcript.delete("1.0", "end")
       self.transcript.config(state="disabled")
       print("✓ Transcript cleared")

   def save_transcript(self):
       """Save transcript to file"""
       try:
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           filename = f"transcript_{timestamp}.txt"

           with open(filename, "w") as f:
               f.write(f"ICS-43434 Transcription\n")
               f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
               f.write(f"Sample Rate: {RATE} Hz\n")
               f.write(f"VAD Threshold: {vad_threshold[0]}\n")
               f.write("-" * 50 + "\n\n")
               text = self.transcript.get("1.0", "end-1c")
               f.write(text)

           messagebox.showinfo("Success", f"Transcript saved to:\n{filename}")
           print(f"✓ Transcript saved to {filename}")
       except Exception as e:
           messagebox.showerror("Error", f"Failed to save: {e}")
           print(f"❌ Error saving: {e}")

   def show_info(self):
       """Show information dialog"""
       info = f"""ICS-43434 I2S Microphone Transcriber

Microphone: ICS-43434 Digital MEMS
Sample Rate: {RATE} Hz (16 kHz)
Format: 24-bit I2S → 16-bit PCM
UDP Port: {PORT}
Model: {MODEL_SIZE}
VAD Threshold: {vad_threshold[0]}

How it works:
1. ESP32 captures audio from ICS-43434
2. Audio sent via UDP to Pi5
3. faster-whisper transcribes in real-time
4. Results displayed with timestamps

Tips:
- Adjust threshold for better voice detection
- Speak clearly at normal volume
- Background noise may affect accuracy
"""
       messagebox.showinfo("About", info)

   def update_status(self):
       """Update status indicators periodically"""
       try:
           # Update connection status
           if connected.is_set():
               self.conn_label.config(text="● Connected", fg="#4caf50")
           else:
               self.conn_label.config(text="● Disconnected", fg="#f44336")

           # Update packet counter
           self.packet_label.config(text=f"Packets: {packet_count[0]}")
           self.bytes_label.config(text=f"Data: {bytes_received[0]/1024:.0f} KB")

           # Update audio stats
           if audio_stats['peak'] > 0:
               self.stats_label.config(
                   text=f"Peak: {audio_stats['peak']} | "
                        f"Min: {audio_stats['min']} | "
                        f"Max: {audio_stats['max']}"
               )

           # Update RMS meter
           rms = live_rms[0]
           thr = vad_threshold[0]
           w = self.meter.winfo_width()

           if w > 1:
               self.meter.delete("all")

               # Draw background
               self.meter.create_rectangle(0, 0, w, 20, fill="#2d2d2d", outline="")

               # Draw RMS bar
               bar_w = min(int(w * rms / 5000), w)
               if rms >= thr:
                   color = "#4caf50"
               else:
                   color = "#ff9800"
               self.meter.create_rectangle(0, 0, bar_w, 20, fill=color, outline="")

               # Draw threshold marker
               tx = max(1, min(int(w * thr / 5000), w - 1))
               self.meter.create_line(tx, 0, tx, 20, fill="#f44336", width=2)

               # Add text labels
               self.meter.create_text(5, 10, text=f"{rms}",
                                     fill="white", font=("Arial", 8), anchor="w")
               self.meter.create_text(w-5, 10, text=f"{thr}",
                                     fill="white", font=("Arial", 8), anchor="e")

           # Update transcript with new results
           new_texts = []
           while not result_queue.empty():
               text = result_queue.get_nowait()
               new_texts.append(text)

           if new_texts:
               self.transcript.config(state="normal")
               for text in new_texts:
                   timestamp = datetime.now().strftime("%H:%M:%S")
                   self.transcript.insert("end", f"[{timestamp}] {text}\n\n")
               self.transcript.see("end")
               self.transcript.config(state="disabled")

       except Exception as e:
           print(f"UI update error: {e}")

       # Schedule next update
       self.root.after(100, self.update_status)

# ── Main entry point ─────────────────────────────────────────────────────────
def main():
   """Main application entry point"""
   # Check dependencies
   try:
       import faster_whisper
       import numpy
   except ImportError as e:
       print(f"❌ Missing dependency: {e}")
       print("\nInstall required packages:")
       print("  pip3 install faster-whisper numpy")
       sys.exit(1)

   # Start threads
   udp_thread_handle = threading.Thread(target=udp_thread, daemon=True)
   transcribe_thread_handle = threading.Thread(target=transcribe_thread, daemon=True)

   udp_thread_handle.start()
   transcribe_thread_handle.start()

   # Start GUI
   root = tk.Tk()
   app = TranscriberGUI(root)

   print("\n" + "="*60)
   print("System Ready!")
   print("="*60)
   print("1. Ensure ESP32 is running and connected")
   print("2. Pi5 must be connected to 'LectureAudio' WiFi")
   print("3. Click 'Start Recording' to begin transcription")
   print("4. Speak into the microphone")
   print("="*60 + "\n")

   # Handle window close
   def on_closing():
       print("\nShutting down...")
       seg_queue.put(None)  # Signal transcription thread to exit
       root.destroy()
       sys.exit(0)

   root.protocol("WM_DELETE_WINDOW", on_closing)
   root.mainloop()

if __name__ == "__main__":
   main()