"""
Pico W Audio Streamer - code.py (Optimized Hardware Sampling)
===========================================================
- Creates a WiFi Access Point
- Binds UDP socket to receive HELLO from Pi5
- Uses analogbufio to sample MAX4466 mic at exactly 16000 Hz
- Sends 16-bit unsigned PCM over UDP to connected Pi5
"""

import wifi
import socketpool
import analogbufio
import board
import time
import struct

# ─── Configuration ────────────────────────────────────────────────────────────

AP_SSID = "LectureAudio"
AP_PASSWORD = "transcribe123"      # min 8 chars for WPA2
UDP_PORT = 5005
SAMPLE_RATE = 16000                # Hz
PACKET_SAMPLES = 320               # 20 ms at 16 kHz

# ─── Start Access Point ───────────────────────────────────────────────────────

print("Starting Access Point...")
wifi.radio.stop_station()
wifi.radio.start_ap(ssid=AP_SSID, password=AP_PASSWORD)
print(f"AP started: SSID='{AP_SSID}' IP={wifi.radio.ipv4_address_ap}")
print(f"Connect Pi5 to this network, then it will receive audio on UDP port {UDP_PORT}")

time.sleep(1) # Give the AP a moment to stabilise

# ─── Setup UDP socket ─────────────────────────────────────────────────────────

pool = socketpool.SocketPool(wifi.radio)
sock = pool.socket(pool.AF_INET, pool.SOCK_DGRAM)
sock.settimeout(0)                 # non-blocking
sock.bind(("0.0.0.0", UDP_PORT))
print(f"Socket bound to port {UDP_PORT}")

# ─── Setup Hardware ADC Sampler ───────────────────────────────────────────────

# analogbufio uses DMA to sample without CPU blocking (returns unsigned 16-bit)
sampler = analogbufio.BufferedIn(board.GP26, sample_rate=SAMPLE_RATE)

# Create a single zero-allocation payload buffer to prevent garbage collection pauses
full_payload = bytearray(4 + PACKET_SAMPLES * 2)
audio_view = memoryview(full_payload)[4:] # View into the payload for the sampler to fill
seq = 0

# ─── Wait for Pi5 HELLO ───────────────────────────────────────────────────────

pi5_addr = None
hello_buf = bytearray(16)
print("Waiting for Pi5 to connect and send HELLO...")

while pi5_addr is None:
    try:
        nbytes, addr = sock.recvfrom_into(hello_buf)
        msg = bytes(hello_buf[:nbytes])
        if msg.startswith(b"HELLO"):
            pi5_addr = addr[0]
            print(f"Pi5 connected from {pi5_addr}")
    except OSError:
        pass
    time.sleep(0.1)

print("Starting audio stream...")

# ─── Main streaming loop ──────────────────────────────────────────────────────

while True:
    # 1. Let hardware fill the buffer perfectly at 16kHz (blocks until full)
    sampler.readinto(audio_view)

    # 2. Pack the header (Sequence + Sample Rate)
    struct.pack_into(">HH", full_payload, 0, seq & 0xFFFF, SAMPLE_RATE)
    seq += 1

    # 3. Fire it off
    try:
        sock.sendto(full_payload, (pi5_addr, UDP_PORT))
    except OSError as e:
        print(f"Send error: {e}")

    # 4. Briefly check for re‑connection HELLO from Pi5
    try:
        nbytes, addr = sock.recvfrom_into(hello_buf)
        msg = bytes(hello_buf[:nbytes])
        if msg.startswith(b"HELLO"):
            pi5_addr = addr[0]
            print(f"Pi5 reconnected from {pi5_addr}")
    except OSError:
        pass
