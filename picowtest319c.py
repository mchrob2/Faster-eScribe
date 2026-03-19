"""
Pico W Audio Streamer — Low Latency Version
============================================
Changes vs previous:
  • PACKET_SAMPLES = 160  →  10 ms packets (was 320 / 20 ms)
    Cuts the per-packet accumulation delay in half.
  • Sample at START of each interval (was end) for tighter timing.
  • Stripped microcontroller import (unused).

Wiring:
    MAX4466 OUT → GP26 (ADC0)
    MAX4466 VCC → 3.3V
    MAX4466 GND → GND

Requirements:
    CircuitPython 9.x on Pico W
"""

import wifi
import socketpool
import analogio
import board
import time
import struct

# ─── Configuration ────────────────────────────────────────────────────────────

AP_SSID        = "LectureAudio"
AP_PASSWORD    = "transcribe123"   # min 8 chars for WPA2
UDP_PORT       = 5005
SAMPLE_RATE    = 16000             # Hz
PACKET_SAMPLES = 160               # ← 10 ms at 16 kHz  (was 320 / 20 ms)

# ─── Start Access Point ───────────────────────────────────────────────────────

print("Starting Access Point...")
wifi.radio.stop_station()
wifi.radio.start_ap(ssid=AP_SSID, password=AP_PASSWORD)
print(f"AP up  SSID='{AP_SSID}'  IP={wifi.radio.ipv4_address_ap}")
time.sleep(1)  # let AP stabilise

# ─── Socket ───────────────────────────────────────────────────────────────────

pool = socketpool.SocketPool(wifi.radio)
sock = pool.socket(pool.AF_INET, pool.SOCK_DGRAM)
sock.settimeout(0)                 # non-blocking
sock.bind(("0.0.0.0", UDP_PORT))
print(f"Bound to port {UDP_PORT}, waiting for Pi5 HELLO...")

# ─── ADC ──────────────────────────────────────────────────────────────────────

adc          = analogio.AnalogIn(board.GP26)
ADC_MIDPOINT = 32768               # 0-65535 -> signed centre

# ─── Buffers ──────────────────────────────────────────────────────────────────
# Packet layout:
#   [0:2]  seq number  (uint16 big-endian)
#   [2:4]  sample rate (uint16 big-endian)
#   [4: ]  PACKET_SAMPLES x int16 little-endian  (320 bytes)

seq        = 0
packet_buf = bytearray(4 + PACKET_SAMPLES * 2)
hello_buf  = bytearray(16)

# ─── Timing ───────────────────────────────────────────────────────────────────

sample_interval = 1.0 / SAMPLE_RATE   # 62.5 us between samples

# ─── Wait for Pi5 HELLO ───────────────────────────────────────────────────────

pi5_addr = None
print("Waiting for Pi5 HELLO...")

while pi5_addr is None:
    try:
        nbytes, addr = sock.recvfrom_into(hello_buf)
        if bytes(hello_buf[:nbytes]).startswith(b"HELLO"):
            pi5_addr = addr[0]
            print(f"Pi5 connected from {pi5_addr}")
    except OSError:
        pass
    time.sleep(0.05)

print("Streaming audio (10 ms packets)...")

# ─── Main streaming loop ──────────────────────────────────────────────────────

while True:
    t_start = time.monotonic()

    for i in range(PACKET_SAMPLES):
        # Sleep until it is time for this sample, THEN read ADC.
        # Sampling at the START of the interval keeps timing consistent.
        target = t_start + i * sample_interval
        now    = time.monotonic()
        if target > now:
            time.sleep(target - now)

        raw    = adc.value
        signed = raw - ADC_MIDPOINT
        struct.pack_into("<h", packet_buf, 4 + i * 2, signed)

    # Write header and transmit
    struct.pack_into(">HH", packet_buf, 0, seq & 0xFFFF, SAMPLE_RATE)
    seq += 1

    try:
        sock.sendto(packet_buf, (pi5_addr, UDP_PORT))
    except OSError as e:
        print(f"Send error: {e}")

    # Check for reconnect HELLO (e.g. Pi5 restarted)
    try:
        nbytes, addr = sock.recvfrom_into(hello_buf)
        if bytes(hello_buf[:nbytes]).startswith(b"HELLO"):
            pi5_addr = addr[0]
            print(f"Pi5 reconnected from {pi5_addr}")
    except OSError:
        pass
