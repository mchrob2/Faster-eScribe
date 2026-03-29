"""
Pico W Audio Streamer - code.py (Improved + Fixed Binding)
===========================================================
- Creates a WiFi Access Point
- Binds UDP socket to receive HELLO from Pi5
- Samples MAX4466 mic at 16000 Hz via ADC
- Sends 16-bit linear PCM over UDP to connected Pi5
"""

import wifi
import socketpool
import analogio
import board
import time
import struct
import microcontroller

# ─── Configuration ────────────────────────────────────────────────────────────

AP_SSID = "LectureAudio"
AP_PASSWORD = "transcribe123"      # min 8 chars for WPA2
UDP_PORT = 5005
SAMPLE_RATE = 16000                 # Hz
PACKET_SAMPLES = 320                 # 20 ms at 16 kHz

# ─── Start Access Point ───────────────────────────────────────────────────────

print("Starting Access Point...")
wifi.radio.stop_station()
wifi.radio.start_ap(ssid=AP_SSID, password=AP_PASSWORD, channel=6)
print(f"AP started: SSID='{AP_SSID}' IP={wifi.radio.ipv4_address_ap}")
print(f"Connect Pi5 to this network, then it will receive audio on UDP port {UDP_PORT}")

# Give the AP a moment to stabilise
time.sleep(1)

# ─── Setup UDP socket and bind to receive HELLO ───────────────────────────────

pool = socketpool.SocketPool(wifi.radio)
sock = pool.socket(pool.AF_INET, pool.SOCK_DGRAM)
sock.settimeout(0)                  # non-blocking

# Bind to the same port we expect to receive HELLO on
sock.bind(("0.0.0.0", UDP_PORT))
print(f"Socket bound to port {UDP_PORT}")

# ─── Setup ADC ────────────────────────────────────────────────────────────────

adc = analogio.AnalogIn(board.GP26)
ADC_MIDPOINT = 32768                 # 0-65535, centre ~32768

# ─── Packet header: sequence number (2 bytes) + sample rate (2 bytes) ─────────
# Total packet = 4 bytes header + (PACKET_SAMPLES * 2) bytes audio (int16)

seq = 0
packet_buf = bytearray(4 + PACKET_SAMPLES * 2)

# ─── Sampling timing (using sleep + drift compensation) ──────────────────────

sample_interval = 1.0 / SAMPLE_RATE   # seconds between samples

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
        # No data yet (non‑blocking socket)
        pass
    time.sleep(0.1)

print("Starting audio stream...")

# ─── Main streaming loop ──────────────────────────────────────────────────────

while True:
    # Plan the sampling times for this packet
    t_start = time.monotonic()
    for i in range(PACKET_SAMPLES):
        # Read ADC and convert to signed int16
        raw = adc.value
        signed = raw - ADC_MIDPOINT          # now in range -32768..32767

        # Pack as little‑endian int16 into the buffer
        struct.pack_into("<h", packet_buf, 4 + i * 2, signed)

        # Wait until the exact sampling time for this sample
        target = t_start + i * sample_interval
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)

    # All samples collected – add header and send
    struct.pack_into(">HH", packet_buf, 0, seq & 0xFFFF, SAMPLE_RATE)
    seq += 1

    try:
        sock.sendto(packet_buf, (pi5_addr, UDP_PORT))
    except OSError as e:
        print(f"Send error: {e}")

    # Check for any re‑connection HELLO from Pi5 (in case it restarted)
    try:
        nbytes, addr = sock.recvfrom_into(hello_buf)
        msg = bytes(hello_buf[:nbytes])
        if msg.startswith(b"HELLO"):
            pi5_addr = addr[0]
            print(f"Pi5 reconnected from {pi5_addr}")
    except OSError:
        pass
