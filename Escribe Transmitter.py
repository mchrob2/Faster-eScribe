# SPDX-FileCopyrightText: 2025 Your Name
# SPDX-License-Identifier: MIT
"""
RFM69 Audio Transmitter – Low Latency (14 samples/packet)
"""

import board
import busio
import digitalio
import analogio
import time
import struct
import array
from rfm69 import RFM69

# ===== CONFIGURATION =====
RADIO_FREQ_MHZ = 915.0
NODE_ID = 1
DEST_ID = 2
ENCRYPTION_KEY = b"\x01\x02\x03\x04\x05\x06\x07\x08\x01\x02\x03\x04\x05\x06\x07\x08"
SAMPLE_RATE = 2000
SAMPLES_PER_PACKET = 14                     # reduced from 28 → 7 ms audio per packet
PACKET_SIZE = 4 + (SAMPLES_PER_PACKET * 2)  # 32 bytes
TARGET_INTERVAL = 1.0 / (SAMPLE_RATE / SAMPLES_PER_PACKET)   # 7 ms

print("=" * 60)
print("RFM69 AUDIO TRANSMITTER – LOW LATENCY (14 samples/packet)")
print("=" * 60)

# ===== PIN SETUP =====
RFM_CS = digitalio.DigitalInOut(board.GP17)
RFM_RST = digitalio.DigitalInOut(board.GP20)
RFM_INT = digitalio.DigitalInOut(board.GP21)   # (unused)

mic = analogio.AnalogIn(board.GP26)
led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT
led.value = False

print(f"Microphone: GP26 | LED: onboard")

# ===== SPI SETUP (10 MHz for lower overhead) =====
spi = busio.SPI(board.GP18, MOSI=board.GP19, MISO=board.GP16)
while not spi.try_lock():
    pass
spi.configure(baudrate=10_000_000, phase=0, polarity=0)  # 10 MHz (RFM69 max)
spi.unlock()
print("SPI at 10 MHz")

# ===== RADIO INITIALIZATION =====
print("Initializing RFM69...")
rf69 = RFM69(spi, RFM_CS, RFM_RST, RADIO_FREQ_MHZ, baudrate=2_000_000)

rf69.tx_power = 20
rf69.encryption_key = ENCRYPTION_KEY
rf69.node = NODE_ID
rf69.destination = DEST_ID

# Disable all ACK/retry features
rf69.ack_retries = 0
rf69.ack_wait = 0
rf69.ack_delay = None
rf69.receive_timeout = 0.1
rf69.xmit_timeout = 0.05               # packet takes <1 ms

# Optional: increase FIFO threshold
rf69.fifo_threshold = 20

print(f"TX Power: {rf69.tx_power} dBm")
print(f"Packet size: {PACKET_SIZE} bytes")
print(f"Target packet interval: {TARGET_INTERVAL*1000:.2f} ms")

# ===== PRE-ALLOCATE BUFFERS =====
packet_buffer = bytearray(PACKET_SIZE)
audio_buffer = array.array('H', [0] * SAMPLES_PER_PACKET)

# ===== SEND TEST PING =====
rf69.idle()
time.sleep(0.05)
rf69.send(b"PING")
print("Test ping sent.\n")

# ===== MAIN LOOP =====
packet_count = 0
start_time = None
last_print_time = 0
packet_errors = 0

# For precise sampling: use time.monotonic_ns() if available
try:
    time.monotonic_ns
    has_ns = True
except AttributeError:
    has_ns = False
    print("Warning: monotonic_ns not available; using rough delays.")

print("\nStarting audio transmission...\n")

while True:
    try:
        # ----- Sample acquisition (precise) -----
        if has_ns:
            # 500 µs between samples using busy-wait
            next_sample_ns = time.monotonic_ns()
            for i in range(SAMPLES_PER_PACKET):
                audio_buffer[i] = mic.value
                next_sample_ns += 500_000          # 500 µs
                while time.monotonic_ns() < next_sample_ns:
                    pass
        else:
            # Fallback: simple sleep (less accurate)
            for i in range(SAMPLES_PER_PACKET):
                audio_buffer[i] = mic.value
                if i < SAMPLES_PER_PACKET - 1:
                    time.sleep(0.0005)             # 500 µs

        # ----- Build packet without allocation -----
        packet_count += 1
        struct.pack_into('<I', packet_buffer, 0, packet_count)
        struct.pack_into('<' + str(SAMPLES_PER_PACKET) + 'H',
                         packet_buffer, 4, *audio_buffer)

        # ----- Send (radio already idle) -----
        success = rf69.send(packet_buffer, keep_listening=False)

        # Simple LED blink
        led.value = True
        led.value = False

        # Record start time after first successful send
        if packet_count == 1 and success:
            start_time = time.monotonic()

        # ----- Statistics every 100 packets -----
        if packet_count % 100 == 0:
            current = time.monotonic()
            if start_time is not None:
                rate = packet_count / (current - start_time)
                audio_pp = max(audio_buffer) - min(audio_buffer)
                print(f"TX #{packet_count:6d} | rate {rate:5.1f}/s | "
                      f"audio pp {audio_pp:5d} | errors {packet_errors}")
            last_print_time = current

        # ----- Maintain exact packet rate -----
        if start_time is not None:
            next_deadline = start_time + packet_count * TARGET_INTERVAL
            now = time.monotonic()
            if now < next_deadline:
                time.sleep(next_deadline - now)

    except KeyboardInterrupt:
        print("\n\nTransmitter stopped.")
        break
    except Exception as e:
        packet_errors += 1
        print(f"Error: {e}")
        time.sleep(0.01)

# ===== CLEANUP =====
rf69.idle()
rf69.sleep()
mic.deinit()
print(f"Total packets sent: {packet_count}, errors: {packet_errors}")
