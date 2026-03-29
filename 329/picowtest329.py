"""
Pico W Audio Streamer - code.py (Battery-Optimised)
=====================================================

Improvements over original:
  1. ADC averaging (4 samples) to reduce WiFi/RF noise on ADC lines
  2. Hardware watchdog -- self-recovers from WiFi hangs or brown-outs
  3. Silence gating -- skips sendto() during silence to save battery
  4. Drift-corrected sample timing using ticks_us instead of time.sleep()
  5. Periodic HELLO broadcast so Pi5 can reconnect after a restart
     without needing to reboot the Pico

Hardware:
  - Raspberry Pi Pico W
  - MAX4466 mic on GP26 (ADC0)
  - Battery supply (LiPo / USB power bank)
"""

import wifi
import socketpool
import analogio
import board
import time
import struct
import microcontroller
from microcontroller import watchdog as wdt
from watchdog import WatchDogMode

# ─── Configuration ────────────────────────────────────────────────────────────

AP_SSID       = "LectureAudio"
AP_PASSWORD   = "transcribe123"
UDP_PORT      = 5005
SAMPLE_RATE   = 16000             # Hz
PACKET_SAMPLES = 320              # 20 ms per packet at 16 kHz

# ADC noise reduction
ADC_AVG_N     = 4                 # samples to average per reading
                                  # higher = less noise, slightly more CPU
ADC_MIDPOINT  = 32768             # centre of 0-65535 ADC range

# Silence gating (battery saving)
# Raise SILENCE_THRESHOLD if speech is being gated out
# Lower it if background noise is preventing sleep
SILENCE_THRESHOLD = 400           # peak deviation from midpoint to count as sound
SILENCE_PACKETS   = 50            # consecutive silent packets before stopping TX
                                  # 50 packets = ~1 second of silence

# Reconnection
HELLO_INTERVAL_SEC = 5.0          # re-broadcast HELLO if no Pi5 seen recently

# Watchdog
WDT_TIMEOUT_SEC = 8               # seconds -- reboot if main loop hangs

# ─── Watchdog ─────────────────────────────────────────────────────────────────
# CircuitPython watchdog API (different from MicroPython's machine.WDT)

wdt.timeout = WDT_TIMEOUT_SEC
wdt.mode    = WatchDogMode.RESET  # reboot the Pico if not fed in time
print("Watchdog started (8s timeout)")

# ─── Start Access Point ───────────────────────────────────────────────────────

print("Starting Access Point...")
wifi.radio.stop_station()
wifi.radio.start_ap(ssid=AP_SSID, password=AP_PASSWORD)
print(f"AP started: SSID='{AP_SSID}'  IP={wifi.radio.ipv4_address_ap}")
print(f"Pi5 should connect to this network and send HELLO to UDP port {UDP_PORT}")
wdt.feed()

time.sleep(1)

# ─── Setup UDP socket ─────────────────────────────────────────────────────────

pool = socketpool.SocketPool(wifi.radio)
sock = pool.socket(pool.AF_INET, pool.SOCK_DGRAM)
sock.settimeout(0)
sock.bind(("0.0.0.0", UDP_PORT))
print(f"Socket bound to 0.0.0.0:{UDP_PORT}")
wdt.feed()

# ─── Setup ADC ────────────────────────────────────────────────────────────────

adc = analogio.AnalogIn(board.GP26)

def read_adc_avg(n: int = ADC_AVG_N) -> int:
    """
    Read ADC n times and return the integer average.
    Averaging reduces high-frequency noise from the WiFi radio,
    which otherwise appears as a faint buzz in the captured audio.
    """
    return sum(adc.value for _ in range(n)) // n

# ─── Packet buffer ────────────────────────────────────────────────────────────
# Layout: [seq: uint16 BE][sample_rate: uint16 BE][samples: int16 LE x N]

packet_buf = bytearray(4 + PACKET_SAMPLES * 2)
seq        = 0

# ─── Wait for Pi5 HELLO ───────────────────────────────────────────────────────

pi5_addr   = None
hello_buf  = bytearray(16)

print("Waiting for Pi5 HELLO...")

while pi5_addr is None:
    wdt.feed()
    try:
        nbytes, addr = sock.recvfrom_into(hello_buf)
        if bytes(hello_buf[:nbytes]).startswith(b"HELLO"):
            pi5_addr = addr[0]
            print(f"Pi5 connected: {pi5_addr}")
    except OSError:
        pass
    time.sleep(0.1)

print("Starting audio stream...\n")

# ─── Timing helpers ───────────────────────────────────────────────────────────

SAMPLE_INTERVAL_US = 1_000_000 // SAMPLE_RATE   # 62 µs at 16 kHz

def collect_packet_timed() -> int:
    """
    Collect PACKET_SAMPLES audio samples using ticks_us for drift-corrected
    timing. Returns the peak absolute deviation from midpoint (used for
    silence detection without an extra pass over the data).

    Unlike time.sleep()-based timing, ticks_us accounts for the time spent
    reading the ADC itself, so sample rate stays accurate over long sessions.
    """
    peak = 0
    t_next = time.monotonic_ns() // 1000   # current time in µs

    for i in range(PACKET_SAMPLES):
        t_next += SAMPLE_INTERVAL_US

        # Read and average ADC
        raw    = read_adc_avg(ADC_AVG_N)
        signed = raw - ADC_MIDPOINT

        # Track peak for silence detection
        magnitude = abs(signed)
        if magnitude > peak:
            peak = magnitude

        # Pack sample into buffer
        struct.pack_into("<h", packet_buf, 4 + i * 2, signed)

        # Drift-corrected wait: sleep only the remaining time for this slot
        now = time.monotonic_ns() // 1000
        remaining_us = t_next - now
        if remaining_us > 0:
            time.sleep(remaining_us / 1_000_000)

    return peak

# ─── State ────────────────────────────────────────────────────────────────────

silence_count     = 0
last_hello_time   = time.monotonic()

# ─── Main streaming loop ──────────────────────────────────────────────────────

while True:
    wdt.feed()

    # Collect one packet of audio (drift-corrected timing)
    peak = collect_packet_timed()

    # Add packet header
    struct.pack_into(">HH", packet_buf, 0, seq & 0xFFFF, SAMPLE_RATE)
    seq += 1

    # ── Silence gating ────────────────────────────────────────────────────────
    # Skip transmission during silence to save WiFi radio power.
    # The Pi5's VAD will handle any brief dropouts gracefully.

    if peak < SILENCE_THRESHOLD:
        silence_count += 1
        if silence_count > SILENCE_PACKETS:
            # In deep silence: just feed watchdog and check for HELLO
            # (don't transmit, don't burn battery on WiFi)
            try:
                nbytes, addr = sock.recvfrom_into(hello_buf)
                if bytes(hello_buf[:nbytes]).startswith(b"HELLO"):
                    pi5_addr = addr[0]
                    print(f"Pi5 reconnected: {pi5_addr}")
                    silence_count = 0   # reset so we start sending again promptly
            except OSError:
                pass
            continue   # skip sendto
    else:
        silence_count = 0

    # ── Transmit packet ───────────────────────────────────────────────────────

    try:
        sock.sendto(packet_buf, (pi5_addr, UDP_PORT))
    except OSError as e:
        print(f"Send error: {e}")

    # ── Periodic HELLO broadcast ──────────────────────────────────────────────
    # If Pi5 restarts and sends a new HELLO while we're streaming,
    # update pi5_addr so we send to the right place.
    # Also: if no contact for HELLO_INTERVAL_SEC, re-listen for a HELLO
    # in case Pi5 came back online.

    now = time.monotonic()
    if now - last_hello_time >= HELLO_INTERVAL_SEC:
        last_hello_time = now
        try:
            nbytes, addr = sock.recvfrom_into(hello_buf)
            if bytes(hello_buf[:nbytes]).startswith(b"HELLO"):
                if addr[0] != pi5_addr:
                    print(f"Pi5 address updated: {addr[0]}")
                pi5_addr = addr[0]
        except OSError:
            pass   # nothing in buffer, that's fine
