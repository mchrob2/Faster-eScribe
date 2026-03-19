"""
Pico W Audio Streamer - code.py
================================
- Creates a WiFi Access Point (no external network needed)
- Samples MAX4466 mic at 16000 Hz via ADC
- Applies u-law compression (16-bit -> 8-bit)
- Streams UDP packets to connected Pi5

Wiring:
    MAX4466 OUT -> GP26 (ADC0)
    MAX4466 VCC -> 3.3V
    MAX4466 GND -> GND
    
Requirements:
    CircuitPython 9.x on Pico W
    No extra libraries needed (uses built-in wifi, socketpool, analogio)
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
AP_PASSWORD = "transcribe123" 	# min 8 chars for WPA2
UDP_PORT = 5005
SAMPLE_RATE = 16000 			# Hz
PACKET_SAMPLES = 320 			# samples per packet = 20ms of audio at 16kHz
                                # 320 bytes after ulaw encoding (fits well in UDP)
                                
# ─── U-law encoding (16-bit PCM -> 8-bit ulaw) ────────────────────────────────

ULAW_BIAS = 0x84
ULAW_CLIP = 32767

_ULAW_EXP_LUT = [0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,
                 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]

def encode_ulaw(sample):
    """Encode a single 16-bit signed PCM sample to 8-bit u-law."""
    if sample < 0:
        sample = -sample
        sign = 0x80
    else:
        sign = 0
    if sample > ULAW_CLIP:
        sample = ULAW_CLIP
    sample += ULAW_BIAS
    exp = _ULAW_EXP_LUT[sample >> 7]
    mantissa = (sample >> (exp + 3)) & 0x0F
    return ~(sign | (exp << 4) | mantissa) & 0xFF

# ─── Start Access Point ───────────────────────────────────────────────────────

print("Starting Access Point...")
wifi.radio.stop_station()
wifi.radio.start_ap(ssid=AP_SSID, password=AP_PASSWORD)
print(f"AP started: SSID='{AP_SSID}' IP={wifi.radio.ipv4_address_ap}")
print(f"Connect Pi5 to this network, then it will receive audio on UDP port {UDP_PORT}")

# ─── Setup UDP socket ─────────────────────────────────────────────────────────

pool = socketpool.SocketPool(wifi.radio)
sock = pool.socket(pool.AF_INET, pool.SOCK_DGRAM)
sock.settimeout(0) # non-blocking

# We broadcast to all clients on the AP subnet
# Pico W AP default gateway is 192.168.4.1, clients get 192.168.4.x
BROADCAST_ADDR = "192.168.4.255"

# ─── Setup ADC ────────────────────────────────────────────────────────────────

adc = analogio.AnalogIn(board.GP26)

# ADC returns 0-65535 (uint16), centre ~32768
# Convert to signed int16 by subtracting 32768
ADC_MIDPOINT = 32768

# ─── Packet header: sequence number (2 bytes) + sample_rate (2 bytes) ─────────
# Total packet = 4 bytes header + PACKET_SAMPLES bytes ulaw audio

seq = 0
packet_buf = bytearray(4 + PACKET_SAMPLES)

# ─── Sampling timing ──────────────────────────────────────────────────────────

sample_interval = 1.0 / SAMPLE_RATE # seconds between samples
pi5_addr = None # will be set when we see a "HELLO" from Pi5

# Check for incoming HELLO from Pi5 to learn its IP
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
    t_start = time.monotonic()

    # Collect all samples first, then send once
    for i in range(PACKET_SAMPLES):
        raw = adc.value
        signed = raw - ADC_MIDPOINT
        packet_buf[4 + i] = encode_ulaw(signed)
        while (time.monotonic() - t_start) < (i * sample_interval):
            pass

    # Outside the for loop, inside while True:
    struct.pack_into(">HH", packet_buf, 0, seq & 0xFFFF, SAMPLE_RATE)
    seq += 1
    try:
        sock.sendto(packet_buf, (pi5_addr, UDP_PORT))
    except OSError as e:
        print(f"Send error: {e}")

    try:
        nbytes, addr = sock.recvfrom_into(hello_buf)
        msg = bytes(hello_buf[:nbytes])
        if msg.startswith(b"HELLO"):
            pi5_addr = addr[0]
            print(f"Pi5 reconnected from {pi5_addr}")
    except OSError:
        pass