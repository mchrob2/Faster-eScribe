"""
RFM69 Audio Receiver
"""

import board
import busio
import digitalio
import time
import struct
import array
from rfm69 import RFM69

# Setup
RADIO_FREQ_MHZ = 915.0
NODE_ID = 2
DEST_ID = 1
ENCRYPTION_KEY = b"\x01\x02\x03\x04\x05\x06\x07\x08\x01\x02\x03\x04\x05\x06\x07\x08"
SAMPLES_PER_PACKET = 14                    
EXPECTED_PACKET_SIZE = 4 + (SAMPLES_PER_PACKET * 2)   # 32 bytes

print("=" * 60)
print("RFM69 AUDIO RECEIVER")
print(f"SAMPLES_PER_PACKET = {SAMPLES_PER_PACKET}")
print("=" * 60)

# PIN SETUP (pi5)
RFM_CS = digitalio.DigitalInOut(board.D5)   # GPIO 5
RFM_RST = digitalio.DigitalInOut(board.D6)  # GPIO 6
RFM_INT = digitalio.DigitalInOut(board.D13) # GPIO 13

#pi5 doesnt have led
led = None

#adjusted for pi5
print("RFM69 Linked: CS=GPIO5, RST=GPIO6, INT=GPIO13")

# SPI SETUP (10 MHz)
spi = busio.SPI(board.GP18, MOSI=board.GP19, MISO=board.GP16)
while not spi.try_lock():
    pass
spi.configure(baudrate=10_000_000, phase=0, polarity=0)
spi.unlock()
print("SPI at 10 MHz")

# RADIO INITIALIZATION
print("Initializing RFM69...")
rf69 = RFM69(spi, RFM_CS, RFM_RST, RADIO_FREQ_MHZ, baudrate=2_000_000)

rf69.encryption_key = ENCRYPTION_KEY
rf69.node = NODE_ID
rf69.destination = DEST_ID

# Disable ACK features for streaming
rf69.ack_delay = None
rf69.ack_retries = 0
rf69.ack_wait = 0
rf69.receive_timeout = 0.01   

print(f"Expecting {SAMPLES_PER_PACKET} samples/packet ({EXPECTED_PACKET_SIZE} bytes)")

# START LISTENING
rf69.listen()
time.sleep(0.1)
print("\nListening for audio packets...\n")

# STATISTICS
packets_received = 0
missing_packets = 0
last_packet_id = -1
rssi_sum = 0.0
rssi_count = 0
start_time = time.monotonic()
last_print_time = start_time
packets_since_print = 0

# Pre‑allocate a buffer for audio samples (optional)
audio_samples = array.array('H', [0] * SAMPLES_PER_PACKET)

while True:
    try:
        # Fast polling: check payload ready without sleeping too long
        if rf69.payload_ready():
            # Receive packet (keep listening, include RadioHead header)
            packet = rf69.receive(keep_listening=True, with_header=True, timeout=0)

            if packet is not None and len(packet) >= 8:
                # Use memoryview to avoid copying
                pkt = memoryview(packet)
                # First 4 bytes are RadioHead header; skip them
                payload = pkt[4:]

                if len(payload) == EXPECTED_PACKET_SIZE:
                    # Valid audio packet
                    packets_received += 1
                    packets_since_print += 1
                    led.value = True

                    # Extract packet ID and audio data
                    packet_id = struct.unpack_from('<I', payload, 0)[0]
                    # Optional: store samples into audio_samples
                    # audio_samples = struct.unpack_from('<' + str(SAMPLES_PER_PACKET) + 'H', payload, 4)

                    # Track missing packets
                    if last_packet_id != -1:
                        expected = (last_packet_id + 1) & 0xFFFFFFFF
                        if packet_id != expected:
                            if packet_id > expected:
                                missing = packet_id - expected
                            else:
                                missing = (0xFFFFFFFF - expected) + packet_id + 1
                            missing_packets += missing

                    last_packet_id = packet_id

                    # RSSI averaging
                    rssi = rf69.last_rssi
                    if rssi != 0:
                        rssi_sum += rssi
                        rssi_count += 1

                    led.value = False

            # Reset error counter
            error_count = 0

        # Very short sleep to prevent busy‑loop hogging CPU
        time.sleep(0.0005)   # 500 µs

        # Print statistics every second
        now = time.monotonic()
        if now - last_print_time >= 1.0 and packets_since_print > 0:
            rate = packets_since_print / (now - last_print_time)
            total = packets_received + missing_packets
            loss_pct = (missing_packets / total * 100) if total > 0 else 0
            avg_rssi = rssi_sum / rssi_count if rssi_count > 0 else -100

            print(f"RX #{packets_received:6d} | rate {rate:5.1f}/s | "
                  f" RSSI {avg_rssi:5.1f} dB")

            # Reset per‑second counters
            packets_since_print = 0
            last_print_time = now
            rssi_sum = 0.0
            rssi_count = 0

    except KeyboardInterrupt:
        print("\n\nReceiver stopped.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(0.01)

# turn off
rf69.idle()
rf69.sleep()
print(f"Total packets received: {packets_received}")
