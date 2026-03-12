"""
RFM69 Audio Receiver (Serial Output Mode)
"""

import board
import busio
import digitalio
import time
import struct
import array
import binascii
from rfm69 import RFM69

# Setup
RADIO_FREQ_MHZ = 915.0
NODE_ID = 2
DEST_ID = 1
ENCRYPTION_KEY = b"\x01\x02\x03\x04\x05\x06\x07\x08\x01\x02\x03\x04\x05\x06\x07\x08"
SAMPLES_PER_PACKET = 14                    
EXPECTED_PACKET_SIZE = 4 + (SAMPLES_PER_PACKET * 2)   # 32 bytes

# PIN SETUP
RFM_CS = digitalio.DigitalInOut(board.GP17)
RFM_RST = digitalio.DigitalInOut(board.GP20)
RFM_INT = digitalio.DigitalInOut(board.GP21)   

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT
led.value = False

# SPI SETUP
spi = busio.SPI(board.GP18, MOSI=board.GP19, MISO=board.GP16)
while not spi.try_lock():
    pass
spi.configure(baudrate=10_000_000, phase=0, polarity=0)
spi.unlock()

rf69 = RFM69(spi, RFM_CS, RFM_RST, RADIO_FREQ_MHZ, baudrate=2_000_000)
rf69.encryption_key = ENCRYPTION_KEY
rf69.node = NODE_ID
rf69.destination = DEST_ID
rf69.ack_delay = None
rf69.ack_retries = 0
rf69.ack_wait = 0
rf69.receive_timeout = 0.01   

rf69.listen()

while True:
    try:
        if rf69.payload_ready():
            packet = rf69.receive(keep_listening=True, with_header=True, timeout=0)

            if packet is not None and len(packet) >= 8:
                pkt = memoryview(packet)
                payload = pkt[4:]

                if len(payload) == EXPECTED_PACKET_SIZE:
                    led.value = True
                    
                    # Extract the 28 bytes of audio (skip the 4-byte ID)
                    audio_bytes = payload[4:]
                    
                    # Output strictly as hex string over USB CDC
                    # e.g., "b23ca10f..."
                    hex_str = binascii.hexlify(audio_bytes).decode('ascii')
                    print(hex_str)
                    
                    led.value = False

        time.sleep(0.0005)

    except KeyboardInterrupt:
        break
    except Exception:
        time.sleep(0.01)

rf69.idle()
rf69.sleep()
