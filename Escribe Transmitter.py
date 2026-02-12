# transmitter_cp_final.py - COMPLETELY FIXED RFM69 Audio Transmitter

import board
import busio
import digitalio
import analogio
import time
import struct
import array
from rfm69 import RFM69

# ===== CONFIGURATION - MUST MATCH RECEIVER =====
RADIO_FREQ_MHZ = 915.0
NODE_ID = 1
DEST_ID = 2
ENCRYPTION_KEY = b"\x01\x02\x03\x04\x05\x06\x07\x08\x01\x02\x03\x04\x05\x06\x07\x08"
SAMPLE_RATE = 2000
SAMPLES_PER_PACKET = 10  # KEEP AT 10 - RECEIVER MUST MATCH!
PACKET_SIZE = 4 + (SAMPLES_PER_PACKET * 2)  # 24 bytes
TARGET_INTERVAL = 1.0 / (SAMPLE_RATE / SAMPLES_PER_PACKET)  # 5ms

print("=" * 60)
print("RFM69 AUDIO TRANSMITTER - FINAL FIXED")
print("=" * 60)

# ===== PIN SETUP =====
print("\n[1/5] Setting up pins...")
RFM_CS = digitalio.DigitalInOut(board.GP17)
RFM_RST = digitalio.DigitalInOut(board.GP20)
RFM_INT = digitalio.DigitalInOut(board.GP21)

mic = analogio.AnalogIn(board.GP26)

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT
led.value = False

print("  ✓ RFM69: CS=GP17, RST=GP20, INT=GP21")
print("  ✓ Microphone: GP26")
print("  ✓ LED: onboard")

# ===== SPI SETUP =====
print("\n[2/5] Setting up SPI...")
spi = busio.SPI(board.GP18, MOSI=board.GP19, MISO=board.GP16)
while not spi.try_lock():
    pass
spi.configure(baudrate=6000000, phase=0, polarity=0)  # 6MHz for stability
spi.unlock()
print("  ✓ SPI: 6MHz")

# ===== TEST RFM69 =====
print("\n[3/5] Testing RFM69 connection...")
try:
    RFM_RST.direction = digitalio.Direction.OUTPUT
    RFM_RST.value = False
    time.sleep(0.01)
    RFM_RST.value = True
    time.sleep(0.01)
    
    test_radio = RFM69(spi, RFM_CS, RFM_RST, RADIO_FREQ_MHZ, baudrate=2000000)
    temp = test_radio.temperature
    print(f"  ✓ RFM69 detected! Temperature: {temp:.1f}°C")
    print(f"  ✓ Chip version: 0x{test_radio._read_u8(0x10):02X}")
    test_radio.idle()
    del test_radio
    time.sleep(0.1)
except Exception as e:
    print(f"  ✗ RFM69 connection FAILED: {e}")
    print("\n=== CHECK WIRING ===")
    print("  □ 3.3V power")
    print("  □ Ground")
    print("  □ SPI pins (GP18,19,16)")
    print("  □ CS=GP17, RST=GP20")
    while True:
        led.value = not led.value
        time.sleep(0.2)

# ===== INITIALIZE RADIO =====
print("\n[4/5] Initializing RFM69...")
try:
    rf69 = RFM69(spi, RFM_CS, RFM_RST, RADIO_FREQ_MHZ, baudrate=2000000)
    
    # Configure for maximum power and reliability
    rf69.tx_power = 20
    rf69.encryption_key = ENCRYPTION_KEY
    rf69.node = NODE_ID
    rf69.destination = DEST_ID
    
    # CRITICAL: Disable all ACK/retry features for audio streaming
    rf69.ack_retries = 0
    rf69.ack_wait = 0
    rf69.ack_delay = None
    rf69.receive_timeout = 0.1
    
    # FIX: Override the xmit_timeout to prevent hanging
    rf69.xmit_timeout = 0.5  # Shorter timeout (was 2.0)
    
    print("  ✓ RFM69 initialized successfully!")
    print(f"  ✓ Node: {rf69.node}, Destination: {rf69.destination}")
    print(f"  ✓ TX Power: {rf69.tx_power} dBm")
    print(f"  ✓ Packet size: {PACKET_SIZE} bytes")
    
except Exception as e:
    print(f"  ✗ RFM69 initialization FAILED: {e}")
    while True:
        led.value = not led.value
        time.sleep(0.1)

# ===== OPTIONAL PING TEST (don't wait for response) =====
print("\n[5/5] Sending test ping...")
try:
    rf69.idle()
    time.sleep(0.05)
    rf69.send(b"PING")
    print("  ✓ Ping sent (receiver may not reply)")
except Exception as e:
    print(f"  ⚠️  Ping failed: {e}")

print("\n" + "=" * 60)
print("🎤 AUDIO TRANSMISSION STARTED")
print(f"   Sample rate: {SAMPLE_RATE} Hz")
print(f"   Samples/packet: {SAMPLES_PER_PACKET}")
print(f"   Packet size: {PACKET_SIZE} bytes")
print(f"   Target rate: {SAMPLE_RATE/SAMPLES_PER_PACKET:.0f} pkts/s")
print(f"   Target interval: {TARGET_INTERVAL*1000:.1f} ms")
print("=" * 60 + "\n")

# ===== AUDIO TRANSMISSION LOOP =====
packet_count = 0
audio_buffer = array.array('H', [0] * SAMPLES_PER_PACKET)
last_time = time.monotonic()
last_print_time = last_time
start_time = None
packet_errors = 0

while True:
    try:
        # Collect audio samples with micro-delays
        for i in range(SAMPLES_PER_PACKET):
            audio_buffer[i] = mic.value
            if i < SAMPLES_PER_PACKET - 1:
                time.sleep(0.0002)  # 200us between samples
        
        # Calculate audio level for visualization
        audio_max = max(audio_buffer)
        audio_min = min(audio_buffer)
        audio_peak_peak = audio_max - audio_min
        
        # Prepare packet
        packet_count += 1
        packet = struct.pack('<I', packet_count) + \
                 struct.pack(f'<{SAMPLES_PER_PACKET}H', *audio_buffer)
        
        # FIX: Always ensure radio is in idle mode before sending
        rf69.idle()
        time.sleep(0.0001)
        
        # Send packet (NO keep_listening, NO ACK)
        success = rf69.send(packet)
        
        # Visual feedback
        led.value = True
        led.value = False
        
        # Initialize start_time after first successful send
        if packet_count == 1 and success:
            start_time = time.monotonic()
        
        # Print status every 100 packets or ~1 second
        current_time = time.monotonic()
        if current_time - last_print_time >= 1.0 and packet_count > 0:
            rate = 0
            if start_time is not None:
                rate = packet_count / (current_time - start_time)
            
            # Audio level visualization
            level = int((audio_peak_peak / 65535) * 40)
            viz = "█" * level + "░" * (40 - level)
            
            print(f"TX #{packet_count:6d} | "
                  f"Rate: {rate:5.1f}/s | "
                  f"Audio: {audio_peak_peak:5d} | "
                  f"Errors: {packet_errors} | "
                  f"[{viz}]")
            
            last_print_time = current_time
        
        # Maintain timing for consistent packet rate
        elapsed = time.monotonic() - last_time
        if elapsed < TARGET_INTERVAL:
            time.sleep(TARGET_INTERVAL - elapsed)
        last_time = time.monotonic()
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("📊 TRANSMITTER FINAL STATISTICS")
        print("=" * 60)
        
        end_time = time.monotonic()
        if start_time is not None:
            runtime = end_time - start_time
            avg_rate = packet_count / runtime if runtime > 0 else 0
            print(f"\n   Packets sent:      {packet_count}")
            print(f"   Runtime:           {runtime:.1f} seconds")
            print(f"   Average rate:      {avg_rate:.1f} packets/sec")
            print(f"   Packet errors:     {packet_errors}")
            print(f"   Success rate:      {((packet_count-packet_errors)/packet_count*100):.1f}%")
        
        # Clean shutdown
        rf69.idle()
        rf69.sleep()
        mic.deinit()
        print("\n✅ Transmitter stopped cleanly")
        break
        
    except Exception as e:
        packet_errors += 1
        print(f"⚠️  Send error #{packet_errors}: {e}")
        # Try to recover
        try:
            rf69.idle()
            time.sleep(0.1)
        except:
            pass
        time.sleep(0.05)
