# receiver_cp_final.py - COMPLETELY FIXED RFM69 Audio Receiver
# MUST MATCH TRANSMITTER: SAMPLES_PER_PACKET = 10

import board
import busio
import digitalio
import time
import struct
import array
from rfm69 import RFM69

# ===== CONFIGURATION - MUST MATCH TRANSMITTER =====
RADIO_FREQ_MHZ = 915.0
NODE_ID = 2
DEST_ID = 1
ENCRYPTION_KEY = b"\x01\x02\x03\x04\x05\x06\x07\x08\x01\x02\x03\x04\x05\x06\x07\x08"
SAMPLES_PER_PACKET = 10  # CRITICAL FIX: Changed from 20 to match transmitter!
EXPECTED_PACKET_SIZE = 4 + (SAMPLES_PER_PACKET * 2)  # 24 bytes

print("=" * 60)
print("RFM69 AUDIO RECEIVER - FINAL FIXED")
print(f"   SAMPLES_PER_PACKET = {SAMPLES_PER_PACKET} (MATCHES TRANSMITTER)")
print("=" * 60)

# ===== PIN SETUP =====
print("\n[1/4] Setting up pins...")
RFM_CS = digitalio.DigitalInOut(board.GP17)
RFM_RST = digitalio.DigitalInOut(board.GP20)
RFM_INT = digitalio.DigitalInOut(board.GP21)  # G0/DIO0 - CRITICAL FOR RECEIVER!

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT
led.value = False

print("  ✓ RFM69: CS=GP17, RST=GP20, INT=GP21 (MANDATORY!)")

# ===== SPI SETUP =====
print("\n[2/4] Setting up SPI...")
spi = busio.SPI(board.GP18, MOSI=board.GP19, MISO=board.GP16)
while not spi.try_lock():
    pass
spi.configure(baudrate=6000000, phase=0, polarity=0)  # 6MHz for stability
spi.unlock()
print("  ✓ SPI: 6MHz")

# ===== TEST RFM69 CONNECTION =====
print("\n[3/4] Testing RFM69 connection...")

def test_rfm69_connection():
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
        return True
    except Exception as e:
        print(f"  ✗ RFM69 connection FAILED: {e}")
        print("\n=== CRITICAL - G0/DIO0 MUST BE CONNECTED TO GP21 ===")
        return False

if not test_rfm69_connection():
    print("\n❌ RFM69 connection FAILED. Fix wiring and restart.")
    while True:
        led.value = not led.value
        time.sleep(0.2)

# ===== INITIALIZE RADIO =====
print("\n[4/4] Initializing RFM69 radio...")
try:
    rf69 = RFM69(spi, RFM_CS, RFM_RST, RADIO_FREQ_MHZ, baudrate=2000000)
    
    # Configure to match transmitter
    rf69.encryption_key = ENCRYPTION_KEY
    rf69.node = NODE_ID
    rf69.destination = DEST_ID
    
    # CRITICAL: Disable all ACK features for audio streaming
    rf69.ack_delay = None
    rf69.ack_retries = 0
    rf69.ack_wait = 0
    rf69.receive_timeout = 0.05  # Short timeout for fast polling
    
    print("  ✓ RFM69 initialized successfully!")
    print(f"  ✓ Node: {rf69.node}")
    print(f"  ✓ Encryption: Enabled")
    print(f"  ✓ Expecting {SAMPLES_PER_PACKET} samples/packet ({EXPECTED_PACKET_SIZE} bytes)")
    
except Exception as e:
    print(f"  ✗ RFM69 initialization FAILED: {e}")
    while True:
        led.value = not led.value
        time.sleep(0.1)

# ===== START LISTENING =====
print("\n" + "=" * 60)
print("🎧 AUDIO RECEIVER READY")
print(f"   Frequency: {RADIO_FREQ_MHZ} MHz")
print(f"   Listening for {SAMPLES_PER_PACKET}-sample packets")
print(f"   Expected packet size: {EXPECTED_PACKET_SIZE} bytes")
print("   Press Ctrl+C to stop")
print("=" * 60 + "\n")

# Start listening
rf69.listen()
time.sleep(0.1)

# ===== RECEIVE LOOP =====
packet_count = 0
packets_received = 0  # Separate counter for audio packets only
missing_packets = 0
last_packet_id = -1
rssi_sum = 0.0
rssi_count = 0
start_time = time.monotonic()
last_print_time = start_time
packets_since_last_print = 0
error_count = 0

print("Waiting for audio packets...\n")

while True:
    try:
        # Check if packet is ready (FAST polling)
        if rf69.payload_ready():
            # Read packet with header
            packet = rf69.receive(keep_listening=True, with_header=True)
            
            if packet is not None:
                # We have a packet! Minimum length: header(4) + packet_id(4) = 8
                if len(packet) >= 8:
                    
                    # Strip RadioHead header (first 4 bytes)
                    packet_data = packet[4:]
                    
                    # Get packet ID (first 4 bytes of payload)
                    if len(packet_data) >= 4:
                        packet_id = struct.unpack_from('<I', packet_data, 0)[0]
                        
                        # Check if this is an audio packet (has the right length)
                        if len(packet_data) == EXPECTED_PACKET_SIZE:
                            # ✅ This is a valid audio packet!
                            packets_received += 1
                            packet_count += 1
                            packets_since_last_print += 1
                            led.value = True
                            
                            # Get RSSI
                            rssi = rf69.last_rssi
                            if rssi != 0:
                                rssi_sum += rssi
                                rssi_count += 1
                                avg_rssi = rssi_sum / rssi_count
                            else:
                                avg_rssi = -50  # Default
                            
                            # Parse audio data (16-bit samples)
                            audio_data = struct.unpack_from(f'<{SAMPLES_PER_PACKET}H', packet_data, 4)
                            
                            # Check for missing packets
                            if last_packet_id != -1:
                                expected_id = (last_packet_id + 1) & 0xFFFFFFFF
                                if packet_id != expected_id:
                                    if packet_id > expected_id:
                                        missing = packet_id - expected_id
                                    else:
                                        missing = (0xFFFFFFFF - expected_id) + packet_id + 1
                                    missing_packets += missing
                            
                            last_packet_id = packet_id
                            
                            # Audio statistics
                            audio_max = max(audio_data)
                            audio_min = min(audio_data)
                            audio_avg = sum(audio_data) // SAMPLES_PER_PACKET
                            audio_peak_peak = audio_max - audio_min
                            
                            # Print statistics every second
                            current_time = time.monotonic()
                            if current_time - last_print_time >= 1.0:
                                if packets_since_last_print > 0:
                                    rate = packets_since_last_print / (current_time - last_print_time)
                                    loss_rate = (missing_packets / (packets_received + missing_packets)) * 100 if (packets_received + missing_packets) > 0 else 0
                                    
                                    # Audio level visualization
                                    level = int((audio_peak_peak / 65535) * 40)
                                    viz = "█" * level + "░" * (40 - level)
                                    
                                    print(f"RX #{packets_received:6d} | "
                                          f"Rate: {rate:5.1f}/s | "
                                          f"RSSI: {avg_rssi:5.1f} dB | "
                                          f"Loss: {loss_rate:5.1f}% | "
                                          f"Audio: {audio_peak_peak:5d} | "
                                          f"[{viz}]")
                                    
                                    # Show actual audio values every 50 packets
                                    if packets_received % 50 == 0:
                                        # Convert to voltage
                                        voltages = [f"{(val/65535*3.3):.2f}V" for val in audio_data[:5]]
                                        print(f"  Samples: {audio_data[0]:5d} {audio_data[1]:5d} "
                                              f"{audio_data[2]:5d} {audio_data[3]:5d} {audio_data[4]:5d}...")
                                        print(f"  Volts:   {voltages[0]} {voltages[1]} {voltages[2]} {voltages[3]} {voltages[4]}...")
                                        print(f"  Min: {audio_min:5d} ({audio_min/65535*3.3:.2f}V) | "
                                              f"Max: {audio_max:5d} ({audio_max/65535*3.3:.2f}V)")
                                    
                                    # Reset counters
                                    packets_since_last_print = 0
                                    last_print_time = current_time
                                    rssi_sum = 0.0
                                    rssi_count = 0
                            
                            led.value = False
                            
                        elif len(packet_data) == 4 and packet_data == b'PING':
                            # This is a ping packet - ignore or optionally reply
                            pass
                            
            # Reset error count on successful receive
            error_count = 0
        
        # Very short delay when no packet - prevents CPU hogging
        time.sleep(0.0005)  # 500us
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("📊 RECEIVER FINAL STATISTICS")
        print("=" * 60)
        
        end_time = time.monotonic()
        runtime = end_time - start_time
        
        print(f"\n   Packets received:   {packets_received}")
        print(f"   Packets lost:       {missing_packets}")
        print(f"   Runtime:            {runtime:.1f} seconds")
        
        if packets_received > 0:
            total_packets = packets_received + missing_packets
            loss_rate = (missing_packets / total_packets) * 100
            avg_rate = packets_received / runtime
            avg_rssi = rssi_sum / rssi_count if rssi_count > 0 else 0
            
            print(f"   Total packets:      {total_packets}")
            print(f"   Loss rate:          {loss_rate:.1f}%")
            print(f"   Average RSSI:       {avg_rssi:.1f} dB")
            print(f"   Average rate:       {avg_rate:.1f} packets/sec")
        
        # Clean shutdown
        rf69.idle()
        rf69.sleep()
        print("\n✅ Receiver stopped cleanly")
        break
        
    except Exception as e:
        error_count += 1
        if error_count <= 3:  # Only show first few errors
            print(f"\n⚠️  Error: {e}")
        time.sleep(0.01)
