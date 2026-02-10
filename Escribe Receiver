import machine
import utime

# RFM69 VIN  -> Pin 36 (3V3)
# RFM69 GND  -> Pin 38 (GND)
# RFM69 CS   -> Pin 22 (GP17)
# RFM69 SCK  -> Pin 24 (GP18)
# RFM69 MOSI -> Pin 25 (GP19)
# RFM69 MISO -> Pin 21 (GP16)
# RFM69 RST  -> Pin 26 (GP20)
# RFM69 G0   -> Pin 27 (GP21) for DIO0 interrupt

# HARDWARE SETUP
cs = machine.Pin(17, machine.Pin.OUT, value=1)  # Chip Select
dio0 = machine.Pin(21, machine.Pin.IN)  # DIO0 for data ready
spi = machine.SPI(0, baudrate=500000, sck=machine.Pin(18), 
                 mosi=machine.Pin(19), miso=machine.Pin(16))

# RFM69 FUNCTIONS
def write(reg, val):
    cs.value(0)
    spi.write(bytearray([reg | 0x80, val]))
    cs.value(1)

def read(reg):
    cs.value(0)
    spi.write(bytearray([reg & 0x7F]))
    result = bytearray(1)
    spi.readinto(result)
    cs.value(1)
    return result[0]

def read_fifo():
    cs.value(0)
    spi.write(bytearray([0x00 & 0x7F]))  # Read FIFO
    result = bytearray(1)
    spi.readinto(result)
    cs.value(1)
    return result[0]

# ============================================================================
# MINIMAL RFM69 RECEIVER SETUP
# ============================================================================
print("📡 RFM69 Audio Receiver - Verification Mode")
print("Setup...")

# Same settings as transmitter (MUST MATCH!)
freq = 915000000
frf = int(freq / (32000000/524288))

write(0x01, 0x00)  # Sleep
utime.sleep_ms(100)

# Frequency (MUST match transmitter!)
write(0x07, (frf >> 16) & 0xFF)
write(0x08, (frf >> 8) & 0xFF)
write(0x09, frf & 0xFF)

# Modulation (MUST match!)
write(0x02, 0x00)  # FSK
write(0x03, 0x02)  # 49.2 kbps
write(0x04, 0x68)
write(0x11, 0x00)  # PA off for RX

# Continuous reception
write(0x37, 0x00)  # Fixed length
write(0x38, 1)     # 1 byte packets

# RX mode
write(0x01, 0x14)  # RX mode
utime.sleep_ms(100)

print("✅ Receiver ready")
print("Waiting for audio...")

# ============================================================================
# AUDIO VERIFICATION WITHOUT SPEAKER
# ============================================================================
sample_count = 0
last_print = utime.ticks_ms()
signal_detected = False
silence_counter = 0
MAX_SILENCE = 100  # If no data for this many checks, signal lost

# Statistics
audio_samples = []
min_sample = 255
max_sample = 0
avg_sample = 0

try:
    while True:
        # Check if data is available (DIO0 goes high)
        if dio0.value() == 1:
            # Read audio sample
            sample = read_fifo()
            
            # Reset silence counter
            silence_counter = 0
            
            # First detection
            if not signal_detected:
                signal_detected = True
                print("\n🎯 SIGNAL DETECTED! Receiving audio...")
                audio_samples = []  # Reset for new signal
            
            # Store for analysis
            audio_samples.append(sample)
            sample_count += 1
            
            # Update min/max
            if sample < min_sample:
                min_sample = sample
            if sample > max_sample:
                max_sample = sample
            
            # Keep last 100 samples for averaging
            if len(audio_samples) > 100:
                audio_samples.pop(0)
            
        else:
            # No data
            silence_counter += 1
            if signal_detected and silence_counter > MAX_SILENCE:
                signal_detected = False
                print("\n⚠️  Signal lost. Waiting...")
        
        # Print status every second
        if utime.ticks_diff(utime.ticks_ms(), last_print) > 1000:
            # Calculate RSSI (signal strength)
            write(0x23, 0x01)  # Start RSSI measurement
            utime.sleep_us(100)
            rssi = -read(0x24) / 2
            
            # Calculate audio statistics if we have samples
            if len(audio_samples) > 10:
                avg = sum(audio_samples) // len(audio_samples)
                variation = max_sample - min_sample
                
                # Visual audio level indicator
                level_bars = int((avg / 255) * 20)
                level_display = "[" + "█" * level_bars + " " * (20 - level_bars) + "]"
                
                print(f"\n📊 Status:")
                print(f"  Samples: {sample_count:,}")
                print(f"  Signal: {'✅ STRONG' if rssi > -80 else '⚠️  WEAK'} ({rssi:.1f} dBm)")
                print(f"  Audio Level: {level_display} {avg}/255")
                print(f"  Variation: {variation} (higher = more audio activity)")
                
                # Audio activity detection
                if variation > 50:
                    print(f"  🎤 AUDIO DETECTED: Voice/audio present")
                elif variation > 20:
                    print(f"  🔊 AUDIO DETECTED: Background noise")
                else:
                    print(f"  🔇 SILENT: No audio detected")
            
            elif signal_detected:
                print(f"\n📡 Receiving... Samples: {sample_count:,} | RSSI: {rssi:.1f} dBm")
            else:
                print(f"\n🔍 Searching for signal... | RSSI: {rssi:.1f} dBm")
            
            last_print = utime.ticks_ms()
        
        # Small delay
        utime.sleep_us(50)

except KeyboardInterrupt:
    print("\nStopping receiver...")

# Final statistics
write(0x01, 0x04)  # Standby mode

print(f"\n" + "="*50)
print("📈 FINAL RECEPTION STATISTICS")
print("="*50)
print(f"Total samples received: {sample_count:,}")
print(f"Signal strength range: {min_sample} to {max_sample}")
if len(audio_samples) > 0:
    avg_final = sum(audio_samples) // len(audio_samples)
    print(f"Average audio level: {avg_final}/255")
    print(f"Audio variation: {max_sample - min_sample}")
    print(f"Signal quality: {'GOOD' if (max_sample - min_sample) > 30 else 'POOR'}")
print("="*50)
