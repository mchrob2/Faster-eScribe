import machine
import utime
# RFM69 Pins - PICO Pins
# RFM69 VIN  - Pin 36 (3V3)
# RFM69 GND  - Pin 38 (GND)
# RFM69 CS   - Pin 22 (GP17)
# RFM69 SCK  - Pin 24 (GP18)
# RFM69 MOSI - Pin 25 (GP19)
# RFM69 MISO - Pin 21 (GP16)
# RFM69 RST  - Pin 26 (GP20)
# MIC OUT    - Pin 31 (GP26)

# MINIMAL HARDWARE SETUP
cs = machine.Pin(17, machine.Pin.OUT, value=1)  # Chip Select/clock
spi = machine.SPI(0, baudrate=500000, sck=machine.Pin(18), mosi=machine.Pin(19), miso=machine.Pin(16))
mic = machine.ADC(26) 


# MINIMAL RFM69 FUNCTIONS
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


# RFM69 SETUP
# Setup for 915MHz (change to 433000000 for 433MHz)
freq = 915000000
frf = int(freq / (32000000/524288))

write(0x01, 0x00)  # Sleep
utime.sleep_ms(100)

# Frequency
write(0x07, (frf >> 16) & 0xFF)
write(0x08, (frf >> 8) & 0xFF)
write(0x09, frf & 0xFF)

# Modulation
write(0x02, 0x00)  # FSK

# Bitrate
write(0x03, 0x02)  # 49.2 kbps
write(0x04, 0x68)

# Power
write(0x11, 0x5F)  # 13 dBm

# Continuous mode
write(0x37, 0x00)  # Fixed length
write(0x38, 1)     # 1 byte packets

# Standby
write(0x01, 0x04)
utime.sleep_ms(100)

# TRANSMIT AUDIO WITH STATUS
print("🎤 RFM69 Audio Transmitter")
print("Starting transmission...")

# Switch to TX mode
write(0x01, 0x0C)  # TX mode
utime.sleep_ms(10)

# Verify TX mode
mode = read(0x01) & 0x1C
if mode != 0x0C:
    print(f"ERROR: Not in TX mode! (0x{mode:02x})")
else:
    print("✅ Transmitter active")

sample_count = 0
last_print = utime.ticks_ms()
transmission_verified = False

try:
    while True:
        # Read mic and transmit
        sample = mic.read_u16() >> 8  # 8-bit sample
        write(0x00, sample)
        sample_count += 1
        
        # Check transmission status every second
        if utime.ticks_diff(utime.ticks_ms(), last_print) > 1000:
            irq = read(0x28)  # Read IRQ flags
            
            # Bit 3 (0x08) = PacketSent flag
            if irq & 0x08:
                print(f"✅ TRANSMITTING | Samples: {sample_count}")
                transmission_verified = True
            else:
                print(f"⚠️  NOT TRANSMITTING | Samples: {sample_count}")
                print(f"   IRQ flags: 0x{irq:02x} (want bit 3=1)")
            
            last_print = utime.ticks_ms()
        
        # 8 kHz sampling
        utime.sleep_us(125)
        
except KeyboardInterrupt:
    print("\nStopping...")

# Cleanup
write(0x01, 0x04)  # Standby mode
print(f"\n📊 Final: {sample_count} samples sent")
print(f"Transmission verified: {'✅ YES' if transmission_verified else '❌ NO'}")
