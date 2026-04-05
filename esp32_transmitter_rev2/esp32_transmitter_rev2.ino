#include <WiFi.h>
#include <WiFiUdp.h>
#include <driver/i2s.h>

// ── AP config ─────────────────────────────────────────────────────────────────
const char*    AP_SSID     = "LectureAudio";
const char*    AP_PASSWORD = "transcribe123";
const int      AP_CHANNEL  = 6;
const char*    UDP_DEST    = "192.168.4.255";
const uint16_t PORT        = 5005;

// ── I2S pins (ICS-43434 / INMP441) ────────────────────────────────────────────
#define I2S_BCK   18   // BCLK
#define I2S_WS    21   // LRCL / WS
#define I2S_DIN   19   // DOUT

// ── Audio ─────────────────────────────────────────────────────────────────────
const uint32_t SAMPLE_RATE   = 16000;
const uint16_t CHUNK         = 512;    // Samples per UDP packet (32ms @ 16kHz)
                                        // Smaller = lower latency for Whisper

// AGC settings — adapts gain instead of blowing out loud rooms
const float    AGC_TARGET    = 4000.0f; // Target RMS (~12% of full scale)
const float    AGC_MAX_GAIN  = 32.0f;   // Hard ceiling on gain
const float    AGC_MIN_GAIN  = 1.0f;
const float    AGC_ATTACK    = 0.005f;  // Fast attack  (respond quickly to loud sounds)
const float    AGC_RELEASE   = 0.0005f; // Slow release (don't fade quiet passages too fast)

// DC offset filter coefficient (α = 1 - 2π·fc/fs, fc ≈ 10 Hz)
// Removes mic bias without affecting voice frequencies
const float    HP_ALPHA      = 0.9961f; // 1.0 - (2*PI*10.0 / 16000.0)

// UDP packet header — lets the Pi side sanity-check packets
struct __attribute__((packed)) PacketHeader {
  uint16_t magic;       // 0xA1D0
  uint16_t seq;
  uint16_t num_samples;
  uint16_t sample_rate_khz; // e.g. 16 → 16 kHz
};
const uint16_t HEADER_MAGIC = 0xA1D0;

// ── Globals ───────────────────────────────────────────────────────────────────
WiFiUDP        udp;
int32_t        raw[CHUNK];
int16_t        pcm[CHUNK];

uint16_t       seqNum          = 0;
uint32_t       packetCount     = 0;
uint32_t       udpErrors       = 0;

float          agcGain         = 8.0f;  // Start with moderate gain
float          dcOffset        = 0.0f;  // High-pass filter state

unsigned long  lastDebugTime   = 0;
unsigned long  lastClientCheck = 0;
int            stationCount    = 0;
bool           i2sReady        = false;

// ── I2S init ──────────────────────────────────────────────────────────────────
bool initI2S() {
  i2s_config_t cfg = {
    .mode               = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate        = SAMPLE_RATE,
    .bits_per_sample    = I2S_BITS_PER_SAMPLE_32BIT,
    // ICS-43434: tie SEL to GND → left channel
    .channel_format     = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags   = ESP_INTR_FLAG_LEVEL1,
    // 8×256 = 2048 samples = 128ms DMA buffer — enough headroom, low latency
    // Previous 16×512 = ~500ms was adding unnecessary lag before Whisper saw audio
    .dma_buf_count      = 8,
    .dma_buf_len        = 256,
    .use_apll           = true,   // Use audio PLL for more accurate sample rate clock
    .tx_desc_auto_clear = false,
    .fixed_mclk         = 0
  };

  i2s_pin_config_t pins = {
    .bck_io_num   = I2S_BCK,
    .ws_io_num    = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_DIN
  };

  esp_err_t err = i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  if (err != ESP_OK) { Serial.printf("❌ i2s_driver_install: %d\n", err); return false; }

  err = i2s_set_pin(I2S_NUM_0, &pins);
  if (err != ESP_OK) { Serial.printf("❌ i2s_set_pin: %d\n", err); return false; }

  err = i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
  if (err != ESP_OK) { Serial.printf("❌ i2s_set_clk: %d\n", err); return false; }

  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("\n╔══════════════════════════════════════╗");
  Serial.println("║   ICS-43434 · ESP32 · UDP Streamer  ║");
  Serial.println("╚══════════════════════════════════════╝\n");

  WiFi.mode(WIFI_AP);
  if (WiFi.softAP(AP_SSID, AP_PASSWORD, AP_CHANNEL)) {
    Serial.printf("✓ AP: %s  IP: %s\n", AP_SSID, WiFi.softAPIP().toString().c_str());
  } else {
    Serial.println("❌ AP start failed!");
  }

  i2sReady = initI2S();
  Serial.printf("%s I2S  (APLL=%s)\n", i2sReady ? "✓" : "❌", i2sReady ? "on" : "off");

  udp.begin(PORT);
  Serial.printf("✓ UDP → %s:%u\n", UDP_DEST, PORT);
  Serial.printf("✓ Chunk: %u samples = %u ms\n", CHUNK, (CHUNK * 1000) / SAMPLE_RATE);
  Serial.printf("✓ AGC target RMS: %.0f  max gain: %.0fx\n\n", AGC_TARGET, AGC_MAX_GAIN);
  Serial.println("▶ Streaming...\n");
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
  if (!i2sReady) { delay(1000); return; }

  // ── Periodic client check (every 3s) ──────────────────────────────────────
  if (millis() - lastClientCheck > 3000) {
    lastClientCheck = millis();
    int n = WiFi.softAPgetStationNum();
    if (n != stationCount) {
      stationCount = n;
      Serial.printf(n ? "✓ %d client(s) connected\n" : "⚠ No clients\n", n);
    }
  }

  // ── Read from I2S DMA ─────────────────────────────────────────────────────
  size_t bytes_read = 0;
  esp_err_t err = i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytes_read, pdMS_TO_TICKS(100));
  if (err != ESP_OK || bytes_read == 0) return;

  int samples = bytes_read / 4; // 4 bytes per 32-bit I2S frame

  // ── Convert + DC removal + AGC ────────────────────────────────────────────
  // ICS-43434 / INMP441: 24-bit audio left-justified in 32-bit frame.
  // Bit 31 = MSB of audio, bits 7..0 = zero-pad.
  // A single arithmetic right-shift of 16 aligns to 16-bit (sign preserved).
  // No need for manual sign extension — raw[] is int32_t, >> is arithmetic.

  float rms_acc = 0.0f;

  for (int i = 0; i < samples; i++) {
    // Step 1: extract 16-bit signed sample
    float s = (float)(raw[i] >> 16);

    // Step 2: remove DC offset (single-pole high-pass, fc ≈ 10 Hz)
    // Prevents mic bias from wasting headroom and causing low-frequency rumble
    dcOffset = HP_ALPHA * dcOffset + (1.0f - HP_ALPHA) * s;
    s -= dcOffset;

    // Step 3: apply AGC gain
    s *= agcGain;

    // Step 4: soft clamp to int16 range
    if      (s >  32767.0f) s =  32767.0f;
    else if (s < -32768.0f) s = -32768.0f;

    pcm[i] = (int16_t)s;
    rms_acc += s * s;
  }

  // Step 5: update AGC after chunk
  // Use RMS as the signal level estimate
  float rms = sqrtf(rms_acc / samples);
  if (rms > 0.0f) {
    float error = AGC_TARGET / rms;   // >1 = too quiet, <1 = too loud
    float rate  = (error < 1.0f) ? AGC_ATTACK : AGC_RELEASE;
    agcGain = agcGain + rate * (agcGain * error - agcGain);
    if (agcGain > AGC_MAX_GAIN) agcGain = AGC_MAX_GAIN;
    if (agcGain < AGC_MIN_GAIN) agcGain = AGC_MIN_GAIN;
  }

  // ── Send UDP ───────────────────────────────────────────────────────────────
  // Only transmit when at least one client is connected
  if (stationCount > 0) {
    PacketHeader hdr = {
      .magic          = HEADER_MAGIC,
      .seq            = seqNum++,
      .num_samples    = (uint16_t)samples,
      .sample_rate_khz = (uint16_t)(SAMPLE_RATE / 1000)
    };

    udp.beginPacket(UDP_DEST, PORT);
    udp.write((uint8_t*)&hdr, sizeof(hdr));
    udp.write((uint8_t*)pcm, samples * 2);

    if (udp.endPacket() == 1) {
      packetCount++;
    } else {
      udpErrors++;
      if (udpErrors % 50 == 1) {
        Serial.printf("⚠ UDP errors: %u\n", udpErrors);
      }
    }
  }

  // ── Debug print every 2s ──────────────────────────────────────────────────
  if (millis() - lastDebugTime > 2000) {
    lastDebugTime = millis();

    // Find peak for status line (only when printing, not every loop)
    int16_t peak = 0;
    for (int i = 0; i < samples; i++) {
      int16_t a = abs(pcm[i]);
      if (a > peak) peak = a;
    }

    float rms_display = sqrtf(rms_acc / samples); // reuse from AGC calc above
    float kbps = (samples * 2 * 8) / ((float)CHUNK / SAMPLE_RATE * 1000.0f);

    Serial.println("┌──────────────────────────────────────────────┐");
    Serial.printf( "│ Pkts: %-6u  Errs: %-6u  Seq: %-6u       │\n",
                   packetCount, udpErrors, seqNum - 1);
    Serial.printf( "│ RMS: %-6.0f   Peak: %-6d  Gain: %-5.1fx     │\n",
                   rms_display, peak, agcGain);
    Serial.printf( "│ Clients: %-2d   Heap: %-6d   ~%-4.0f kbps      │\n",
                   stationCount, ESP.getFreeHeap(), kbps);
    Serial.printf( "│ DC offset: %-8.1f                          │\n", dcOffset);

    const char* status;
    if      (peak > 8000) status = "✓ LOUD";
    else if (peak > 2000) status = "✓ GOOD";
    else if (peak > 500 ) status = "⚠ QUIET";
    else                  status = "✗ SILENT";
    Serial.printf( "│ Signal: %-38s│\n", status);
    Serial.println("└──────────────────────────────────────────────┘");
  }
}
