#include <WiFi.h>
#include <WiFiUdp.h>
#include <driver/i2s.h>

// ── AP config ─────────────────────────────────────────────────────────────────
const char*    AP_SSID     = "LectureAudio";
const char*    AP_PASSWORD = "transcribe123";
const int      AP_CHANNEL  = 6;
const char*    UDP_DEST    = "192.168.4.255";
const uint16_t PORT        = 5005;

// ── I2S pins ──────────────────────────────────────────────────────────────────
#define I2S_BCK   18   // BCLK
#define I2S_WS    21   // LRCL / WS
#define I2S_DIN   19   // DOUT

// ── Audio ─────────────────────────────────────────────────────────────────────
const uint32_t SAMPLE_RATE = 16000;
const uint16_t CHUNK       = 512;   // 32ms per packet @ 16kHz

// Datasheet: mic outputs zeros for first 2^18 SCK cycles after power-up.
// At SCK = 64 × 16kHz = 1.024 MHz → ~256ms. Hold 300ms to be safe.
// Without this the AGC rails to max gain during the silent startup window,
// then slams the first real audio with 32x gain → loud thump + slow recovery.
const unsigned long STARTUP_MUTE_MS = 300;

// ── AGC ───────────────────────────────────────────────────────────────────────
// INMP441: sensitivity -26 dBFS at 94 dB SPL, full scale = 2^23.
// After >> 16, normal speech (~70 dB SPL) lands ~100-200 counts.
// AGC brings this up to a comfortable level for Whisper.
const float AGC_TARGET  = 4000.0f; // Target RMS (~12% of int16 full scale)
const float AGC_MAX     = 32.0f;   // If this rails consistently, lower TARGET
const float AGC_MIN     = 1.0f;
const float AGC_ATTACK  = 0.005f;  // Fast: clamp loud sounds quickly
const float AGC_RELEASE = 0.0005f; // Slow: don't boost quiet gaps too aggressively

// ── DC offset / high-pass ─────────────────────────────────────────────────────
// INMP441 has a hardware HPF (~1.2 Hz @ 16kHz). This software HPF at ~10 Hz
// is a useful extra layer on top — not redundant, different cutoff.
// α = 1.0 - (2π × 10 / 16000)
const float HP_ALPHA = 0.9961f;

// ── UDP packet header ─────────────────────────────────────────────────────────
struct __attribute__((packed)) PacketHeader {
  uint16_t magic;           // 0xA1D0  — sanity check on receiver
  uint16_t seq;
  uint16_t num_samples;
  uint16_t sample_rate_khz; // e.g. 16
};
const uint16_t HEADER_MAGIC = 0xA1D0;

// ── Globals ───────────────────────────────────────────────────────────────────
WiFiUDP   udp;
int32_t   raw[CHUNK];
int16_t   pcm[CHUNK];

uint16_t  seqNum       = 0;
uint32_t  packetCount  = 0;
uint32_t  udpErrors    = 0;
uint32_t  clipCount    = 0;

float     agcGain      = 4.0f;
float     dcOffset     = 0.0f;

unsigned long startupUntil   = 0;   // set in setup() after I2S init
unsigned long lastDebugTime  = 0;
unsigned long lastClientCheck = 0;
int           stationCount   = 0;
bool          i2sReady       = false;

// ── I2S init ──────────────────────────────────────────────────────────────────
bool initI2S() {
  i2s_config_t cfg = {
    .mode               = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate        = SAMPLE_RATE,
    .bits_per_sample    = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format     = I2S_CHANNEL_FMT_ONLY_LEFT, // L/R pin → GND = left ch
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags   = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count      = 8,    // 8×256 = 128ms DMA buffer
    .dma_buf_len        = 256,
    .use_apll           = true, // Audio PLL: more accurate clock than APB
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
  Serial.println("║   INMP441 · ESP32 · UDP Streamer    ║");
  Serial.println("╚══════════════════════════════════════╝\n");

  WiFi.mode(WIFI_AP);
  if (WiFi.softAP(AP_SSID, AP_PASSWORD, AP_CHANNEL)) {
    Serial.printf("✓ AP: %s  IP: %s\n", AP_SSID, WiFi.softAPIP().toString().c_str());
  } else {
    Serial.println("❌ AP start failed!");
  }

  i2sReady = initI2S();
  Serial.printf("%s I2S initialized (APLL=on)\n", i2sReady ? "✓" : "❌");

  // Start mute timer AFTER I2S init — the mic starts its 256ms countdown
  // from the moment the I2S clock begins, which is when the driver installs.
  startupUntil = millis() + STARTUP_MUTE_MS;

  udp.begin(PORT);
  Serial.printf("✓ UDP → %s:%u\n", UDP_DEST, PORT);
  Serial.printf("✓ Chunk: %u samples = %u ms\n", CHUNK, (CHUNK * 1000) / SAMPLE_RATE);
  Serial.printf("✓ Muting for %lu ms (mic startup)\n\n", STARTUP_MUTE_MS);
  Serial.println("▶ Streaming...\n");
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
  if (!i2sReady) { delay(1000); return; }

  // ── Client check every 3s ─────────────────────────────────────────────────
  if (millis() - lastClientCheck > 3000) {
    lastClientCheck = millis();
    int n = WiFi.softAPgetStationNum();
    if (n != stationCount) {
      stationCount = n;
      Serial.printf(n ? "✓ %d client(s) connected\n" : "⚠ No clients\n", n);
    }
  }

  // ── Read I2S ──────────────────────────────────────────────────────────────
  size_t bytes_read = 0;
  esp_err_t err = i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytes_read, pdMS_TO_TICKS(100));
  if (err != ESP_OK || bytes_read == 0) return;

  int samples = bytes_read / 4;

  // ── Startup mute ─────────────────────────────────────────────────────────
  // Drain DMA during mic startup window but don't process or send.
  // Prevents AGC from railing on the silent startup period.
  if (millis() < startupUntil) {
    agcGain  = 4.0f;
    dcOffset = 0.0f;
    return;
  }

  // ── Convert + DC removal + AGC ────────────────────────────────────────────
  // FIX 1: currentRMS declared here so it's in scope for both the AGC block
  //        and the debug printf below. Was previously declared inside the if()
  //        block which caused a compile error when referenced outside it.
  float currentRMS = 0.0f;
  float rms_acc    = 0.0f;
  uint32_t chunkClips = 0;

  for (int i = 0; i < samples; i++) {
    // INMP441: 24-bit audio, left-justified in 32-bit I2S frame.
    // Bit 31 = MSB of audio, bits 7:0 = zero padding (per datasheet Fig 9).
    // FIX 2: Use >> 16 directly. Previous >> 8 gave a 24-bit value in a
    //        32-bit range (~±8M), which instantly saturated the int16 clamp
    //        (±32767) and produced nothing but a wall of distortion.
    //        >> is arithmetic on int32_t so sign extension is automatic —
    //        the manual sign-extension block from the earlier version was dead code.
    float s = (float)(raw[i] >> 16);

    // DC offset removal — software HPF at ~10 Hz.
    // Complements the INMP441's built-in hardware HPF (~1.2 Hz at 16kHz).
    dcOffset = HP_ALPHA * dcOffset + (1.0f - HP_ALPHA) * s;
    s -= dcOffset;

    // Apply AGC
    s *= agcGain;

    // Clamp + count clips
    if (s > 32767.0f) {
      s = 32767.0f;
      chunkClips++;
    } else if (s < -32768.0f) {
      s = -32768.0f;
      chunkClips++;
    }

    pcm[i] = (int16_t)s;
    rms_acc += s * s;
  }

  clipCount += chunkClips;

  // ── AGC update ────────────────────────────────────────────────────────────
  currentRMS = sqrtf(rms_acc / (float)samples);
  if (currentRMS > 0.0f) {
    float targetGain = AGC_TARGET / currentRMS;
    // Attack when we need to reduce gain (loud signal), release when boosting
    float rate = (targetGain < agcGain) ? AGC_ATTACK : AGC_RELEASE;
    agcGain = agcGain * (1.0f - rate) + targetGain * rate;
    // FIX 3: Don't use != for float comparison. Just clamp unconditionally.
    if (agcGain > AGC_MAX) agcGain = AGC_MAX;
    if (agcGain < AGC_MIN) agcGain = AGC_MIN;
  }

  // ── UDP send ──────────────────────────────────────────────────────────────
  if (stationCount > 0) {
    PacketHeader hdr = {
      .magic           = HEADER_MAGIC,
      .seq             = seqNum++,
      .num_samples     = (uint16_t)samples,
      .sample_rate_khz = (uint16_t)(SAMPLE_RATE / 1000)
    };

    udp.beginPacket(UDP_DEST, PORT);
    udp.write((uint8_t*)&hdr, sizeof(hdr));
    udp.write((uint8_t*)pcm, samples * 2);

    if (udp.endPacket() == 1) {
      packetCount++;
    } else {
      udpErrors++;
      if (udpErrors % 50 == 1) Serial.printf("⚠ UDP errors: %u\n", udpErrors);
    }
  }

  // ── Debug every 2s ────────────────────────────────────────────────────────
  if (millis() - lastDebugTime > 2000) {
    lastDebugTime = millis();

    int16_t peak = 0;
    for (int i = 0; i < samples; i++) {
      int16_t a = abs(pcm[i]);
      if (a > peak) peak = a;
    }

    // FIX 4: Previous kbps calc had integer division: CHUNK/SAMPLE_RATE = 0
    //        before the float cast applied. Cast both operands explicitly.
    float chunkDurationMs = ((float)CHUNK / (float)SAMPLE_RATE) * 1000.0f;
    float kbps = ((float)(samples * 2 * 8)) / chunkDurationMs;

    Serial.println("┌──────────────────────────────────────────────┐");
    Serial.printf( "│ Pkts: %-6u  Errs: %-6u  Seq: %-6u       │\n",
                   packetCount, udpErrors, (uint16_t)(seqNum - 1));
    Serial.printf( "│ RMS: %-6.0f   Peak: %-6d  Gain: %-5.1fx     │\n",
                   currentRMS, peak, agcGain);
    Serial.printf( "│ Clips: %-5u   Heap: %-6d   ~%-4.0f kbps      │\n",
                   clipCount, ESP.getFreeHeap(), kbps);
    Serial.printf( "│ DC: %-8.1f  Clients: %-2d                  │\n",
                   dcOffset, stationCount);

    const char* status;
    if      (peak > 8000) status = "✓ LOUD";
    else if (peak > 2000) status = "✓ GOOD";
    else if (peak > 500 ) status = "⚠ QUIET";
    else                  status = "✗ SILENT / STARTUP";
    Serial.printf( "│ Signal: %-38s│\n", status);
    Serial.println("└──────────────────────────────────────────────┘");
  }
}
