#include <WiFi.h>
#include <WiFiUdp.h>
#include <driver/i2s.h>

// ---- AP config ----------------------------------------------------------
const char*    AP_SSID     = "LectureAudio";
const char*    AP_PASSWORD = "transcribe123";
const int      AP_CHANNEL  = 6;
const char*    UDP_DEST    = "192.168.4.255";
const uint16_t PORT        = 5005;

// ---- I2S pins -----------------------------------------------------------
#define I2S_BCK   18   // BCLK
#define I2S_WS    21   // LRCL / WS
#define I2S_DIN   19   // Data out  (SD on INMP441)

// ---- Audio --------------------------------------------------------------
#define SAMPLE_RATE  16000
#define CHUNK        512
#define AUDIO_GAIN   8     // INMP441 is louder than ICS-43434; start lower

// -------------------------------------------------------------------------
WiFiUDP udp;
int32_t raw[CHUNK];
int16_t pcm[CHUNK];
unsigned long lastDebugTime = 0;
int packetCount = 0;
bool i2sInitialized = false;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n\n╔════════════════════════════════════╗");
  Serial.println("║   INMP441 I2S Microphone          ║");
  Serial.println("║   ESP32 Transcriber Starting...   ║");
  Serial.println("╚════════════════════════════════════╝\n");

  // Start AP
  WiFi.mode(WIFI_AP);
  if (WiFi.softAP(AP_SSID, AP_PASSWORD, AP_CHANNEL)) {
    Serial.print("✓ AP Started: ");
    Serial.println(AP_SSID);
    Serial.print("✓ AP IP Address: ");
    Serial.println(WiFi.softAPIP());
  } else {
    Serial.println("❌ Failed to start AP!");
  }

  // Initialize I2S
  if (initI2S()) {
    i2sInitialized = true;
    Serial.println("✓ I2S initialized for INMP441");
    Serial.printf("✓ Audio gain: %dx\n", AUDIO_GAIN);
  } else {
    Serial.println("❌ I2S initialization failed!");
  }

  udp.begin(PORT);
  Serial.println("✓ UDP initialized");
  Serial.println("\n▶ Streaming audio...\n");
}

bool initI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,   // INMP441 L/R pin = GND → left ch
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num   = I2S_BCK,
    .ws_io_num    = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_DIN
  };

  esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("❌ i2s_driver_install failed: %d\n", err);
    return false;
  }

  err = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("❌ i2s_set_pin failed: %d\n", err);
    return false;
  }


  err = i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
  if (err != ESP_OK) {
    Serial.printf("❌ i2s_set_clk failed: %d\n", err);
    return false;
  }

  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

void loop() {
  if (!i2sInitialized) {
    delay(1000);
    return;
  }

  size_t bytes_read = 0;
  esp_err_t err = i2s_read(I2S_NUM_0, (void*)raw, sizeof(raw), &bytes_read, portMAX_DELAY);

  if (err != ESP_OK) return;

  int samples = bytes_read / 4;

  if (samples > 0) {
    int16_t min_val = 32767, max_val = -32768;
    int32_t sum = 0;

    for (int i = 0; i < samples; i++) {
      // INMP441 outputs 24-bit data left-justified in a 32-bit I2S frame.
      // Shift right by 8 to align to the lower 24 bits.
      int32_t sample_24bit = raw[i] >> 8;

      // Sign-extend from 24-bit to 32-bit
      if (sample_24bit & 0x800000) {
        sample_24bit |= 0xFF000000;
      }

      // Scale down to 16-bit range, then apply gain
      int32_t amplified = (sample_24bit >> 8) * AUDIO_GAIN;

      // Clamp to int16 range
      if (amplified >  32767) amplified =  32767;
      if (amplified < -32768) amplified = -32768;

      pcm[i] = (int16_t)amplified;

      if (pcm[i] < min_val) min_val = pcm[i];
      if (pcm[i] > max_val) max_val = pcm[i];
      sum += abs(pcm[i]);
    }

    // Send UDP packet
    udp.beginPacket(UDP_DEST, PORT);
    udp.write((uint8_t*)pcm, samples * 2);
    udp.endPacket();
    packetCount++;

    // Debug every 2 seconds
    if (millis() - lastDebugTime > 2000) {
      lastDebugTime = millis();
      int16_t avg = sum / samples;

      Serial.println("┌─────────────────────────────────────────┐");
      Serial.printf("│ Packet #: %-6d  Samples: %-4d        │\n", packetCount, samples);
      Serial.printf("│ Audio Range: %6d → %-6d           │\n", min_val, max_val);
      Serial.printf("│ Avg |amp|:   %-6d                     │\n", avg);

      if (abs(max_val) > 8000) {
        Serial.println("│ Status:   ✓ LOUD AUDIO                 │");
      } else if (abs(max_val) > 2000) {
        Serial.println("│ Status:   ✓ AUDIO DETECTED             │");
      } else if (abs(max_val) > 500) {
        Serial.println("│ Status:   ⚠ MEDIUM AUDIO               │");
      } else if (abs(max_val) > 100) {
        Serial.println("│ Status:   ⚠ LOW AUDIO                  │");
      } else {
        Serial.println("│ Status:   ✗ VERY LOW / NO AUDIO        │");
      }
      Serial.println("└─────────────────────────────────────────┘");
    }
  }

  delay(1);
}
