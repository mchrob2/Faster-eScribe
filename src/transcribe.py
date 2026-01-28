from faster_whisper import WhisperModel

print("Loading model...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("Transcribing...")
segments, info = model.transcribe("test.wav")

print("Detected language:", info.language)

for s in segments:
    print(f"[{s.start:.2f}s → {s.end:.2f}s] {s.text}")

