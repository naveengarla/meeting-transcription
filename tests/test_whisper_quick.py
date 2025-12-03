"""Quick test of Whisper transcription with actual audio"""
import numpy as np
import config
from transcription import WhisperEngine

# Create test audio (1 second of silence + tone)
sample_rate = config.SAMPLE_RATE
duration = 3  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
# Generate a simple tone at 440 Hz (A note)
audio_data = (np.sin(2 * np.pi * 440 * t[:sample_rate]) * 16384).astype(np.int16)
# Add silence
silence = np.zeros(int(sample_rate * 2), dtype=np.int16)
audio_data = np.concatenate([audio_data, silence])

print(f"Test audio created: {len(audio_data)} samples, {len(audio_data)/sample_rate:.1f}s @ {sample_rate}Hz")
print(f"Audio dtype: {audio_data.dtype}, shape: {audio_data.shape}")
print(f"Audio range: [{audio_data.min()}, {audio_data.max()}]")

# Test Whisper
print("\n" + "="*60)
print("Testing Whisper Engine")
print("="*60)

engine = WhisperEngine(model_size='tiny')

if not engine.is_available():
    print("❌ Whisper not available")
    exit(1)

print(f"✓ Whisper is available")
print(f"Loading {engine.name}...")

try:
    segments = engine.transcribe(audio_data, sample_rate)
    print(f"\n✓ Transcription successful!")
    print(f"Segments: {len(segments)}")
    for seg in segments:
        print(f"  {seg}")
except Exception as e:
    print(f"\n❌ Error during transcription:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
