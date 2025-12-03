"""Test Whisper with recorded speech"""
import sounddevice as sd
import numpy as np
import config
from transcription import WhisperEngine

print("="*60)
print("Record Speech Test for Whisper")
print("="*60)

# Get default microphone
default_mic = sd.query_devices(kind='input')
print(f"\nMicrophone: {default_mic['name']}")
print(f"Sample rate: {config.SAMPLE_RATE} Hz")

duration = 5  # seconds
print(f"\n🎤 Recording for {duration} seconds...")
print("Say something like: 'Hello, this is a test of the whisper transcription system'")
print("\nRecording in: 3...")
import time
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("GO! 🔴")

# Record
audio_data = sd.rec(
    int(duration * config.SAMPLE_RATE),
    samplerate=config.SAMPLE_RATE,
    channels=1,
    dtype='int16'
)
sd.wait()

print("✓ Recording complete!")
print(f"Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
print(f"Audio range: [{audio_data.min()}, {audio_data.max()}]")

# Check if actually captured audio (not silence)
if np.abs(audio_data).max() < 100:
    print("\n⚠ Warning: Very quiet or no audio detected. Check your microphone!")
else:
    print(f"✓ Audio detected (peak: {np.abs(audio_data).max()})")

# Flatten if needed
if len(audio_data.shape) > 1:
    audio_data = audio_data.flatten()

# Test Whisper
print("\n" + "="*60)
print("Transcribing with Whisper...")
print("="*60)

engine = WhisperEngine(model_size='base')

try:
    segments = engine.transcribe(audio_data, config.SAMPLE_RATE)
    
    print(f"\n✓ Transcription complete!")
    print(f"Found {len(segments)} segments:\n")
    
    if segments:
        for i, seg in enumerate(segments, 1):
            print(f"{i}. [{seg.start_time:.1f}s - {seg.end_time:.1f}s]")
            print(f"   {seg.text}")
            print()
    else:
        print("No speech detected in the recording.")
        print("This could mean:")
        print("  - The audio was too quiet")
        print("  - No speech was present")
        print("  - Background noise only")
        
except Exception as e:
    print(f"\n❌ Error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
