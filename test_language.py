"""Test script to verify language parameter is working"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from transcription import WhisperEngine
import numpy as np

# Create engine with Telugu
print("Creating WhisperEngine with Telugu language...")
engine = WhisperEngine(model_size="base", language="te", task="transcribe")
print(f"Engine.language = {engine.language}")
print(f"Engine.task = {engine.task}")

# Create some dummy audio (silence)
audio = np.zeros(16000, dtype=np.int16)  # 1 second of silence

print("\nAttempting transcription...")
try:
    # This will load the model and attempt transcription
    # Even with silence, we'll see what language Whisper uses
    segments = engine.transcribe(audio, 16000)
    print(f"Transcription completed: {len(segments)} segments")
    for seg in segments:
        print(f"  {seg}")
except Exception as e:
    print(f"Error: {e}")

print("\n✓ Test complete")
