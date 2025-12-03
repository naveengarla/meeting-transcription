"""
Test script for transcription engines
Run this to verify Whisper, Chrome, and Azure transcription
"""
import sys
import wave
import numpy as np
from pathlib import Path
from transcription import TranscriptionManager, WhisperEngine, ChromeSpeechEngine, AzureSpeechEngine
import config


def create_test_audio():
    """Create a simple test audio file (sine wave beep)"""
    print("Creating test audio file...")
    
    duration = 3  # seconds
    frequency = 440  # Hz (A note)
    sample_rate = config.SAMPLE_RATE
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Convert to int16
    audio = (audio * 32767).astype(np.int16)
    
    # Save to file
    test_file = config.RECORDINGS_DIR / "test_audio.wav"
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    
    print(f"✓ Test audio created: {test_file}")
    return audio, test_file


def load_audio_file(file_path: Path):
    """Load audio from WAV file"""
    try:
        with wave.open(str(file_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
        
        return audio, sample_rate
    except Exception as e:
        print(f"✗ Error loading audio file: {e}")
        return None, None


def test_list_engines():
    """Test: List available transcription engines"""
    print("=" * 70)
    print("TEST 1: List Available Engines")
    print("=" * 70)
    
    manager = TranscriptionManager()
    engines = manager.list_engines()
    
    print(f"\n✓ Found {len(engines)} transcription engine(s):\n")
    
    for engine_name in engines:
        engine = manager.get_engine(engine_name)
        available = "✓ Available" if engine.is_available() else "✗ Not configured"
        print(f"  {available:15} {engine.name}")
        
        # Show setup hints for unavailable engines
        if not engine.is_available():
            if engine_name == 'whisper':
                print(f"                   → Install: pip install openai-whisper torch")
            elif engine_name == 'azure':
                print(f"                   → Configure AZURE_SPEECH_KEY in .env")
    
    print(f"\n✓ Active engine: {manager.active_engine}")
    
    return manager, len(engines) > 0


def test_chrome_engine(manager):
    """Test: Chrome Web Speech API engine"""
    print("\n" + "=" * 70)
    print("TEST 2: Chrome Web Speech API")
    print("=" * 70)
    
    chrome_engine = manager.get_engine('chrome')
    
    if not chrome_engine:
        print("✗ Chrome engine not available")
        return False
    
    print(f"\n✓ Engine: {chrome_engine.name}")
    print(f"✓ Available: {chrome_engine.is_available()}")
    
    print("\nℹ Chrome Web Speech API features:")
    print("  • Real-time browser-based transcription")
    print("  • Uses Google's speech recognition")
    print("  • Requires internet connection")
    print("  • Best for live transcription")
    
    print("\n⚠ Chrome engine is designed for real-time use")
    print("  To test: Run the main app and click 'Open Chrome Speech Recognition'")
    
    # Generate HTML page
    print("\n✓ Generating Chrome speech recognition page...")
    html_content = chrome_engine.get_html_page()
    print(f"  HTML page size: {len(html_content)} bytes")
    
    return True


def test_whisper_engine(manager, audio_data, sample_rate):
    """Test: Whisper local transcription engine"""
    print("\n" + "=" * 70)
    print("TEST 3: Whisper Engine")
    print("=" * 70)
    
    whisper_engine = manager.get_engine('whisper')
    
    if not whisper_engine:
        print("✗ Whisper engine not available")
        print("  Install with: pip install openai-whisper torch")
        return False
    
    print(f"\n✓ Engine: {whisper_engine.name}")
    print(f"✓ Available: {whisper_engine.is_available()}")
    
    # Check for real audio file
    recordings = list(config.RECORDINGS_DIR.glob("*.wav"))
    
    if not recordings:
        print("\n⚠ No recordings found for transcription test")
        print("  To test Whisper:")
        print("    1. Run: python test_audio.py")
        print("    2. Record some speech")
        print("    3. Run this test again")
        return None
    
    # Use first recording
    test_file = recordings[0]
    print(f"\n✓ Using recording: {test_file.name}")
    
    # Load audio
    audio, sr = load_audio_file(test_file)
    if audio is None:
        return False
    
    print(f"  Duration: {len(audio) / sr:.2f} seconds")
    print(f"  Samples: {len(audio)}")
    
    # Transcribe
    try:
        print(f"\n⏳ Transcribing with Whisper (this may take a while)...")
        segments = whisper_engine.transcribe(audio, sr)
        
        print(f"\n✓ Transcription complete!")
        print(f"  Segments: {len(segments)}")
        
        if segments:
            print("\n  Preview:")
            for i, seg in enumerate(segments[:3], 1):  # Show first 3 segments
                print(f"    {i}. [{seg.start_time:.1f}s] {seg.text}")
            
            if len(segments) > 3:
                print(f"    ... and {len(segments) - 3} more segments")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Whisper transcription failed: {e}")
        return False


def test_azure_engine(manager, audio_data, sample_rate):
    """Test: Azure Speech Service engine"""
    print("\n" + "=" * 70)
    print("TEST 4: Azure Speech Service")
    print("=" * 70)
    
    azure_engine = manager.get_engine('azure')
    
    if not azure_engine:
        print("✗ Azure engine not available")
        return False
    
    print(f"\n✓ Engine: {azure_engine.name}")
    
    if not azure_engine.is_available():
        print("✗ Azure not configured")
        print("\n  To configure:")
        print("    1. Get Azure Speech Service credentials")
        print("    2. Edit .env file:")
        print("       AZURE_SPEECH_KEY=your_key")
        print("       AZURE_SPEECH_REGION=your_region")
        return False
    
    print(f"✓ Configured: {azure_engine.region}")
    
    # Check for real audio file
    recordings = list(config.RECORDINGS_DIR.glob("*.wav"))
    
    if not recordings:
        print("\n⚠ No recordings found for transcription test")
        return None
    
    # Use first recording
    test_file = recordings[0]
    print(f"\n✓ Using recording: {test_file.name}")
    
    # Load audio
    audio, sr = load_audio_file(test_file)
    if audio is None:
        return False
    
    print(f"  Duration: {len(audio) / sr:.2f} seconds")
    
    # Transcribe
    try:
        print(f"\n⏳ Transcribing with Azure...")
        segments = azure_engine.transcribe(audio, sr)
        
        print(f"\n✓ Transcription complete!")
        print(f"  Segments: {len(segments)}")
        
        if segments:
            print("\n  Result:")
            for seg in segments:
                print(f"    {seg.text}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Azure transcription failed: {e}")
        return False


def test_export(manager):
    """Test: Export transcript to file"""
    print("\n" + "=" * 70)
    print("TEST 5: Export Transcript")
    print("=" * 70)
    
    # Create sample segments
    from transcription import TranscriptSegment
    
    segments = [
        TranscriptSegment("Hello, this is a test transcript.", 0.0, 2.5, confidence=0.95),
        TranscriptSegment("This is the second segment.", 2.5, 5.0, confidence=0.92),
        TranscriptSegment("And this is the final segment.", 5.0, 7.5, confidence=0.88),
    ]
    
    print(f"\n✓ Created {len(segments)} test segments")
    
    # Export as Markdown
    try:
        md_path = config.TRANSCRIPTS_DIR / "test_transcript.md"
        manager.export_transcript(segments, md_path, format='markdown')
        print(f"\n✓ Markdown export successful: {md_path}")
        
        # Show preview
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("\n  Preview (first 300 chars):")
            print("  " + "-" * 66)
            for line in content[:300].split('\n'):
                print(f"  {line}")
            print("  ...")
        
    except Exception as e:
        print(f"✗ Markdown export failed: {e}")
        return False
    
    # Export as Text
    try:
        txt_path = config.TRANSCRIPTS_DIR / "test_transcript.txt"
        manager.export_transcript(segments, txt_path, format='txt')
        print(f"\n✓ Text export successful: {txt_path}")
        
    except Exception as e:
        print(f"✗ Text export failed: {e}")
        return False
    
    return True


def main():
    """Run all transcription tests"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "TRANSCRIPTION ENGINE TEST SUITE" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    
    results = {}
    
    # Test 1: List engines
    manager, has_engines = test_list_engines()
    results['list_engines'] = has_engines
    
    if not has_engines:
        print("\n✗ No transcription engines available")
        sys.exit(1)
    
    # Create test audio
    audio_data, test_file = create_test_audio()
    
    # Test 2: Chrome engine
    results['chrome'] = test_chrome_engine(manager)
    
    # Test 3: Whisper engine
    if 'whisper' in manager.list_engines():
        results['whisper'] = test_whisper_engine(manager, audio_data, config.SAMPLE_RATE)
    else:
        print("\n⊘ Whisper engine not available")
        results['whisper'] = None
    
    # Test 4: Azure engine
    if 'azure' in manager.list_engines():
        results['azure'] = test_azure_engine(manager, audio_data, config.SAMPLE_RATE)
    else:
        print("\n⊘ Azure engine not available")
        results['azure'] = None
    
    # Test 5: Export
    results['export'] = test_export(manager)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else ("⊘ SKIP" if result is None else "✗ FAIL")
        print(f"  {status:8} {test_name}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Tests cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
