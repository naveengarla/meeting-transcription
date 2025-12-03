"""
Test script for audio capture functionality
Run this to verify audio devices and test recording
"""
import time
import sys
from datetime import datetime
from audio_capture import AudioCapture
import config


def test_list_devices():
    """Test: List all available audio devices"""
    print("=" * 70)
    print("TEST 1: List Audio Devices")
    print("=" * 70)
    
    capture = AudioCapture()
    input_devices, loopback_devices = capture.list_devices()
    
    print(f"\n✓ Found {len(input_devices)} input device(s):")
    for i, dev in enumerate(input_devices, 1):
        print(f"  {i}. {dev}")
    
    print(f"\n✓ Found {len(loopback_devices)} loopback device(s):")
    if loopback_devices:
        for i, dev in enumerate(loopback_devices, 1):
            print(f"  {i}. {dev}")
    else:
        print("  ⚠ No loopback devices found")
        print("  ℹ Enable 'Stereo Mix' in Windows Sound settings or install VB-Cable")
    
    return capture, len(input_devices) > 0


def test_default_devices(capture):
    """Test: Get default audio devices"""
    print("\n" + "=" * 70)
    print("TEST 2: Default Devices")
    print("=" * 70)
    
    mic_device, speaker_device = capture.get_default_devices()
    
    print(f"\n✓ Default Microphone: {mic_device}")
    print(f"✓ Default Speaker/Loopback: {speaker_device}")
    
    return mic_device, speaker_device


def test_microphone_recording(capture):
    """Test: Record from microphone"""
    print("\n" + "=" * 70)
    print("TEST 3: Microphone Recording (5 seconds)")
    print("=" * 70)
    
    input("\nPress ENTER to start 5-second microphone test...")
    
    print("\n🎤 Recording... Please speak into your microphone!")
    success = capture.start_recording(record_mic=True, record_speaker=False)
    
    if not success:
        print("✗ Failed to start recording")
        return False
    
    # Record for 5 seconds
    for i in range(5, 0, -1):
        print(f"  {i}...", end=' ', flush=True)
        time.sleep(1)
    print("\n")
    
    mic_audio, _ = capture.stop_recording()
    
    if mic_audio is not None:
        # Save the recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_mic_{timestamp}"
        saved_path = capture.save_audio(mic_audio, filename)
        print(f"\n✓ Test recording saved: {saved_path}")
        print(f"  Duration: {len(mic_audio) / config.SAMPLE_RATE:.2f} seconds")
        print(f"  Samples: {len(mic_audio)}")
        return True
    else:
        print("✗ No audio captured")
        return False


def test_speaker_recording(capture):
    """Test: Record system audio (loopback)"""
    print("\n" + "=" * 70)
    print("TEST 4: System Audio Recording (5 seconds)")
    print("=" * 70)
    
    print("\n⚠ Make sure you have:")
    print("  1. Stereo Mix enabled in Windows Sound settings, OR")
    print("  2. VB-Audio Virtual Cable installed")
    print("\nℹ Play some music/video in another window during this test")
    
    input("\nPress ENTER to start 5-second system audio test...")
    
    print("\n🔊 Recording system audio... Play something!")
    success = capture.start_recording(record_mic=False, record_speaker=True)
    
    if not success:
        print("✗ Failed to start recording")
        print("  Tip: Enable Stereo Mix or install VB-Cable")
        return False
    
    # Record for 5 seconds
    for i in range(5, 0, -1):
        print(f"  {i}...", end=' ', flush=True)
        time.sleep(1)
    print("\n")
    
    _, speaker_audio = capture.stop_recording()
    
    if speaker_audio is not None:
        # Save the recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_speaker_{timestamp}"
        saved_path = capture.save_audio(speaker_audio, filename)
        print(f"\n✓ Test recording saved: {saved_path}")
        print(f"  Duration: {len(speaker_audio) / config.SAMPLE_RATE:.2f} seconds")
        print(f"  Samples: {len(speaker_audio)}")
        return True
    else:
        print("✗ No audio captured")
        print("  Tip: Check if Stereo Mix is enabled and set as default recording device")
        return False


def test_dual_recording(capture):
    """Test: Record both microphone and system audio simultaneously"""
    print("\n" + "=" * 70)
    print("TEST 5: Dual Recording (Mic + System Audio, 5 seconds)")
    print("=" * 70)
    
    print("\nℹ This will record both your microphone AND system audio")
    print("  Speak while playing audio in the background")
    
    input("\nPress ENTER to start dual recording test...")
    
    print("\n🎤🔊 Recording both sources...")
    success = capture.start_recording(record_mic=True, record_speaker=True)
    
    if not success:
        print("✗ Failed to start recording")
        return False
    
    # Record for 5 seconds
    for i in range(5, 0, -1):
        print(f"  {i}...", end=' ', flush=True)
        time.sleep(1)
    print("\n")
    
    mic_audio, speaker_audio = capture.stop_recording()
    
    if mic_audio is not None or speaker_audio is not None:
        # Merge and save
        merged_audio = capture.merge_audio_channels(mic_audio, speaker_audio)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_dual_{timestamp}"
        saved_path = capture.save_audio(merged_audio, filename)
        print(f"\n✓ Test recording saved: {saved_path}")
        print(f"  Duration: {len(merged_audio) / config.SAMPLE_RATE:.2f} seconds")
        print(f"  Mic samples: {len(mic_audio) if mic_audio is not None else 0}")
        print(f"  Speaker samples: {len(speaker_audio) if speaker_audio is not None else 0}")
        return True
    else:
        print("✗ No audio captured from either source")
        return False


def main():
    """Run all audio capture tests"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "AUDIO CAPTURE TEST SUITE" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\nThis will test audio capture functionality step by step.")
    print("Make sure you have a working microphone connected.")
    
    results = {}
    
    # Test 1: List devices
    capture, has_input = test_list_devices()
    results['list_devices'] = has_input
    
    if not has_input:
        print("\n✗ No input devices found. Cannot continue tests.")
        sys.exit(1)
    
    # Test 2: Default devices
    mic_device, speaker_device = test_default_devices(capture)
    results['default_devices'] = mic_device is not None
    
    # Test 3: Microphone recording
    try:
        results['mic_recording'] = test_microphone_recording(capture)
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        results['mic_recording'] = False
    except Exception as e:
        print(f"\n✗ Error during microphone test: {e}")
        results['mic_recording'] = False
    
    # Test 4: Speaker recording (optional)
    if speaker_device:
        try:
            results['speaker_recording'] = test_speaker_recording(capture)
        except KeyboardInterrupt:
            print("\n\n⚠ Test interrupted by user")
            results['speaker_recording'] = False
        except Exception as e:
            print(f"\n✗ Error during speaker test: {e}")
            results['speaker_recording'] = False
    else:
        print("\n⚠ Skipping speaker recording test (no loopback device)")
        results['speaker_recording'] = None
    
    # Test 5: Dual recording (optional)
    if speaker_device and results.get('mic_recording') and results.get('speaker_recording'):
        try:
            results['dual_recording'] = test_dual_recording(capture)
        except KeyboardInterrupt:
            print("\n\n⚠ Test interrupted by user")
            results['dual_recording'] = False
        except Exception as e:
            print(f"\n✗ Error during dual recording test: {e}")
            results['dual_recording'] = False
    else:
        print("\n⚠ Skipping dual recording test")
        results['dual_recording'] = None
    
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
    
    if results['mic_recording']:
        print("\n✓ Audio capture is working!")
        print(f"  Check the 'recordings/' folder for test files")
    else:
        print("\n⚠ Audio capture issues detected")
        print("  Please check:")
        print("    1. Microphone is connected and enabled")
        print("    2. Windows has microphone permissions for Python")
        print("    3. No other application is using the microphone")
    
    if not results.get('speaker_recording') and results.get('speaker_recording') is not None:
        print("\n⚠ System audio capture not working")
        print("  To fix:")
        print("    1. Enable Stereo Mix in Windows Sound settings")
        print("    2. Or install VB-Audio Virtual Cable")
        print("    3. See README.md for detailed instructions")
    
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
