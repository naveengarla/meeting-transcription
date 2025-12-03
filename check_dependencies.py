"""
Dependency Compatibility Checker
Validates that all required dependencies can be imported and work correctly.
"""

import sys


def check_dependencies():
    """Check if all dependencies are working correctly."""
    print("=" * 60)
    print("Dependency Compatibility Check")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # Core dependencies
    dependencies = [
        ("PyQt6", "PyQt6"),
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("faster_whisper", "faster_whisper"),
        ("psutil", "psutil"),
        ("python-dotenv", "dotenv"),
    ]
    
    print("\nChecking core dependencies...")
    
    for name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"   ✅ {name}")
        except ImportError as e:
            issues.append(f"❌ {name} not installed: {str(e)}")
            print(f"   ❌ {name} - NOT FOUND")
        except Exception as e:
            warnings.append(f"⚠️  {name} import failed: {str(e)}")
            print(f"   ⚠️  {name} - ERROR: {str(e)}")
    
    # Special check: sounddevice (common DLL issues)
    print("\nChecking audio library...")
    try:
        import sounddevice as sd
        
        # Try to query devices (this will fail if PortAudio DLL is missing)
        devices = sd.query_devices()
        print(f"   ✅ sounddevice working ({len(devices)} audio devices detected)")
        
    except OSError as e:
        if "PortAudio" in str(e) or "DLL" in str(e):
            issues.append("❌ PortAudio DLL missing - install Visual C++ Redistributable")
            print(f"   ❌ PortAudio DLL error")
            print(f"   Solution: Install Visual C++ Redistributable")
            print(f"   Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        else:
            issues.append(f"❌ sounddevice error: {str(e)}")
            print(f"   ❌ sounddevice error: {str(e)}")
            
    except ImportError:
        issues.append("❌ sounddevice not installed")
        print("   ❌ sounddevice not installed")
        
    except Exception as e:
        warnings.append(f"⚠️  sounddevice check failed: {str(e)}")
        print(f"   ⚠️  Unexpected error: {str(e)}")
    
    # Special check: PyQt6 (GUI framework)
    print("\nChecking GUI framework...")
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QThread
        print("   ✅ PyQt6 working")
    except ImportError:
        issues.append("❌ PyQt6 not installed or incomplete")
        print("   ❌ PyQt6 not working")
    except Exception as e:
        warnings.append(f"⚠️  PyQt6 check failed: {str(e)}")
        print(f"   ⚠️  PyQt6 error: {str(e)}")
    
    # Special check: faster-whisper (transcription engine)
    print("\nChecking transcription engine...")
    try:
        from faster_whisper import WhisperModel
        print("   ✅ faster-whisper installed")
        
        # Check if CTranslate2 is available
        import ctranslate2
        print("   ✅ CTranslate2 backend available")
        
    except ImportError as e:
        if "ctranslate2" in str(e).lower():
            issues.append("❌ CTranslate2 not installed - faster-whisper won't work")
            print("   ❌ CTranslate2 missing")
        else:
            issues.append(f"❌ faster-whisper not installed: {str(e)}")
            print(f"   ❌ faster-whisper not working")
    except Exception as e:
        warnings.append(f"⚠️  faster-whisper check failed: {str(e)}")
        print(f"   ⚠️  Error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if not issues and not warnings:
        print("\n✅ All dependencies working correctly!")
        return True
    
    if issues:
        print(f"\n❌ Found {len(issues)} critical issue(s):")
        for issue in issues:
            print(f"   {issue}")
        print("\nSolutions:")
        print("   1. Reinstall dependencies: pip install -r requirements.txt")
        print("   2. Install Visual C++ Redistributable if audio fails")
        print("   3. Try with --no-cache-dir flag: pip install --no-cache-dir -r requirements.txt")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   {warning}")
    
    if issues:
        print("\n❌ Please fix critical issues before running the application.")
        return False
    else:
        print("\n⚠️  System will work but some features may be limited.")
        return True


def test_audio_recording():
    """Quick test of audio recording functionality."""
    print("\n" + "=" * 60)
    print("Audio Recording Test (Optional)")
    print("=" * 60)
    
    try:
        import sounddevice as sd
        import numpy as np
        
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        for idx, device in enumerate(input_devices):
            print(f"   [{idx}] {device['name']}")
            print(f"       Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']} Hz")
        
        if not input_devices:
            print("   ❌ No input devices found")
            return False
        
        print("\n✅ Audio devices detected successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Audio test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n🔍 Meeting Transcription - Dependency Check\n")
    
    result = check_dependencies()
    
    if result:
        print("\n" + "=" * 60)
        test_audio_recording()
    
    if result:
        print("\n✅ All dependencies ready!")
        print("   You can now run: python run.py")
        sys.exit(0)
    else:
        print("\n❌ Dependency issues detected.")
        print("   Run setup again: .\\setup.ps1")
        sys.exit(1)
