"""
System Compatibility Test for Whisper
Tests hardware, dependencies, and runs a quick Whisper transcription
"""
import sys
import platform
import subprocess

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def test_system_info():
    """Display system information"""
    print_header("SYSTEM INFORMATION")
    
    print(f"\nOperating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    
    return True

def test_gpu_availability():
    """Check for CUDA/GPU support"""
    print_header("GPU AVAILABILITY CHECK")
    
    has_gpu = False
    
    try:
        import torch
        print(f"\n✓ PyTorch installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: YES")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            has_gpu = True
        else:
            print("⚠ CUDA available: NO")
            print("  Whisper will run on CPU (slower)")
            
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch")
        return None
    except Exception as e:
        print(f"⚠ Error checking GPU: {e}")
    
    return has_gpu

def test_whisper_installation():
    """Check if Whisper is installed"""
    print_header("WHISPER INSTALLATION CHECK")
    
    try:
        import whisper
        print(f"\n✓ Whisper installed: {whisper.__version__ if hasattr(whisper, '__version__') else 'installed'}")
        return True
    except ImportError:
        print("\n✗ Whisper not installed")
        print("  Install with: pip install openai-whisper")
        return False

def test_available_models():
    """List available Whisper models"""
    print_header("WHISPER MODELS")
    
    try:
        import whisper
        
        models = {
            'tiny': {'size': '~75 MB', 'speed': 'Very Fast', 'quality': 'Fair'},
            'base': {'size': '~142 MB', 'speed': 'Fast', 'quality': 'Good'},
            'small': {'size': '~466 MB', 'speed': 'Medium', 'quality': 'Better'},
            'medium': {'size': '~1.5 GB', 'speed': 'Slow', 'quality': 'Great'},
            'large': {'size': '~2.9 GB', 'speed': 'Very Slow', 'quality': 'Best'},
        }
        
        print("\nAvailable models:")
        print(f"\n{'Model':<10} {'Size':<12} {'Speed':<12} {'Quality':<10}")
        print("-" * 50)
        
        for model_name, info in models.items():
            print(f"{model_name:<10} {info['size']:<12} {info['speed']:<12} {info['quality']:<10}")
        
        print("\n💡 Recommendation:")
        print("  • For testing: tiny or base")
        print("  • For daily use: base or small")
        print("  • For best quality: medium or large (requires good CPU/GPU)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_whisper_quick_run():
    """Test Whisper with a quick transcription"""
    print_header("WHISPER QUICK TEST")
    
    try:
        import whisper
        import numpy as np
        import time
        
        print("\n⏳ Loading Whisper 'tiny' model (first time downloads ~75MB)...")
        start_time = time.time()
        
        model = whisper.load_model("tiny")
        load_time = time.time() - start_time
        
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
        # Create a short silent audio sample
        print("\n⏳ Testing transcription with 3-second sample...")
        sample_rate = 16000
        duration = 3
        audio = np.zeros(sample_rate * duration, dtype=np.float32)
        
        start_time = time.time()
        result = model.transcribe(audio, language='en', verbose=False)
        transcribe_time = time.time() - start_time
        
        print(f"✓ Transcription completed in {transcribe_time:.2f} seconds")
        print(f"  Result: {result['text'] if result['text'].strip() else '(silent audio - no text)'}")
        
        # Performance assessment
        print("\n📊 Performance Assessment:")
        if transcribe_time < 2:
            print("  ✓ EXCELLENT - Whisper will run smoothly on your system")
            recommended = "You can use 'base' or 'small' models"
        elif transcribe_time < 5:
            print("  ✓ GOOD - Whisper will work well")
            recommended = "Recommend 'tiny' or 'base' models"
        elif transcribe_time < 10:
            print("  ⚠ FAIR - Whisper will be slow but usable")
            recommended = "Recommend 'tiny' model only"
        else:
            print("  ⚠ SLOW - Whisper may be too slow for real-time use")
            recommended = "Consider using Chrome Web Speech API instead"
        
        print(f"  💡 Recommendation: {recommended}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_requirements():
    """Check available system memory"""
    print_header("MEMORY CHECK")
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        print(f"\nTotal RAM: {mem.total / 1024**3:.2f} GB")
        print(f"Available RAM: {mem.available / 1024**3:.2f} GB")
        print(f"Used RAM: {mem.percent}%")
        
        # Recommendations
        print("\n💡 Memory Requirements:")
        print("  • tiny model: ~1 GB RAM")
        print("  • base model: ~1-2 GB RAM")
        print("  • small model: ~2-3 GB RAM")
        print("  • medium model: ~5-6 GB RAM")
        print("  • large model: ~10+ GB RAM")
        
        if mem.available / 1024**3 < 2:
            print("\n⚠ WARNING: Low available memory")
            print("  Close other applications before using Whisper")
        else:
            print(f"\n✓ Sufficient memory available ({mem.available / 1024**3:.2f} GB)")
        
        return True
        
    except ImportError:
        print("\n⚠ psutil not installed (optional)")
        print("  Install with: pip install psutil")
        return None
    except Exception as e:
        print(f"\n⚠ Error checking memory: {e}")
        return None

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "WHISPER COMPATIBILITY TEST" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    
    results = {}
    
    # Test 1: System Info
    results['system'] = test_system_info()
    
    # Test 2: Memory
    results['memory'] = test_memory_requirements()
    
    # Test 3: GPU
    results['gpu'] = test_gpu_availability()
    
    # Test 4: Whisper Installation
    results['whisper_installed'] = test_whisper_installation()
    
    if not results['whisper_installed']:
        print("\n" + "=" * 70)
        print("INSTALLATION REQUIRED")
        print("=" * 70)
        print("\nWhisper is not installed. To install:")
        print("\n  1. Ensure you're in the virtual environment:")
        print("     .\\venv\\Scripts\\Activate.ps1")
        print("\n  2. Install Whisper:")
        print("     pip install openai-whisper")
        print("\n  3. Run this test again:")
        print("     python test_whisper_system.py")
        return
    
    # Test 5: Available Models
    results['models'] = test_available_models()
    
    # Test 6: Quick Run
    results['quick_test'] = test_whisper_quick_run()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        if result is None:
            status = "⊘ SKIP"
        elif result:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"  {status:8} {test_name}")
    
    # Final Recommendations
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    
    if results.get('quick_test'):
        print("\n✓ Your system SUPPORTS Whisper!")
        print("\n📝 Next Steps:")
        print("  1. Run the main app: python main.py")
        print("  2. Select 'Whisper (base)' from the engine dropdown")
        print("  3. Record audio and click 'Transcribe'")
        print("\n💡 Tips:")
        print("  • Start with 'base' model for good balance")
        print("  • Use 'tiny' if transcription is too slow")
        print("  • Upgrade to 'small' for better accuracy")
        
        if results.get('gpu'):
            print("\n🚀 BONUS: You have GPU support!")
            print("   Whisper will run much faster on your system")
        else:
            print("\n💻 Running on CPU")
            print("   Transcription will be slower but will work fine")
    
    elif results.get('whisper_installed'):
        print("\n⚠ Whisper is installed but the test had issues")
        print("  Check the error messages above")
        print("  You may need to reinstall dependencies:")
        print("    pip install --upgrade openai-whisper torch")
    
    else:
        print("\n⚠ Whisper needs to be installed")
        print("  Run: pip install openai-whisper")
    
    print("\n" + "=" * 70)
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Test cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
