"""
Pre-download Whisper models for offline use.
Run this script to download models before first use.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_downloader import (
    download_faster_whisper_model,
    check_model_availability,
)


def main():
    """Main function to download recommended models."""
    print("=" * 70)
    print("Whisper Model Pre-Downloader")
    print("=" * 70)
    print("\nThis will download Whisper models for offline transcription.")
    print("Models are stored in ~/.cache/huggingface/hub/")
    
    # Recommended models
    recommended = {
        'tiny': {
            'size': '~75 MB',
            'speed': 'Very Fast',
            'quality': 'Basic',
            'use_case': 'Quick testing, low-end hardware'
        },
        'base': {
            'size': '~142 MB', 
            'speed': 'Fast',
            'quality': 'Good',
            'use_case': 'Recommended for most users'
        },
        'small': {
            'size': '~466 MB',
            'speed': 'Medium',
            'quality': 'Better',
            'use_case': 'Higher accuracy needed'
        },
        'medium': {
            'size': '~1.5 GB',
            'speed': 'Slow',
            'quality': 'Best',
            'use_case': 'Professional use, powerful hardware'
        }
    }
    
    print("\n📦 Available Models:")
    print("-" * 70)
    for model, info in recommended.items():
        availability = check_model_availability(model, 'faster-whisper')
        status = "✅ Downloaded" if availability['available'] else "❌ Not downloaded"
        
        print(f"\n{model.upper()}")
        print(f"  Size:     {info['size']}")
        print(f"  Speed:    {info['speed']}")
        print(f"  Quality:  {info['quality']}")
        print(f"  Use case: {info['use_case']}")
        print(f"  Status:   {status}")
        
        if availability['available']:
            print(f"  Path:     {availability['path']}")
    
    print("\n" + "=" * 70)
    print("DOWNLOAD OPTIONS")
    print("=" * 70)
    print("\n1. Download 'base' model (recommended, ~142 MB)")
    print("2. Download 'tiny' model (fastest, ~75 MB)")
    print("3. Download 'small' model (better quality, ~466 MB)")
    print("4. Download 'medium' model (best quality, ~1.5 GB)")
    print("5. Download all models (tiny + base + small, ~683 MB)")
    print("6. Skip download (models will auto-download on first use)")
    
    try:
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            models_to_download = ['base']
        elif choice == '2':
            models_to_download = ['tiny']
        elif choice == '3':
            models_to_download = ['small']
        elif choice == '4':
            models_to_download = ['medium']
        elif choice == '5':
            models_to_download = ['tiny', 'base', 'small']
        elif choice == '6':
            print("\n✓ Skipping download. Models will be downloaded automatically on first use.")
            return
        else:
            print("\n❌ Invalid choice. Exiting.")
            return
        
        print("\n" + "=" * 70)
        print("DOWNLOADING MODELS")
        print("=" * 70)
        
        success_count = 0
        for model in models_to_download:
            # Check if already downloaded
            availability = check_model_availability(model, 'faster-whisper')
            if availability['available']:
                print(f"\n✓ Model '{model}' already downloaded, skipping...")
                success_count += 1
                continue
            
            # Download
            if download_faster_whisper_model(model):
                success_count += 1
            else:
                print(f"\n⚠️  Failed to download '{model}' model")
        
        print("\n" + "=" * 70)
        print(f"✅ Downloaded {success_count}/{len(models_to_download)} models successfully!")
        print("=" * 70)
        
        if success_count == len(models_to_download):
            print("\n🚀 All models ready! You can now run: python run.py")
        else:
            print("\n⚠️  Some models failed to download.")
            print("   Don't worry - models will auto-download on first use.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
