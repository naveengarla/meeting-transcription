"""
Model Downloader with Progress Indication
Downloads Whisper models with visual progress feedback.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Callable
import urllib.request
import urllib.error


class ProgressBar:
    """Simple progress bar for terminal."""
    
    def __init__(self, total: int, prefix: str = "", width: int = 50):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
        
    def update(self, downloaded: int):
        """Update progress bar."""
        self.current = downloaded
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        filled = int(self.width * self.current / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (self.width - filled)
        
        # Format size
        size_mb = self.current / (1024 * 1024)
        total_mb = self.total / (1024 * 1024)
        
        # Print progress
        sys.stdout.write(f"\r{self.prefix} [{bar}] {percent:.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def download_with_progress(url: str, destination: Path, description: str = "Downloading") -> bool:
    """
    Download a file with progress indication.
    
    Args:
        url: URL to download from
        destination: Local path to save file
        description: Description to show in progress bar
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if destination.exists():
            print(f"✓ {description} already exists: {destination}")
            return True
        
        print(f"📥 {description}...")
        print(f"   From: {url}")
        print(f"   To:   {destination}")
        
        # Download with progress
        def reporthook(block_num, block_size, total_size):
            """Progress callback."""
            if not hasattr(reporthook, 'progress_bar'):
                reporthook.progress_bar = ProgressBar(total_size, prefix=f"   ")
            
            downloaded = block_num * block_size
            reporthook.progress_bar.update(min(downloaded, total_size))
        
        urllib.request.urlretrieve(url, destination, reporthook=reporthook)
        
        print(f"✓ Download complete: {destination}")
        return True
        
    except urllib.error.URLError as e:
        print(f"\n❌ Download failed: {e.reason}")
        print(f"   Check your internet connection and try again")
        return False
    except Exception as e:
        print(f"\n❌ Error downloading file: {str(e)}")
        return False


def download_whisper_model(model_size: str, models_dir: Path = Path("models")) -> bool:
    """
    Download OpenAI Whisper model with progress indication.
    
    Args:
        model_size: Model size (tiny, base, small, medium, large)
        models_dir: Directory to store models
        
    Returns:
        True if successful, False otherwise
    """
    # Model URLs and sizes (approximate)
    models = {
        'tiny': {
            'url': 'https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt',
            'size_mb': 75
        },
        'base': {
            'url': 'https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt',
            'size_mb': 142
        },
        'small': {
            'url': 'https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt',
            'size_mb': 466
        },
        'medium': {
            'url': 'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt',
            'size_mb': 1462
        },
        'large': {
            'url': 'https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt',
            'size_mb': 2888
        }
    }
    
    if model_size not in models:
        print(f"❌ Unknown model size: {model_size}")
        print(f"   Available: {', '.join(models.keys())}")
        return False
    
    model_info = models[model_size]
    model_path = models_dir / f"{model_size}.pt"
    
    # Check if already downloaded
    if model_path.exists():
        print(f"✓ Model '{model_size}' already downloaded: {model_path}")
        return True
    
    # Warn about large downloads
    if model_info['size_mb'] > 500:
        print(f"\n⚠️  WARNING: '{model_size}' model is ~{model_info['size_mb']} MB")
        print(f"   This may take several minutes to download.")
        response = input("   Continue? (y/n): ").strip().lower()
        if response != 'y':
            print("   Download cancelled.")
            return False
    
    # Download
    description = f"Whisper model '{model_size}' (~{model_info['size_mb']} MB)"
    success = download_with_progress(model_info['url'], model_path, description)
    
    if success:
        print(f"✅ Model '{model_size}' ready to use!")
    
    return success


def download_faster_whisper_model(model_size: str, cache_dir: Optional[Path] = None) -> bool:
    """
    Download faster-whisper model (Hugging Face).
    
    Note: faster-whisper downloads models automatically on first use,
    but this function can pre-download them with progress indication.
    
    Args:
        model_size: Model size (tiny, base, small, medium, large-v2, large-v3)
        cache_dir: Optional cache directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from faster_whisper import WhisperModel, download_model
        from huggingface_hub import snapshot_download
        from tqdm import tqdm
        
        # Map model sizes to Hugging Face model IDs
        model_ids = {
            'tiny': 'Systran/faster-whisper-tiny',
            'base': 'Systran/faster-whisper-base',
            'small': 'Systran/faster-whisper-small',
            'medium': 'Systran/faster-whisper-medium',
            'large-v2': 'Systran/faster-whisper-large-v2',
            'large-v3': 'Systran/faster-whisper-large-v3',
        }
        
        # Normalize model size
        if model_size == 'large':
            model_size = 'large-v3'  # Default to latest
        
        if model_size not in model_ids:
            print(f"❌ Unknown faster-whisper model: {model_size}")
            print(f"   Available: {', '.join(model_ids.keys())}")
            return False
        
        model_id = model_ids[model_size]
        
        print(f"\n📥 Downloading faster-whisper model: {model_size}")
        print(f"   From Hugging Face: {model_id}")
        
        # Estimate sizes (approximate)
        sizes = {
            'tiny': 75,
            'base': 142,
            'small': 466,
            'medium': 1462,
            'large-v2': 2888,
            'large-v3': 2888,
        }
        
        size_mb = sizes.get(model_size, 1000)
        
        if size_mb > 500:
            print(f"\n⚠️  WARNING: Model is approximately {size_mb} MB")
            print(f"   First download may take several minutes.")
        
        # Download with tqdm progress
        print(f"\n   Downloading model files...")
        
        # Set cache directory
        if cache_dir:
            os.environ['HF_HOME'] = str(cache_dir)
        
        # Download using snapshot_download with progress
        try:
            snapshot_download(
                repo_id=model_id,
                allow_patterns=["*.json", "*.bin", "model.bin", "vocabulary.*", "tokenizer.json", "config.json"],
                cache_dir=cache_dir,
            )
            print(f"\n✅ Model '{model_size}' downloaded successfully!")
            print(f"   Cache location: {cache_dir or 'default (~/.cache/huggingface)'}")
            return True
            
        except Exception as e:
            print(f"\n❌ Download failed: {str(e)}")
            print(f"   Model will auto-download on first use.")
            return False
        
    except ImportError as e:
        print(f"❌ Required packages not installed: {e}")
        print(f"   Install with: pip install faster-whisper huggingface_hub tqdm")
        return False


def check_model_availability(model_size: str, engine: str = "whisper") -> dict:
    """
    Check if model is already downloaded.
    
    Args:
        model_size: Model size to check
        engine: Engine type ('whisper' or 'faster-whisper')
        
    Returns:
        Dictionary with availability info
    """
    result = {
        'available': False,
        'path': None,
        'size_mb': 0,
    }
    
    if engine == "whisper":
        # Check local models directory
        model_path = Path("models") / f"{model_size}.pt"
        if model_path.exists():
            result['available'] = True
            result['path'] = str(model_path)
            result['size_mb'] = model_path.stat().st_size / (1024 * 1024)
            
    elif engine == "faster-whisper":
        # Check Hugging Face cache
        try:
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            
            # Look for model directories
            model_patterns = [
                f"models--Systran--faster-whisper-{model_size}",
                f"models--guillaumekln--faster-whisper-{model_size}",
            ]
            
            for pattern in model_patterns:
                matching_dirs = list(cache_dir.glob(pattern))
                if matching_dirs:
                    result['available'] = True
                    result['path'] = str(matching_dirs[0])
                    # Calculate directory size
                    total_size = sum(f.stat().st_size for f in matching_dirs[0].rglob('*') if f.is_file())
                    result['size_mb'] = total_size / (1024 * 1024)
                    break
                    
        except Exception:
            pass
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Whisper models with progress indication")
    parser.add_argument("model", choices=['tiny', 'base', 'small', 'medium', 'large'], 
                       help="Model size to download")
    parser.add_argument("--engine", choices=['whisper', 'faster-whisper'], default='faster-whisper',
                       help="Transcription engine (default: faster-whisper)")
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory to store models (for whisper engine)")
    
    args = parser.parse_args()
    
    print(f"\n🚀 Whisper Model Downloader\n")
    
    # Check if already available
    availability = check_model_availability(args.model, args.engine)
    if availability['available']:
        print(f"✓ Model '{args.model}' already available!")
        print(f"  Path: {availability['path']}")
        print(f"  Size: {availability['size_mb']:.1f} MB")
        sys.exit(0)
    
    # Download
    if args.engine == "whisper":
        success = download_whisper_model(args.model, Path(args.models_dir))
    else:
        success = download_faster_whisper_model(args.model)
    
    sys.exit(0 if success else 1)
