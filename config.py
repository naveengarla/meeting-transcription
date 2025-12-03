"""
Configuration module for Meeting Transcription App
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
RECORDINGS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Audio settings
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
AUDIO_FORMAT = "int16"  # 16-bit PCM

# Transcription settings
TRANSCRIPTION_MODE = os.getenv("TRANSCRIPTION_MODE", "whisper")  # chrome, whisper, or azure
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large

# Preferred audio devices (empty string = use system default)
PREFERRED_MICROPHONE = os.getenv("PREFERRED_MICROPHONE", "")
PREFERRED_SPEAKER = os.getenv("PREFERRED_SPEAKER", "")

# Azure Speech Service (optional)
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")

# UI settings
WINDOW_TITLE = "Meeting Transcription & MoM Generator"
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

# Export settings
DEFAULT_EXPORT_FORMAT = "markdown"  # txt or markdown
INCLUDE_TIMESTAMPS = True
TIMESTAMP_FORMAT = "%H:%M:%S"
