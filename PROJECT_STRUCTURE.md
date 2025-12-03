# Project Structure

```
meeting-transcription/
│
├── src/                          # Application source code
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Main GUI application
│   ├── audio_capture.py         # Audio recording module
│   ├── transcription.py         # Transcription engines
│   └── config.py                # Configuration management
│
├── tests/                        # Test suite
│   ├── __init__.py              # Test package initialization
│   ├── test_audio.py            # Audio device tests
│   ├── test_transcription.py    # Transcription engine tests
│   ├── test_whisper_system.py   # System compatibility tests
│   ├── test_whisper_quick.py    # Quick Whisper tests
│   └── test_whisper_speech.py   # Speech recording tests
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md          # Technical architecture (with Mermaid diagrams)
│   ├── REQUIREMENTS.md          # Requirements specification
│   └── ROADMAP.md               # Feature roadmap and wishlist
│
├── recordings/                   # Recorded audio files (gitignored)
│   └── .gitkeep
│
├── transcripts/                  # Exported transcripts (gitignored)
│   └── .gitkeep
│
├── models/                       # Whisper model cache (gitignored)
│   └── .gitkeep
│
├── venv/                         # Virtual environment (gitignored)
│
├── run.py                        # Application entry point
├── setup.ps1                     # Windows setup script
├── requirements.txt              # Python dependencies
├── .env.example                  # Configuration template
├── .env                          # User configuration (gitignored)
├── .gitignore                    # Git ignore rules
├── README.md                     # Main documentation
├── QUICKSTART.md                 # Quick start guide
└── LICENSE                       # MIT License (if added)
```

## Directory Purposes

### `/src` - Application Code
Contains all main application modules:
- **main.py**: PyQt6 GUI and application orchestration
- **audio_capture.py**: WASAPI audio recording (mic + system)
- **transcription.py**: Multi-engine transcription (Whisper, Chrome, Azure)
- **config.py**: Centralized configuration from .env

### `/tests` - Test Suite
Automated and manual tests:
- Audio device enumeration and recording tests
- Transcription engine validation
- System compatibility checks
- Performance benchmarks

### `/docs` - Documentation
Comprehensive project documentation:
- **ARCHITECTURE.md**: System design with Mermaid diagrams
- **REQUIREMENTS.md**: Functional and non-functional requirements
- **ROADMAP.md**: Future features and development plan

### `/recordings` - Audio Storage
WAV files from recording sessions (excluded from git)

### `/transcripts` - Exported Documents
Saved transcripts in TXT and Markdown format (excluded from git)

### `/models` - AI Model Cache
Whisper models downloaded on first use (excluded from git)

## Running the Application

### Entry Point
The application is launched via `run.py`, which:
1. Adds `src/` to Python path
2. Imports and executes `main.py`
3. Ensures proper module resolution

### Command
```powershell
python run.py
```

## Why This Structure?

✅ **Separation of Concerns**: Source code, tests, and docs clearly separated
✅ **Professional Layout**: Follows Python best practices
✅ **Easy Navigation**: Intuitive directory names
✅ **Scalable**: Easy to add new modules or tests
✅ **Clean Root**: Only essential files in root directory
✅ **Import Clarity**: Explicit `src/` package structure
