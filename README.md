# Meeting Transcription & MoM Generator

A Windows desktop application for real-time meeting transcription and Minutes of Meeting (MoM) generation. Captures both microphone and system audio, transcribes speech to text, and exports formatted transcripts.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

### Audio Capture
- **Microphone Recording** - Capture your voice during meetings
- **System Audio Loopback** - Record audio from Teams, Zoom, or any application
- **Dual Channel Support** - Record both mic and system audio simultaneously
- **WASAPI Backend** - High-quality Windows audio capture

### Transcription Engines (3 Options)

#### 🌐 Chrome Web Speech API (Recommended for most users)
- ✅ **Free** - No API costs
- ✅ **Real-time** - See transcription as you speak
- ✅ **Good accuracy** - Google-powered
- ✅ **Easy setup** - No model downloads
- ❌ Requires internet connection
- ❌ Privacy concern (audio sent to Google)

#### 🔒 Whisper (Best for privacy)
- ✅ **Offline** - Works without internet
- ✅ **High quality** - OpenAI's SOTA model
- ✅ **Private** - All processing local
- ✅ **Free** - Open source
- ❌ Requires GPU/CPU power
- ❌ Large model downloads (1-3 GB)
- ❌ Slower than real-time

#### ☁️ Azure Speech Service (Enterprise)
- ✅ **Highly accurate** - Enterprise-grade
- ✅ **Fast** - Cloud processing
- ✅ **Reliable** - Microsoft infrastructure
- ❌ Requires Azure subscription
- ❌ API costs apply
- ❌ Requires internet

### Export & Formatting
- 📄 **Markdown Export** - Formatted with timestamps and sections
- 📝 **Plain Text Export** - Simple timestamped transcript
- ⏱️ **Timestamps** - Track when each segment was spoken
- 💾 **Auto-save Recordings** - All audio saved to `recordings/` folder

## 🚀 Quick Start

### Prerequisites
- **Windows 10/11** (64-bit)
- **Python 3.10+** ([Download here](https://www.python.org/downloads/))
- **Chrome or Edge browser** (for Chrome Speech API option)

### Installation

1. **Clone or download this repository**
   ```powershell
   cd c:\Src2\speech2text
   ```

2. **Run the setup script**
   ```powershell
   .\setup.ps1
   ```
   
   This will:
   - Create a virtual environment
   - Install all dependencies
   - Create configuration files

3. **Activate the virtual environment**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. **(Optional) Configure Azure Speech Service**
   
   If you want to use Azure transcription:
   - Edit `.env` file
   - Add your Azure credentials:
     ```
     AZURE_SPEECH_KEY=your_key_here
     AZURE_SPEECH_REGION=your_region_here
     TRANSCRIPTION_MODE=azure
     ```

5. **Run the application**
   ```powershell
   python run.py
   ```

## 📖 Usage Guide

### Recording a Meeting

#### Option 1: Chrome Web Speech (Real-time)

1. **Select "Chrome Web Speech API" from the engine dropdown**
2. **Click "Open Chrome Speech Recognition"**
   - Browser window opens
   - Grant microphone permissions
3. **Click "Start Listening" in the browser**
4. **Speak naturally** - transcript appears in real-time in both browser and app
5. **Save transcript** when done

**Best for:** Quick meetings, real-time transcription, minimal setup

#### Option 2: Record then Transcribe (Whisper/Azure)

1. **Select audio sources:**
   - ✓ Microphone - Check to record your voice
   - ✓ System Audio - Check to record Teams/Zoom audio
2. **Select transcription engine** (Whisper or Azure)
3. **Click "Start Recording"**
4. **Conduct your meeting**
5. **Click "Stop Recording"**
6. **Click "Transcribe"** - wait for processing
7. **Save transcript** as TXT or Markdown

**Best for:** Offline transcription, high accuracy, longer meetings

### Exporting Transcripts

**Save as Markdown (.md)**
```markdown
# Meeting Transcript

**Date:** 2025-12-03 14:30:00
**Duration:** 245.50 seconds
**Segments:** 12

---

### Segment 1 [0:00:00]

Hello everyone, welcome to today's meeting...

### Segment 2 [0:00:15]

Let's discuss the project timeline...
```

**Save as Text (.txt)**
```
Meeting Transcript - 2025-12-03 14:30:00
======================================================================

[0:00:00] Hello everyone, welcome to today's meeting...
[0:00:15] Let's discuss the project timeline...
```

## 🛠️ Configuration

### Audio Settings (`config.py` or `.env`)

```python
SAMPLE_RATE=16000      # Audio sample rate (Hz)
CHANNELS=1             # Mono (1) or Stereo (2)
CHUNK_SIZE=1024        # Audio buffer size
```

### Transcription Settings

```python
TRANSCRIPTION_MODE=chrome   # Default engine: chrome, whisper, or azure
WHISPER_MODEL=base          # Whisper model size: tiny, base, small, medium, large
```

**Whisper Model Sizes:**
| Model  | Size   | Speed   | Accuracy  | Use Case          |
| ------ | ------ | ------- | --------- | ----------------- |
| tiny   | 75 MB  | Fast    | Good      | Quick transcripts |
| base   | 142 MB | Fast    | Better    | **Recommended**   |
| small  | 466 MB | Medium  | Great     | High accuracy     |
| medium | 1.5 GB | Slow    | Excellent | Best quality      |
| large  | 2.9 GB | Slowest | Best      | Maximum accuracy  |

## 🔧 Troubleshooting

### Common Issues

#### "No loopback device found" / Can't record system audio

**Solution 1: Enable Stereo Mix (Built-in Windows)**
1. Right-click speaker icon in taskbar → **Sounds**
2. Go to **Recording** tab
3. Right-click empty area → **Show Disabled Devices**
4. Right-click **Stereo Mix** → **Enable**
5. Restart the app and refresh devices

**Solution 2: Use Virtual Audio Cable**
1. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free)
2. Set as default playback device
3. Restart app

#### Chrome Web Speech not working

**Symptoms:** Browser says "Speech recognition not supported"

**Solutions:**
- ✓ Use Chrome or Edge browser (not Firefox/Safari)
- ✓ Check microphone permissions in browser
- ✓ Ensure website has HTTPS or is localhost
- ✓ Try restarting the browser

#### Whisper transcription is very slow

**Solutions:**
- Use smaller model: `WHISPER_MODEL=tiny` or `base`
- Install CUDA for GPU acceleration:
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- Reduce audio quality: `SAMPLE_RATE=8000`

#### "Module not found" errors

**Solution:**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

#### Transcription accuracy is poor

**Solutions:**
- Use better microphone
- Reduce background noise
- Speak clearly and at moderate pace
- Use larger Whisper model: `WHISPER_MODEL=medium`
- Switch to Azure Speech Service for best accuracy

### Debug Audio Devices

Run the audio capture test:
```powershell
python audio_capture.py
```

This will list all available devices and test recording.

## 📁 Project Structure

```
speech2text/
├── main.py                  # Main GUI application
├── audio_capture.py         # Audio recording module
├── transcription.py         # Transcription engines
├── config.py               # Configuration
├── requirements.txt        # Python dependencies
├── setup.ps1              # Setup script
├── .env                   # User configuration (not in git)
├── .env.example          # Configuration template
│
├── recordings/           # Saved audio files (auto-created)
├── transcripts/         # Exported transcripts (auto-created)
├── models/             # Whisper model cache (auto-created)
└── venv/              # Virtual environment (auto-created)
```

## 🔌 Architecture

```
┌─────────────────┐
│   PyQt6 GUI     │
│   (main.py)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──────┐  │
│  Audio   │  │
│ Capture  │  │
│ (WASAPI) │  │
└──────────┘  │
         │    │
         │  ┌─▼───────────────┐
         │  │ Transcription   │
         │  │   Manager       │
         │  └─┬───┬───┬───────┘
         │    │   │   │
    ┌────▼────▼───▼───▼────┐
    │  Chrome  Whisper  Azure │
    │   (Web)  (Local) (Cloud)│
    └─────────────────────────┘
```

## 🎯 Use Cases

### Business Meetings
- Record team discussions
- Generate meeting minutes automatically
- Track action items and decisions
- Share transcripts with absent team members

### Interviews
- Transcribe job interviews
- Document user research sessions
- Create searchable interview archives

### Lectures & Training
- Capture webinars and presentations
- Create study notes from online courses
- Transcribe training sessions

### Personal
- Journal voice notes
- Transcribe podcasts
- Create content from brainstorming sessions

## 🔒 Privacy & Security

### Local Transcription (Whisper)
- ✅ All processing happens on your computer
- ✅ No data sent to external servers
- ✅ Complete privacy

### Cloud Transcription (Chrome/Azure)
- ⚠️ Audio sent to external servers
- ⚠️ Subject to provider's privacy policy
- ✅ Encrypted in transit (HTTPS/WSS)

**Recommendation:** Use Whisper for sensitive/confidential meetings.

## 🚧 Known Limitations

1. **System audio capture requires Stereo Mix or virtual audio cable**
   - Not all systems have this enabled by default
   
2. **Chrome Web Speech requires internet**
   - Not suitable for offline meetings
   
3. **Whisper is slow on CPU-only systems**
   - Consider GPU acceleration or smaller models
   
4. **No speaker diarization yet**
   - Cannot automatically identify different speakers
   - Planned for future version

5. **Windows-only**
   - WASAPI is Windows-specific
   - macOS/Linux support planned

## 🗺️ Roadmap

### Phase 2 (Future Enhancements)
- [ ] Speaker diarization (identify who's speaking)
- [ ] Real-time transcription for Whisper
- [ ] Multi-language support
- [ ] Keyword extraction and summarization
- [ ] Integration with calendar apps
- [ ] Audio playback with synchronized transcript
- [ ] Export to PDF, DOCX
- [ ] Custom vocabulary/terminology

## 📄 License

MIT License - Feel free to use, modify, and distribute.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 💬 Support

**Issues?** Check the Troubleshooting section above or open a GitHub issue.

**Questions?** Start a discussion in GitHub Discussions.

## 🙏 Credits

Built with:
- [OpenAI Whisper](https://github.com/openai/whisper) - Local transcription
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - Desktop GUI
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio capture
- [Azure Speech SDK](https://docs.microsoft.com/azure/cognitive-services/speech-service/) - Cloud transcription
- [Chrome Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API) - Browser-based transcription

---

**Made with ❤️ for better meeting productivity**
