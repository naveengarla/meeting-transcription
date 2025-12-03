# Meeting Transcription & MoM Generator

A Windows desktop application for real-time meeting transcription and Minutes of Meeting (MoM) generation. Captures both microphone and system audio, transcribes speech to text, and exports formatted transcripts.

![Version](https://img.shields.io/badge/version-0.0.2-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

### Audio Capture
- **Microphone Recording** - Capture your voice during meetings
- **System Audio Loopback (Stereo Mix)** - Record audio from Teams, Zoom, or any application
- **Dual Channel Support** - Record both mic and system audio simultaneously
- **Auto-detection** - Automatically detects device capabilities (mono/stereo)
- **Built-in Audio Player** - Play recordings directly in the app with volume control

### Transcription Engines (3 Options)

#### 🚀 faster-whisper (Recommended - 6x faster!)
- ✅ **Blazing fast** - 6-7x real-time speed (2.7 min audio in 25 seconds!)
- ✅ **Offline** - Works without internet
- ✅ **High quality** - Same OpenAI Whisper models, optimized engine
- ✅ **Multi-threaded** - Uses 8 CPU cores efficiently
- ✅ **Low memory** - int8 quantization (~300 MB vs 800+ MB)
- ✅ **Private** - All processing local
- ✅ **Free** - Open source
- ✅ **99+ languages** - Including Telugu, Kannada, Hindi, Tamil, etc.
- ✅ **Performance logging** - Track speed and resource usage
- ❌ Requires CPU power (works great on i7-1270P and similar)

#### 🌐 Chrome Web Speech API
- ✅ **Free** - No API costs
- ✅ **Real-time** - See transcription as you speak
- ✅ **Good accuracy** - Google-powered
- ✅ **Easy setup** - No model downloads
- ❌ **Microphone only** - Cannot capture system audio/speakers
- ❌ **Not for meetings** - Cannot transcribe other participants (Teams, Zoom)
- ❌ **Dictation use case only** - For typing what you say, not recording meetings
- ❌ Requires internet connection
- ❌ Limited language support (primarily English)
- ❌ Privacy concern (audio sent to Google)

#### ☁️ Azure Speech Service (Enterprise)
- ✅ **Highly accurate** - Enterprise-grade
- ✅ **Fast** - Cloud processing
- ✅ **Reliable** - Microsoft infrastructure
- ❌ Requires Azure subscription
- ❌ API costs apply
- ❌ Requires internet

### Recording Management
- 📁 **Recording Manager** - Browse, play, and manage all recordings
- 🎵 **Built-in Player** - Play recordings with play/pause/stop/volume controls
- 📝 **Transcribe Later** - Record meetings now, transcribe when convenient
- 🔄 **Queue System** - Batch transcribe multiple recordings
- 🗑️ **Selective Deletion** - Delete individual or all recordings

### Performance Monitoring
- 📊 **Automatic Logging** - Performance metrics logged to `logs/performance_YYYYMM.jsonl`
- ⚡ **Speed Tracking** - Real-time factor, speed multiplier (e.g., "6.69x real-time")
- 💻 **Resource Monitoring** - CPU usage %, memory consumption
- 📈 **Analysis Tool** - `analyze_performance.py` for detailed reports
- 🕒 **Historical Data** - Monthly log files for trend analysis

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

#### Option 2: Record then Transcribe (faster-whisper - Recommended!)

**For real-time meetings:**
1. **Enable System Audio:**
   - Click "🔊 Open Sound Settings" button
   - Enable "Stereo Mix" (see troubleshooting guide below)
   - Click "🔄 Refresh Devices"

2. **Select audio sources:**
   - ✓ Microphone - Check to record your voice
   - ✓ System Audio - Check to record Teams/Zoom participants

3. **Select "Whisper (base, 8 cores)" engine**

4. **Click "🎙 Start Recording"** BEFORE joining meeting

5. **Conduct your meeting** (record everything)

6. **Click "⏹ Stop Recording"** when done

7. **Choose one:**
   - **Transcribe now:** Click "📝 Transcribe" (wait ~4-5 min for 30-min meeting)
   - **Transcribe later:** Click "📁 Recording Manager" → select recording → "📝 Transcribe Selected"

**Best for:** Offline transcription, high accuracy, longer meetings, batch processing

#### Option 3: Record Now, Transcribe Later (NEW!)

**The productivity workflow:**
1. **Morning:** Record 3 meetings back-to-back (just click record/stop)
2. **Lunch:** Open Recording Manager, select all 3, queue them for transcription
3. **Afternoon:** Review all transcripts, export to Markdown

**Benefits:**
- No waiting during meetings
- Batch process multiple recordings
- Transcribe when you have time
- Queue fills automatically

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
CHANNELS=2             # Preferred channels (auto-detects device capability)
CHUNK_SIZE=1024        # Audio buffer size

# Preferred audio devices (leave empty for system default)
PREFERRED_MICROPHONE=Jabra Evolve2 30 SE
PREFERRED_SPEAKER=Stereo Mix
```

### Transcription Settings

```python
TRANSCRIPTION_MODE=whisper     # Default engine: chrome, whisper, or azure
WHISPER_MODEL=base             # Model size: tiny, base, small, medium, large
WHISPER_LANGUAGE=en            # Language code or 'auto' for auto-detect
WHISPER_TASK=transcribe        # 'transcribe' or 'translate' (to English)
```

**faster-whisper Model Performance (on i7-1270P, 8 cores):**
| Model  | Size   | Speed       | Memory | Accuracy  | Use Case                  |
| ------ | ------ | ----------- | ------ | --------- | ------------------------- |
| tiny   | 75 MB  | ~10-12x RT  | ~200MB | Good      | Quick drafts, testing     |
| base   | 142 MB | **6-7x RT** | ~300MB | Better    | **Recommended (default)** |
| small  | 466 MB | ~4-5x RT    | ~500MB | Great     | High accuracy needed      |
| medium | 1.5 GB | ~2-3x RT    | ~1GB   | Excellent | Best quality              |
| large  | 2.9 GB | ~1-2x RT    | ~2GB   | Best      | Maximum accuracy          |

**RT = Real-time** (e.g., 6x RT = 30-min meeting transcribed in 5 min)

**Language Support:**
- **99+ languages** supported including:
  - Indian languages: Telugu (te), Kannada (kn), Hindi (hi), Tamil (ta), Malayalam (ml), etc.
  - European: English (en), Spanish (es), French (fr), German (de), etc.
  - Asian: Chinese (zh), Japanese (ja), Korean (ko), etc.
- Set `WHISPER_LANGUAGE=auto` for automatic detection
- See `docs/LANGUAGE_SUPPORT.md` for complete list and examples

## 🔧 Troubleshooting

### Common Issues

#### "No loopback device found" / Can't record system audio

**Solution 1: Enable Stereo Mix (Built-in Windows)**
1. Right-click speaker icon in taskbar → **Sounds**
2. Go to **Recording** tab
3. Right-click empty area → **Show Disabled Devices**
4. Right-click **Stereo Mix** → **Enable**
5. Restart the app and refresh devices

**Detailed Guide:** See `docs/MEETING_RECORDING_GUIDE.md` for step-by-step instructions with screenshots.

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

#### Whisper transcription is slow

**Solutions:**
- **Use faster-whisper (default):** Already 6-7x real-time speed with base model
- **Try smaller model:** `WHISPER_MODEL=tiny` (10-12x speed) for quick drafts
- **Check performance:** Run `python analyze_performance.py` to see actual metrics
- **Verify CPU usage:** 190-215% CPU is normal and safe (uses ~2 cores efficiently)
- **Model comparison:** 
  - base model: 6-7x RT (30-min meeting → ~4.5 min transcription)
  - small model: 4-5x RT (higher accuracy, slower)
  - tiny model: 10-12x RT (fastest, lower accuracy)

**Performance is normal if:** You see speed multipliers 6-7x for base, audio processes faster than real-time

**Performance is slow if:** Transcription takes longer than audio duration - check logs with `analyze_performance.py`

#### Incorrect language detection

**Solutions:**
- Set specific language: `WHISPER_LANGUAGE=te` (for Telugu), `en` (English), etc.
- Use auto-detection: `WHISPER_LANGUAGE=auto` (works well for single-language audio)
- Check detected language in logs: `logs/performance_YYYYMM.jsonl`
- See all supported languages: `docs/LANGUAGE_SUPPORT.md`

#### High memory usage

**Solutions:**
- Use smaller model: `tiny` (~200MB), `base` (~300MB), `small` (~500MB)
- faster-whisper already uses int8 quantization (default, memory efficient)
- Close other applications during transcription
- Check actual usage: Run `python analyze_performance.py` for statistics

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
- Use larger Whisper model: `WHISPER_MODEL=small` or `medium`
- Set correct language: `WHISPER_LANGUAGE=en` (or your language code)
- Check language detection in logs
- Switch to Azure Speech Service for best accuracy

#### Performance monitoring

**Check transcription metrics:**
```powershell
python analyze_performance.py
```

**What you'll see:**
- Speed multiplier (6-7x is excellent for base model)
- Total audio processed vs. time taken
- Time saved calculations
- CPU and memory usage statistics
- Language detection breakdown
- Recent transcription history

**Log files location:** `logs/performance_YYYYMM.jsonl` (auto-created monthly)

### Debug Audio Devices

Run the audio capture test:
```powershell
python audio_capture.py
```

This will list all available devices and test recording.

## 📁 Project Structure

```
speech2text/
├── src/
│   ├── main.py              # Main GUI application
│   ├── audio_capture.py     # Audio recording module
│   ├── transcription.py     # Transcription engines (faster-whisper)
│   └── config.py            # Configuration management
│
├── docs/
│   ├── ARCHITECTURE.md              # System design
│   ├── LANGUAGE_SUPPORT.md          # 99+ supported languages
│   ├── MEETING_RECORDING_GUIDE.md   # Stereo Mix setup guide
│   ├── REQUIREMENTS.md              # Project requirements
│   ├── ROADMAP.md                   # Feature roadmap
│   └── concepts/                    # Technical tutorials (24,000+ words)
│       ├── INDEX.md                 # Learning roadmap
│       ├── speech.md                # Digital speech processing
│       ├── segments.md              # Segmentation & alignment
│       ├── transcription.md         # ASR theory & Whisper
│       ├── models.md                # Neural networks & optimization
│       └── multithreading.md        # Python concurrency
│
├── logs/
│   └── performance_YYYYMM.jsonl    # Auto-generated performance metrics
│
├── recordings/          # Saved audio files (auto-created)
├── transcripts/         # Exported transcripts (auto-created)
├── models/             # Whisper model cache (auto-created)
├── venv/              # Virtual environment (auto-created)
│
├── run.py                      # Quick launch script
├── analyze_performance.py      # Performance metrics analyzer
├── requirements.txt            # Python dependencies
├── setup.ps1                   # Automated setup script
├── .env                       # User configuration (not in git)
└── .env.example              # Configuration template
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
   - See `docs/MEETING_RECORDING_GUIDE.md` for setup instructions
   
2. **Chrome Web Speech requires internet**
   - Not suitable for offline meetings
   
3. **Whisper model downloads required on first use**
   - Model sizes: 75MB (tiny) to 2.9GB (large)
   - Models cached locally after first download
   - faster-whisper downloads optimized models automatically
   
4. **No speaker diarization yet**
   - Cannot automatically identify different speakers
   - Planned for future version

5. **Windows-only**
   - WASAPI is Windows-specific
   - macOS/Linux support planned

## 📊 Performance Benchmarks

**Test Environment:**
- CPU: Intel i7-1270P (12 cores, 16 threads)
- Model: base (recommended)
- Engine: faster-whisper 1.2.1 with int8 quantization
- Configuration: 8 worker threads, VAD filtering enabled

**Actual Results:**

| Audio Length | Transcription Time | Speed Multiplier | CPU Usage | Memory  |
| ------------ | ------------------ | ---------------- | --------- | ------- |
| 13.6 seconds | 5.3 seconds        | 2.55x RT         | ~190%     | 310 MB  |
| 164.7 sec    | 24.6 seconds       | **6.69x RT**     | ~215%     | 324 MB  |
| 2.7 min      | ~25 seconds        | ~6.5x RT         | ~200%     | ~320 MB |

**Projections:**

| Meeting Length | Transcription Time (base model) | Time Saved      |
| -------------- | ------------------------------- | --------------- |
| 15 minutes     | ~2.3 minutes                    | 12.7 min (85%)  |
| 30 minutes     | ~4.6 minutes                    | 25.4 min (85%)  |
| 1 hour         | ~9.2 minutes                    | 50.8 min (85%)  |
| 2 hours        | ~18.4 minutes                   | 101.6 min (85%) |

**Performance Notes:**
- **Speed increases with audio length** due to model warmup overhead
- **CPU usage is safe:** ~200% = 2 cores actively used (out of 8 configured)
- **Memory efficient:** ~320 MB peak usage regardless of audio length
- **Linear scaling:** Longer audio processes proportionally faster
- **Best practices:** 
  - Use base model for optimal speed/accuracy balance (6-7x)
  - Use tiny model for quick drafts (10-12x)
  - Use small/medium for higher accuracy (4-5x / 2-3x)

**Check your performance:**
```powershell
python analyze_performance.py
```

Shows detailed metrics, statistics, and recent transcription history from `logs/performance_YYYYMM.jsonl`.

## 🗺️ Roadmap

### Phase 2 (Future Enhancements)
- [ ] Speaker diarization (identify who's speaking)
- [ ] Real-time transcription for Whisper
- [ ] Multi-language support enhancement (already supports 99+ languages)
- [ ] Keyword extraction and summarization
- [ ] Integration with calendar apps
- [ ] Audio playback with synchronized transcript (basic player already included)
- [ ] Export to PDF, DOCX
- [ ] Custom vocabulary/terminology
- [ ] GPU acceleration detection and auto-configuration

### Completed ✅
- [x] faster-whisper integration (v0.0.2)
- [x] Performance monitoring and logging (v0.0.2)
- [x] Built-in audio player (v0.0.2)
- [x] Recording queue manager (v0.0.2)
- [x] Batch transcription support (v0.0.2)
- [x] Stereo Mix support for meeting recording (v0.0.2)

## 📄 License

MIT License - Feel free to use, modify, and distribute.

## 📚 Documentation

### Project Documentation
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and implementation
- [REQUIREMENTS.md](docs/REQUIREMENTS.md) - Project requirements and specifications
- [ROADMAP.md](docs/ROADMAP.md) - Feature roadmap and development plans
- [LANGUAGE_SUPPORT.md](docs/LANGUAGE_SUPPORT.md) - Complete list of 99+ supported languages
- [MEETING_RECORDING_GUIDE.md](docs/MEETING_RECORDING_GUIDE.md) - How to set up Stereo Mix

### Technical Tutorials (/docs/concepts)

**Comprehensive technical guides** (24,000+ words) covering the fundamentals needed to build advanced speech recognition applications:

- **[INDEX.md](docs/concepts/INDEX.md)** - Learning roadmap with prerequisites and resources
- **[speech.md](docs/concepts/speech.md)** - Digital Speech Processing
  - Sampling theory (Nyquist theorem)
  - Quantization & bit depth
  - Frequency domain analysis (FFT, STFT, mel spectrograms)
  - Audio processing (VAD, normalization, noise reduction)
  - Audio codecs & psychoacoustics
  
- **[segments.md](docs/concepts/segments.md)** - Segmentation & Alignment
  - Voice Activity Detection (energy-based, WebRTC, Silero VAD)
  - Alignment algorithms (DTW, HMM, neural CTC)
  - CTC theory (loss function, decoding)
  - Attention mechanisms (Bahdanau, Luong, multi-head)
  - Speaker diarization (embeddings, clustering, pyannote.audio)
  
- **[transcription.md](docs/concepts/transcription.md)** - Automatic Speech Recognition
  - ASR evolution (1952-2022: HMM-GMM → Neural → Transformers)
  - Classical ASR pipeline (Kaldi)
  - End-to-end models (CTC, LAS, DeepSpeech)
  - Transformer-based ASR (Conformer, wav2vec 2.0)
  - **Whisper architecture deep dive** (encoder/decoder, special tokens, model sizes)
  - Decoding strategies (greedy, beam search)
  - Evaluation metrics (WER, CER, RTF)
  
- **[models.md](docs/concepts/models.md)** - Neural Networks & Optimization
  - Neural network fundamentals (layers, activations, transformers)
  - Encoder-decoder architecture (complete PyTorch implementations)
  - Whisper model internals (74M parameter breakdown, vocabulary, tokenization)
  - **Model quantization** (FP32→INT8, PTQ/QAT, CTranslate2)
  - Inference optimization (ONNX, TensorRT, OpenVINO)
  - Model compression (pruning, distillation, low-rank factorization)
  - Hardware acceleration (CPU, GPU, edge devices)
  
- **[multithreading.md](docs/concepts/multithreading.md)** - Python Concurrency
  - Concurrency fundamentals (I/O-bound vs CPU-bound)
  - **Python GIL** (what it is, when released, bypassing strategies)
  - Threading (locks, semaphores, events, producer-consumer)
  - Multiprocessing (processes, IPC, batch transcription)
  - Async/await (asyncio, WebSockets)
  - **GUI threading** (PyQt QThread with real examples from this project)
  - Real-time audio streaming (ring buffers, latency optimization)
  - Profiling & debugging

**All tutorials include:**
- Mathematical foundations and formulas
- Complete, runnable Python code examples
- References to academic papers and tools
- Production-ready patterns and best practices

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
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Optimized Whisper transcription (CTranslate2)
- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper models
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - Desktop GUI framework
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio capture (WASAPI)
- [pygame](https://www.pygame.org/) - Built-in audio player
- [psutil](https://github.com/giampaolo/psutil) - Performance monitoring
- [Azure Speech SDK](https://docs.microsoft.com/azure/cognitive-services/speech-service/) - Cloud transcription (optional)
- [Chrome Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API) - Browser-based transcription (optional)

---

**Made with ❤️ for better meeting productivity**
