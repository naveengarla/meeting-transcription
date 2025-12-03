# Quick Start Guide - Meeting Transcription App

## First-Time Setup (5 minutes)

### 1. Install Python
- Download Python 3.10+ from https://www.python.org/downloads/
- ✅ During installation, check "Add Python to PATH"

### 2. Setup the App
Open PowerShell in this folder and run:
```powershell
.\setup.ps1
```

This installs everything automatically.

### 3. Run the App
```powershell
.\venv\Scripts\Activate.ps1
python run.py
```

## Quick Usage

### For Quick Meetings (Chrome Method - Recommended)
1. Select **"Chrome Web Speech API"** from dropdown
2. Click **"Open Chrome Speech Recognition"**
3. Click **"Start Listening"** in browser
4. Speak - transcript appears in real-time!
5. Click **"Save as Markdown"** when done

**Pros:** Free, fast, no setup
**Cons:** Requires internet

### For Private/Offline Meetings (Whisper Method)
1. Select **"Whisper (base)"** from dropdown
2. Check ✓ Microphone (and ✓ System Audio if in Teams call)
3. Click **"Start Recording"**
4. Conduct your meeting
5. Click **"Stop Recording"**
6. Click **"Transcribe"** (wait 1-2 minutes)
7. Click **"Save as Markdown"**

**Pros:** Offline, private, good accuracy
**Cons:** Slower, uses CPU/GPU

## Troubleshooting

### "No loopback device found"
**Fix:** Enable Stereo Mix
1. Right-click speaker icon → Sounds
2. Recording tab → Right-click → Show Disabled Devices
3. Right-click Stereo Mix → Enable

### Whisper is very slow
**Fix:** Use smaller model
1. Edit `.env` file
2. Change: `WHISPER_MODEL=tiny`
3. Restart app

### Chrome speech not working
**Fix:** Use Chrome or Edge browser (not Firefox)

## Common Tasks

### Record a Teams/Zoom Meeting
1. Enable System Audio (see above)
2. Check ✓ System Audio in app
3. Start recording before joining call
4. Stop when meeting ends
5. Transcribe and save

### Save Transcripts to Specific Folder
When saving, navigate to your desired folder in the save dialog.
Default location: `c:\Src2\speech2text\transcripts\`

### Change Whisper Model Quality
Edit `.env`:
- `WHISPER_MODEL=tiny` - Fastest, lower quality
- `WHISPER_MODEL=base` - Balanced (recommended)
- `WHISPER_MODEL=small` - Slower, better quality
- `WHISPER_MODEL=medium` - Best quality, very slow

## Tips

✅ **For best accuracy:**
- Use external microphone
- Reduce background noise
- Speak clearly at moderate pace

✅ **For long meetings:**
- Use Chrome method (real-time)
- Or use Whisper with `tiny` model

✅ **For sensitive meetings:**
- Use Whisper (offline, private)
- Disable internet connection

## Getting Help

- **Check:** README.md (detailed documentation)
- **Test:** Run `python test_audio.py` to verify audio
- **Test:** Run `python test_transcription.py` to verify transcription
- **Issues:** See "Troubleshooting" section in README.md

## Next Steps

Once comfortable with basic usage:
1. Try Azure Speech Service for best accuracy (requires API key)
2. Customize timestamps and export formats in config.py
3. Set up keyboard shortcuts for quick recording

**Happy transcribing! 🎤📝**
