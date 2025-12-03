# Meeting Recording Troubleshooting Guide

## Problem: Nothing Recorded When Others Are Talking

### Root Cause
When you're in an online meeting (Teams, Zoom, Google Meet, etc.), **other participants' voices come through your speakers/headphones**, NOT your microphone. To record them, you need **System Audio (Stereo Mix)** enabled.

### Solution Steps

#### 1. Enable Stereo Mix in Windows
1. Click the **"🔊 Open Sound Settings"** button in the app
2. Click **"Sound Control Panel"** on the right side
3. Go to the **"Recording"** tab
4. **Right-click** in the empty area → Check **"Show Disabled Devices"**
5. Find **"Stereo Mix"** or **"Wave Out Mix"**
6. **Right-click** on it → Select **"Enable"**
7. **Right-click** again → Select **"Set as Default Device"**
8. Click **"OK"**

#### 2. Configure App Settings
1. Restart the transcription app
2. Verify both checkboxes are checked:
   - ✅ **Record Microphone (Your Voice)**
   - ✅ **Record System Audio (Meeting Participants)** ← This is CRITICAL!
3. Check that **"Stereo Mix"** appears in the System Audio dropdown
4. Click **"🎙 Start Recording"** before your meeting starts

#### 3. Verify Recording Setup
- **CHANNELS = 2** (Stereo) - Now configured for better meeting audio
- **Both audio sources selected** - Default is both mic and system audio
- **Stereo Mix enabled** - Check the System Audio dropdown shows a device

### What Gets Recorded

| Audio Source                  | What It Captures                                                |
| ----------------------------- | --------------------------------------------------------------- |
| **Microphone**                | Your voice, your questions, your comments                       |
| **System Audio (Stereo Mix)** | Meeting participants' voices, screen share audio, shared videos |
| **Both (Recommended)**        | Complete meeting - everyone's voices!                           |

### Common Mistakes

❌ **Only microphone checked** → Only YOUR voice recorded  
❌ **Stereo Mix disabled** → Only YOUR voice recorded  
❌ **Started recording too late** → Missed beginning of meeting  
✅ **Both sources + Stereo Mix enabled** → Full meeting captured!

### Quick Test Before Meeting

1. Enable Stereo Mix (steps above)
2. Play a YouTube video or music
3. Click **"🎙 Start Recording"** in the app
4. Let it record for 10 seconds
5. Click **"⏹ Stop Recording"**
6. Click **"📁 Recording Manager"**
7. Double-click the recording to play it
8. You should hear the YouTube/music audio ✅

### Still Not Working?

1. **Check Sound Settings**
   - Right-click speaker icon in Windows taskbar
   - Sound Settings → Advanced sound options
   - Verify Stereo Mix is enabled and set as default recording device

2. **Update Audio Drivers**
   - Some Realtek audio drivers disable Stereo Mix
   - Download latest drivers from manufacturer

3. **Alternative: Virtual Audio Cable**
   - If Stereo Mix is not available, use VB-Audio Virtual Cable (free)
   - Download from: https://vb-audio.com/Cable/

4. **Meeting Platform Settings**
   - Teams: Settings → Devices → Make sure audio is not muted
   - Zoom: Settings → Audio → Test Speaker/Microphone
   - Ensure meeting audio is playing through speakers/headphones

### Best Practices

1. **Start recording BEFORE joining the meeting**
2. **Test your setup with a sample recording first**
3. **Keep System Audio checkbox ALWAYS checked for meetings**
4. **Use headphones to prevent audio feedback**
5. **Check that both audio devices are detected in dropdowns**

### Record Now, Transcribe Later Workflow

**NEW FEATURE**: You can now record meetings and transcribe them later when you have time!

#### How It Works:
1. **During Meeting**: Just record
   - Click "🎙 Start Recording"
   - Let the meeting happen
   - Click "⏹ Stop Recording"
   - ✅ Recording saved automatically!

2. **Later (When You Have Time)**: Transcribe
   - Click "📁 Recording Manager"
   - Select the recording you want to transcribe
   - Click "📝 Transcribe Selected"
   - Wait for transcription to complete

#### Benefits:
- ✅ Record multiple meetings back-to-back
- ✅ Transcribe during lunch break or after hours
- ✅ No need to wait 30+ minutes for transcription during meeting
- ✅ Queue multiple recordings for batch transcription
- ✅ Recordings are saved permanently until you delete them

#### Example Workflow:
```
9:00 AM - Record Meeting 1 (30 min)
10:00 AM - Record Meeting 2 (45 min)  
11:00 AM - Record Meeting 3 (60 min)
---
12:00 PM - Lunch: Transcribe all 3 meetings (queue them up!)
         - Total transcription time: ~60-90 minutes
         - You can leave it running and eat lunch
```

### Technical Details

- **Mono (1 channel)**: Only captures one audio stream - good for single speaker
- **Stereo (2 channels)**: Captures left + right audio - BETTER for meetings
- **Sample Rate**: 16000 Hz is optimal for Whisper transcription
- **Whisper Model**: Base model is fast and accurate for English

### Need Help?

If you're still having issues:
1. Take a screenshot of your Sound Control Panel → Recording tab
2. Check what devices appear in the app's Audio Source dropdowns
3. Verify the "Record System Audio" checkbox is checked
4. Test with both a local audio file and a meeting
