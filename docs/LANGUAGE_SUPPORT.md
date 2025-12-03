# Multi-Language Support Guide

## Supported Languages

The Meeting Transcription app supports **99+ languages** through OpenAI Whisper, including all major Indian languages.

### Indian Languages Supported

| Language      | Code | Script | Notes           |
| ------------- | ---- | ------ | --------------- |
| **Telugu**    | `te` | తెలుగు    | Fully supported |
| **Kannada**   | `kn` | ಕನ್ನಡ   | Fully supported |
| **Hindi**     | `hi` | हिन्दी    | Fully supported |
| **Tamil**     | `ta` | தமிழ்    | Fully supported |
| **Malayalam** | `ml` | മലയാളം   | Fully supported |
| **Bengali**   | `bn` | বাংলা     | Fully supported |
| **Marathi**   | `mr` | मराठी    | Fully supported |
| **Gujarati**  | `gu` | ગુજરાતી   | Fully supported |
| **Punjabi**   | `pa` | ਪੰਜਾਬੀ    | Fully supported |
| **Urdu**      | `ur` | اردو   | Fully supported |

### Other Popular Languages

- English (`en`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Chinese (`zh`)
- Japanese (`ja`)
- Korean (`ko`)
- Arabic (`ar`)
- And 80+ more!

## How to Use

### Option 1: Auto-Detect (Recommended)

1. Select **"Auto-detect"** from the Language dropdown
2. Whisper will automatically identify the spoken language
3. Works well for single-language meetings

**Best for:** Most use cases, mixed-language environments

### Option 2: Specify Language

1. Select your language from the dropdown (e.g., "Telugu (తెలుగు)")
2. Record and transcribe as usual
3. Faster and more accurate for single-language content

**Best for:** Known language, better accuracy needed

## Telugu Transcription Example

### Recording Telugu Speech

1. **Select Language**: Choose "Telugu (తెలుగు)" from Language dropdown
2. **Record**: Click Start Recording and speak in Telugu
3. **Transcribe**: Click Transcribe
4. **Export**: Save as Markdown or TXT

### Sample Output (Telugu)

```
[00:05] నమస్కారం, ఈ రోజు మన మీటింగ్ లో మనం ప్రాజెక్ట్ స్టేటస్ గురించి చర్చిద్దాం
[00:12] మొదట, మనం కంప్లీట్ చేసిన టాస్క్స్ గురించి చూద్దాం
```

## Kannada Transcription Example

### Recording Kannada Speech

1. **Select Language**: Choose "Kannada (ಕನ್ನಡ)" from dropdown
2. **Record and transcribe**

### Sample Output (Kannada)

```
[00:05] ನಮಸ್ಕಾರ, ಇಂದು ನಮ್ಮ ಮೀಟಿಂಗ್‌ನಲ್ಲಿ ಪ್ರಾಜೆಕ್ಟ್ ಸ್ಥಿತಿಯ ಬಗ್ಗೆ ಚರ್ಚಿಸೋಣ
[00:12] ಮೊದಲು, ನಾವು ಪೂರ್ಣಗೊಳಿಸಿದ ಕಾರ್ಯಗಳನ್ನು ನೋಡೋಣ
```

## Translation to English

Want to transcribe non-English speech and get English translation?

### Configuration

Edit `.env` file:
```env
WHISPER_TASK=translate
```

Or keep default `transcribe` for original language.

### How It Works

- **transcribe**: Output in original language (Telugu → Telugu text)
- **translate**: Translate to English (Telugu speech → English text)

### Example: Telugu to English

**Speech (Telugu):** "నమస్కారం, ఈ రోజు మన మీటింగ్"

**transcribe mode:** "నమస్కారం, ఈ రోజు మన మీటింగ్"

**translate mode:** "Hello, today our meeting"

## Mixed Language Meetings

### Scenario: English + Telugu/Kannada

For meetings with multiple languages (e.g., code-switching between English and Telugu):

1. Use **"Auto-detect"** mode
2. Whisper will detect and transcribe both languages
3. Each segment may be in different language based on speech

### Example Output

```
[00:05] Hello everyone, welcome to the meeting
[00:10] ఈ రోజు agenda ఏమిటంటే project status review
[00:18] మొదట we'll discuss the technical challenges
```

## Tips for Best Accuracy

### For Indian Languages (Telugu/Kannada/Hindi etc.)

1. **Clear Speech**: Speak clearly and at moderate pace
2. **Reduce Background Noise**: Use a good microphone
3. **Select Language**: Specify language instead of auto-detect for better accuracy
4. **Model Size**: Use `base` or `small` model for good balance
   - `tiny`: Fastest, lower accuracy
   - `base`: **Recommended** for Indian languages
   - `small`: Better accuracy, slower
   - `medium/large`: Best accuracy, requires more RAM

### Configuration for Indian Languages

Edit `.env`:
```env
WHISPER_MODEL=small
WHISPER_LANGUAGE=te     # Telugu
# or
WHISPER_LANGUAGE=kn     # Kannada
# or
WHISPER_LANGUAGE=auto   # Auto-detect
```

## Limitations

1. **Chrome Web Speech API**: Only supports English (and some popular languages, not Telugu/Kannada)
   - **Solution**: Use Whisper engine for Indian languages

2. **Azure Speech Service**: Supports Telugu, Kannada, but requires API configuration
   - Telugu: Supported
   - Kannada: Supported
   - Requires Azure subscription

3. **Accuracy**: May vary based on:
   - Accent and dialect
   - Audio quality
   - Background noise
   - Speaking speed

## Recommended Setup for Telugu/Kannada

### Best Configuration

```env
TRANSCRIPTION_MODE=whisper
WHISPER_MODEL=small
WHISPER_LANGUAGE=te      # or kn for Kannada
WHISPER_TASK=transcribe
```

### In GUI

1. Engine: **Whisper (small)**
2. Language: **Telugu (తెలుగు)** or **Kannada (ಕನ್ನಡ)**
3. Record with good microphone
4. Transcribe

## Full Language List

Whisper supports these languages:

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, **Kannada**, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, **Tamil**, **Telugu**, Thai, Turkish, Ukrainian, **Urdu**, Vietnamese, Welsh

**Total: 99+ languages**

## Getting Help

If you experience issues with Telugu/Kannada transcription:

1. Check microphone is working
2. Try `small` or `base` model
3. Ensure clear speech
4. Test with sample audio
5. Check logs for errors

For questions, open an issue on GitHub!
