# Digital Speech Processing Fundamentals

## Table of Contents
1. [Introduction to Digital Audio](#1-introduction-to-digital-audio)
2. [Sampling Theory](#2-sampling-theory)
3. [Quantization & Bit Depth](#3-quantization--bit-depth)
4. [Frequency Domain Analysis](#4-frequency-domain-analysis)
5. [Audio Processing Techniques](#5-audio-processing-techniques)
6. [Audio Codecs & Formats](#6-audio-codecs--formats)
7. [Psychoacoustics](#7-psychoacoustics)
8. [Practical Implementation](#8-practical-implementation)

---

## 1. Introduction to Digital Audio

### Analog vs Digital
- **Analog**: Continuous voltage representing sound pressure waves
- **Digital**: Discrete samples representing amplitude at fixed time intervals
- **A/D Conversion**: Microphone → ADC → Digital samples
- **D/A Conversion**: Digital samples → DAC → Speaker

### Sound as a Physical Phenomenon
- **Sound Wave**: Longitudinal pressure wave through medium (air)
- **Frequency**: Pitch (Hz) - Human hearing: 20 Hz to 20 kHz
- **Amplitude**: Loudness (dB SPL) - Threshold of hearing: 0 dB, pain: ~120 dB
- **Timbre**: Harmonic content, what makes a piano sound different from violin

---

## 2. Sampling Theory

### Nyquist-Shannon Sampling Theorem
**Theorem**: To perfectly reconstruct a signal, sample at **≥2× the highest frequency**.

```
f_sample ≥ 2 × f_max
```

**Examples**:
- CD Audio: 44.1 kHz (captures up to 22.05 kHz, beyond human hearing)
- Telephony: 8 kHz (captures up to 4 kHz, sufficient for speech intelligibility)
- **Speech Recognition**: 16 kHz (captures up to 8 kHz, optimal for ASR models)

### Aliasing
When sampling below Nyquist rate, high frequencies "fold back" into lower frequencies, causing distortion.

**Solution**: **Anti-aliasing filter** (low-pass filter) before ADC removes frequencies above f_sample/2.

```python
# Example: Why 16 kHz for speech?
# Human speech fundamental frequencies: 85-255 Hz (male/female)
# Formants (vowel resonances): 200 Hz - 8 kHz
# Consonants (fricatives): up to 10 kHz
# → 16 kHz captures essential speech information while keeping file size reasonable
```

### Practical Sample Rates
| Rate      | Use Case                     | Max Frequency |
| --------- | ---------------------------- | ------------- |
| 8 kHz     | Telephony, low-quality voice | 4 kHz         |
| 16 kHz    | **Speech recognition (ASR)** | 8 kHz         |
| 22.05 kHz | Low-quality music            | 11 kHz        |
| 44.1 kHz  | CD audio, high-quality       | 22.05 kHz     |
| 48 kHz    | Professional audio/video     | 24 kHz        |
| 96 kHz+   | Studio recording, audiophile | 48+ kHz       |

---

## 3. Quantization & Bit Depth

### Bit Depth
**Definition**: Number of bits used to represent each sample's amplitude.

**Dynamic Range**: `DR = 6.02 × bit_depth` (dB)

| Bit Depth  | Levels       | Dynamic Range | Use Case           |
| ---------- | ------------ | ------------- | ------------------ |
| 8-bit      | 256          | ~48 dB        | Retro games, µ-law |
| **16-bit** | **65,536**   | **~96 dB**    | **CD, Speech ASR** |
| 24-bit     | 16.7 million | ~144 dB       | Professional audio |
| 32-bit     | 4.3 billion  | ~192 dB       | Floating point     |

### Quantization Noise
Rounding continuous amplitude to discrete levels introduces noise.

**Signal-to-Quantization-Noise Ratio (SQNR)**:
```
SQNR ≈ 6.02 × N + 1.76 dB  (N = bit depth)
```

For 16-bit: SQNR ≈ 98 dB (inaudible for most applications)

### Integer vs Floating Point

**Integer (PCM - Pulse Code Modulation)**:
- Fixed range: [-32768, 32767] for 16-bit signed
- No overhead, fast processing
- Standard for WAV files

**Floating Point (32-bit float)**:
- Range: [-1.0, 1.0] with high precision
- Used internally by ML models (Whisper, wav2vec)
- Prevents clipping during processing

```python
# Conversion: int16 → float32
import numpy as np
audio_int16 = np.array([0, 16384, 32767, -32768], dtype=np.int16)
audio_float32 = audio_int16.astype(np.float32) / 32768.0
# Result: [0.0, 0.5, 0.999969, -1.0]
```

---

## 4. Frequency Domain Analysis

### Fourier Transform
**Time Domain** (waveform) ⟷ **Frequency Domain** (spectrum)

**Discrete Fourier Transform (DFT)**:
```
X[k] = Σ(n=0 to N-1) x[n] · e^(-j2πkn/N)
```

**Fast Fourier Transform (FFT)**: Efficient algorithm, O(N log N) vs O(N²)

### Short-Time Fourier Transform (STFT)
Audio is non-stationary (frequency content changes over time).

**Solution**: Divide audio into overlapping windows, apply FFT to each.

```python
import numpy as np
from scipy.signal import stft

# Parameters
fs = 16000          # Sample rate
window_size = 512   # 32 ms at 16 kHz
hop_length = 256    # 50% overlap

f, t, Zxx = stft(audio, fs, nperseg=window_size, noverlap=window_size-hop_length)
spectrogram = np.abs(Zxx)  # Magnitude
```

**Output**: 2D spectrogram (frequency vs time)

### Mel Scale & MFCCs
**Problem**: Human hearing is non-linear (we perceive pitch logarithmically).

**Mel Scale**: Perceptual frequency scale matching human auditory system.
```
mel(f) = 2595 × log₁₀(1 + f/700)
```

**Mel Spectrogram**:
1. Compute STFT → Power spectrogram
2. Apply mel filterbank (triangular filters spaced on mel scale)
3. Sum energy in each mel band

**MFCCs (Mel-Frequency Cepstral Coefficients)**:
1. Mel spectrogram → Log amplitude
2. Apply Discrete Cosine Transform (DCT)
3. Keep first 13-40 coefficients

**Why MFCCs?**
- Compact representation (13 numbers vs 512-dim spectrum)
- Decorrelated features
- Used in classical ASR (HMM-GMM systems)

**Modern ASR**: Use **log-mel spectrograms** directly (80-128 mel bins) as input to neural networks (Whisper, wav2vec).

---

## 5. Audio Processing Techniques

### Normalization
**Peak Normalization**: Scale to use full bit depth range.
```python
audio = audio / np.max(np.abs(audio))  # Now in [-1, 1]
```

**RMS Normalization**: Target specific loudness.
```python
target_rms = 0.1
current_rms = np.sqrt(np.mean(audio**2))
audio = audio * (target_rms / current_rms)
```

### Voice Activity Detection (VAD)
Detect speech vs silence/noise.

**Energy-based** (simple):
```python
frame_energy = np.sum(frame**2)
is_speech = frame_energy > threshold
```

**Model-based** (robust):
- WebRTC VAD (Google's algorithm)
- Silero VAD (neural network)
- Whisper's built-in VAD

**Benefits**:
- Reduce false positives (transcribing silence)
- Save compute (skip non-speech frames)
- Improve segmentation

### Noise Reduction
**Spectral Subtraction**:
1. Estimate noise spectrum from silent regions
2. Subtract from signal spectrum
3. Reconstruct time-domain signal

**Wiener Filtering**: Optimal filter minimizing MSE.

**Deep Learning**: 
- RNNoise (Mozilla)
- NSNet (Microsoft)
- End-to-end neural denoisers

### Pre-emphasis
Boost high frequencies (speech has energy concentrated in low frequencies).

```python
# FIR filter: y[n] = x[n] - α·x[n-1], α ≈ 0.97
pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
```

**Why?** Improves SNR for high-frequency consonants (s, f, th).

---

## 6. Audio Codecs & Formats

### Uncompressed: WAV (PCM)
**Structure**: RIFF container with raw PCM data.
- **Pros**: Lossless, simple, universally supported
- **Cons**: Large file size (16-bit, 16kHz mono ≈ 2 MB/min)
- **Use**: Archival, processing, ASR input

### Lossless Compression: FLAC
- **Compression**: ~50% size reduction vs WAV
- **Decoding**: Bit-perfect reconstruction
- **Use**: Audio archival, music libraries

### Lossy Compression
| Codec | Compression | Quality   | Latency  | Use Case              |
| ----- | ----------- | --------- | -------- | --------------------- |
| MP3   | 10-20:1     | Good      | Low      | Music distribution    |
| AAC   | 10-20:1     | Better    | Low      | Streaming, podcasts   |
| Opus  | 20-50:1     | Excellent | Very low | VoIP, real-time       |
| Speex | 10-30:1     | Good      | Low      | Voice-only (obsolete) |

**Perceptual Coding**: Exploits psychoacoustic masking to discard inaudible information.

**ASR Considerations**:
- Use lossless (WAV/FLAC) for best accuracy
- Opus at high bitrate (≥32 kbps) acceptable for real-time
- Avoid low-bitrate MP3 (<64 kbps) - artifacts degrade WER

---

## 7. Psychoacoustics

### Critical Bands
Human ear has ~24 critical bands (bark scale), not uniform frequency resolution.

**Implication**: Mel scale, perceptual codecs, audio masking.

### Masking
**Frequency Masking**: Loud tone masks nearby quiet tones.
**Temporal Masking**: Sound masked just before/after loud sound.

**Codec Exploitation**: Discard masked frequencies → compression.

### Loudness Perception
**Equal Loudness Contours** (Fletcher-Munson): Frequency-dependent sensitivity.
- Most sensitive: 2-5 kHz (speech range)
- Reduced sensitivity: <200 Hz, >10 kHz

**Implications**: 
- Phone audio (300-3400 Hz) sacrifices bass/treble but retains intelligibility
- ASR models focus on 200 Hz - 8 kHz

---

## 8. Practical Implementation

### Capturing Audio (Python)
```python
import sounddevice as sd
import numpy as np

# Record 5 seconds of audio
fs = 16000  # Sample rate
duration = 5  # seconds
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()

# Convert to float32 for processing
audio_float = audio.astype(np.float32) / 32768.0
```

### Computing Mel Spectrogram
```python
import librosa

# Load audio
audio, sr = librosa.load('speech.wav', sr=16000, mono=True)

# Mel spectrogram (80 mel bins, 25ms window, 10ms hop)
mel_spec = librosa.feature.melspectrogram(
    y=audio, 
    sr=sr, 
    n_fft=400,      # 25 ms window
    hop_length=160, # 10 ms hop
    n_mels=80,
    fmin=0,
    fmax=8000
)

# Convert to log scale (dB)
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
```

### VAD with Silero
```python
import torch
from scipy.io import wavfile

# Load Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, *_) = utils

# Read audio
wav = read_audio('speech.wav')

# Get speech timestamps
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
# Output: [{'start': 1600, 'end': 48000}, {'start': 56000, 'end': 80000}, ...]
```

### Audio Normalization Pipeline
```python
def preprocess_audio(audio, sr=16000):
    """Production-ready preprocessing for ASR."""
    # 1. Convert to mono (if stereo)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # 2. Resample to 16 kHz (if needed)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # 3. Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    
    # 4. Peak normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # 5. Optional: Pre-emphasis
    # audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    
    return audio
```

---

## Key Takeaways for ASR Development

1. **16 kHz, 16-bit PCM** is the industry standard for speech recognition.
2. **Mel spectrograms** (80-128 bins) are the input to modern neural ASR models.
3. **VAD** is essential for production systems - improves accuracy and reduces compute.
4. **Normalization** prevents clipping and ensures consistent model input.
5. **Float32** is used internally by ML frameworks, even if input is int16.
6. Understand **sampling theorem** to avoid aliasing artifacts.
7. **Psychoacoustics** explains why lossy codecs work and where they fail for ASR.

---

## Further Reading

### Books
- *Digital Signal Processing* - Oppenheim & Schafer
- *Speech and Audio Signal Processing* - Ben Gold, Nelson Morgan
- *The Scientist and Engineer's Guide to Digital Signal Processing* - Steven W. Smith (free online)

### Papers
- "Mel Frequency Cepstral Coefficients for Music Modeling" (Logan, 2000)
- "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition" (Rabiner, 1989)

### Tools & Libraries
- **librosa**: Audio analysis in Python
- **torchaudio**: PyTorch audio processing
- **soundfile**: Read/write audio files
- **pydub**: High-level audio manipulation
- **ffmpeg**: Swiss-army knife for audio/video

### Datasets
- **LibriSpeech**: 1000h English audiobooks
- **Common Voice**: Multilingual crowdsourced speech
- **VCTK**: Multi-speaker English corpus
