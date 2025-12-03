# Segmentation & Alignment in Speech Recognition

## Table of Contents
1. [Introduction to Segmentation](#1-introduction-to-segmentation)
2. [Voice Activity Detection (VAD)](#2-voice-activity-detection-vad)
3. [Alignment Algorithms](#3-alignment-algorithms)
4. [Connectionist Temporal Classification (CTC)](#4-connectionist-temporal-classification-ctc)
5. [Attention Mechanisms](#5-attention-mechanisms)
6. [Speaker Diarization](#6-speaker-diarization)
7. [Timestamp Precision](#7-timestamp-precision)
8. [Practical Implementation](#8-practical-implementation)

---

## 1. Introduction to Segmentation

### The Segmentation Problem
**Challenge**: Audio is continuous, but we need discrete text segments with timestamps for:
- **UI Display**: Show transcripts incrementally
- **Editing**: Allow users to jump to specific moments
- **Analysis**: Track speaker turns, topic changes
- **Alignment**: Sync subtitles, translations, or annotations

### Segment Data Structure
```python
@dataclass
class TranscriptSegment:
    text: str           # "Hello, how are you?"
    start_time: float   # 2.35 seconds from audio start
    end_time: float     # 4.12 seconds
    speaker: str        # "Speaker 1" (optional, from diarization)
    confidence: float   # 0.95 (model confidence score)
    words: List[Word]   # Word-level timestamps (optional)
```

### Segmentation Levels
| Level      | Granularity | Use Case                           |
| ---------- | ----------- | ---------------------------------- |
| Utterance  | ~5-20s      | Natural speech pauses, turn-taking |
| Sentence   | ~2-10s      | Linguistic boundaries, subtitles   |
| **Phrase** | **1-5s**    | **Default ASR output (Whisper)**   |
| Word       | ~0.1-0.5s   | Karaoke, pronunciation analysis    |
| Phoneme    | ~0.01-0.2s  | Linguistic research, TTS           |

---

## 2. Voice Activity Detection (VAD)

### Why VAD Matters
- **Accuracy**: Prevent transcribing silence, background noise, music
- **Efficiency**: Skip processing 40-60% of typical audio (pauses, silence)
- **Segmentation**: Natural breakpoints for segments

### Classical VAD: Energy-Based

**Algorithm**:
1. Compute short-term energy (frame ~10-30ms)
2. Compare to threshold
3. Apply smoothing (avoid flickering)

```python
import numpy as np

def energy_vad(audio, frame_size=512, threshold=0.01):
    """Simple energy-based VAD."""
    n_frames = len(audio) // frame_size
    vad_flags = []
    
    for i in range(n_frames):
        frame = audio[i*frame_size:(i+1)*frame_size]
        energy = np.sum(frame**2) / frame_size
        vad_flags.append(energy > threshold)
    
    return np.array(vad_flags)
```

**Limitations**: 
- Sensitive to noise (SNR-dependent threshold)
- Misses low-energy speech (whispers, consonants)

### Spectral VAD: Zero-Crossing Rate (ZCR)

**Intuition**: Speech has low ZCR (vowels), noise has high ZCR.

```python
def zcr_vad(audio, frame_size=512, threshold=0.3):
    """Zero-crossing rate VAD."""
    n_frames = len(audio) // frame_size
    vad_flags = []
    
    for i in range(n_frames):
        frame = audio[i*frame_size:(i+1)*frame_size]
        # Count sign changes
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size)
        vad_flags.append(zcr < threshold)  # Low ZCR = speech
    
    return np.array(vad_flags)
```

### WebRTC VAD
**Google's production VAD** (used in Chrome, WebRTC stack).

**Features**:
- Gaussian Mixture Models (GMM)
- Handles various SNR conditions
- Low latency (~10ms)
- Three aggressiveness modes (0-3)

```python
import webrtcvad

vad = webrtcvad.Vad(2)  # Aggressiveness: 0 (lenient) to 3 (aggressive)

# Process 10ms frames
frame_duration = 10  # ms
frame_size = int(16000 * frame_duration / 1000)  # 160 samples @ 16kHz

for i in range(0, len(audio), frame_size):
    frame = audio[i:i+frame_size]
    is_speech = vad.is_speech(frame.tobytes(), sample_rate=16000)
```

### Neural VAD: Silero VAD

**Architecture**: Recurrent Neural Network (LSTM/GRU)

**Advantages**:
- Robust to noise, music, reverberation
- Language-independent
- Confidence scores (not just binary)

```python
import torch

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
get_speech_timestamps = utils[0]

# Detect speech segments
speech_timestamps = get_speech_timestamps(
    audio, 
    model,
    sampling_rate=16000,
    threshold=0.5,              # Speech probability threshold
    min_speech_duration_ms=250, # Minimum speech chunk
    min_silence_duration_ms=100 # Merge close segments
)

# Output: [{'start': 1600, 'end': 48000}, ...]
```

---

## 3. Alignment Algorithms

### The Alignment Problem
**Input**: Audio + Transcript text
**Output**: Precise timestamps for each word

**Use Cases**:
- Subtitle generation
- Audio-text synchronization
- Training data creation (forced alignment)

### Dynamic Time Warping (DTW)

**Concept**: Find optimal non-linear mapping between two sequences.

**Application**: Align acoustic features (MFCC) to phoneme sequence.

**Algorithm** (simplified):
```python
def dtw(seq1, seq2):
    """Dynamic Time Warping distance."""
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[0, :] = np.inf
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # Insertion
                dtw_matrix[i, j-1],    # Deletion
                dtw_matrix[i-1, j-1]   # Match
            )
    
    return dtw_matrix[n, m]
```

### Hidden Markov Models (HMM) - Classical Forced Alignment

**Tools**: Kaldi, Montreal Forced Aligner (MFA)

**Pipeline**:
1. **Phoneme Dictionary**: Map words to phonemes ("hello" → /h ə l oʊ/)
2. **Acoustic Model**: HMM-GMM trained on labeled data
3. **Viterbi Decoding**: Find most likely phoneme sequence given audio
4. **Alignment**: Extract phoneme/word boundaries from path

**Example (MFA)**:
```bash
# 1. Prepare data
# audio/file1.wav, audio/file1.txt
# 2. Align
mfa align audio/ lexicon.dict acoustic_model.zip output/

# 3. Result: TextGrid files with word-level timestamps
```

### Neural Forced Alignment

**Modern Approach**: End-to-end neural models (wav2vec 2.0 + CTC).

**Advantages**:
- No phoneme dictionary required
- Language-agnostic
- Better handling of out-of-vocabulary words

**Example (torchaudio CTC Forced Alignment)**:
```python
import torchaudio

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()

# Get emission probabilities
emissions, _ = model(waveform)

# Align transcript
transcript = "HELLO WORLD"
tokens = [bundle.get_labels().index(c) for c in transcript]

# CTC forced alignment
from torchaudio.functional import forced_align
path = forced_align(emissions, tokens, blank=0)

# Convert to timestamps
```

---

## 4. Connectionist Temporal Classification (CTC)

### The Problem CTC Solves
**Input**: Variable-length audio (e.g., 1000 frames)
**Output**: Variable-length text (e.g., 10 characters)

**Challenge**: Without alignment, how do we train?

### CTC Solution: Introduce Blank Token

**Alphabet**: {a, b, c, ..., z, space, **blank (ε)**}

**Alignment**: Map each audio frame to a label or blank.

**Example**:
```
Audio frames:  [f1]  [f2]  [f3]  [f4]  [f5]  [f6]  [f7]
CTC path:       ε     h     h     e     ε     l     l
Collapsed:           "h"   "h"   "e"         "l"   "l"
Final text:          "hell"
```

**Collapse Rule**: Remove blanks, merge repeated characters.

### CTC Loss Function

**Idea**: Sum probabilities of all valid alignments.

```
L_CTC = -log P(y|x) = -log Σ_π P(π|x)
```

Where π is any alignment path that collapses to ground truth y.

**Computation**: Forward-backward algorithm (dynamic programming).

### CTC Decoding

**Greedy Decoding**: Pick most likely label per frame.
```python
# emissions shape: [time, vocab_size]
best_path = emissions.argmax(dim=-1)
transcript = collapse_repeats(remove_blanks(best_path))
```

**Beam Search**: Maintain top-k hypotheses.

**Prefix Beam Search**: Group hypotheses by prefix (more efficient).

### CTC Limitations
- **Conditional Independence**: Assumes each frame's label independent (given audio)
- **No Language Model**: Pure acoustic model
- **Solution**: Hybrid CTC/Attention models, LM fusion

---

## 5. Attention Mechanisms

### Encoder-Decoder with Attention

**Architecture**:
```
Audio → Encoder (CNN/RNN/Transformer) → Hidden states (h1, h2, ..., hT)
                                              ↓
                                          Attention
                                              ↓
Decoder (RNN/Transformer) → Output tokens (y1, y2, ...)
```

### How Attention Works

**At each decoder step t**:
1. Compute **attention scores**: How relevant is each encoder hidden state?
   ```
   e_ti = score(decoder_state_t, encoder_hidden_i)
   ```

2. Compute **attention weights** (softmax):
   ```
   α_ti = exp(e_ti) / Σ_j exp(e_tj)
   ```

3. Compute **context vector** (weighted sum):
   ```
   c_t = Σ_i α_ti · h_i
   ```

4. Generate output:
   ```
   y_t = f(c_t, decoder_state_t)
   ```

### Attention Variants

**Additive (Bahdanau)**:
```
score(s, h) = v^T · tanh(W_s · s + W_h · h)
```

**Multiplicative (Luong)**:
```
score(s, h) = s^T · W · h
```

**Scaled Dot-Product (Transformer)**:
```
score(Q, K) = (Q · K^T) / sqrt(d_k)
```

### Multi-Head Attention

**Idea**: Learn multiple attention patterns simultaneously.

```
head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)
MultiHead = Concat(head_1, ..., head_h) · W^O
```

**Benefits**:
- Capture different relationships (phonetic, syntactic, semantic)
- Used in Transformer-based ASR (Conformer, Whisper)

### Cross-Attention in Whisper

**Whisper Decoder** attends to **encoder output** (audio features).

**Alignment**: Attention weights at each decoding step show which audio frames contributed.

```python
# Extract attention weights from Whisper
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.wav", word_timestamps=True)

# word_timestamps use cross-attention weights
for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
```

---

## 6. Speaker Diarization

### The Diarization Problem
**"Who spoke when?"**

**Input**: Audio
**Output**: Time segments labeled by speaker ID

### Diarization Pipeline

**1. Speech Segmentation** (VAD)
Extract speech regions, discard silence.

**2. Speaker Embedding Extraction**
For each speech segment, extract fixed-dimensional vector (embedding) representing speaker characteristics.

**Models**:
- **i-vectors** (classical): GMM-based, ~400-dim
- **x-vectors** (DNN): Time-delay neural network, ~512-dim
- **d-vectors** (deep): LSTM-based, ~256-dim
- **ECAPA-TDNN** (modern): State-of-the-art, ~192-dim

**3. Clustering**
Group embeddings by speaker.

**Algorithms**:
- **Agglomerative Hierarchical Clustering**: Bottom-up, merge closest pairs
- **Spectral Clustering**: Graph-based, handles non-convex clusters
- **K-means**: Fast, requires known number of speakers

**4. Re-segmentation** (optional)
Refine boundaries using Viterbi alignment with speaker models.

### Practical: pyannote.audio

```python
from pyannote.audio import Pipeline

# Load pretrained pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Run diarization
diarization = pipeline("audio.wav")

# Output
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}")

# Example output:
# 0.0s - 3.5s: SPEAKER_00
# 3.5s - 7.2s: SPEAKER_01
# 7.2s - 12.1s: SPEAKER_00
```

### Integration with ASR

**Option 1: Pre-diarization**
1. Diarize audio → speaker segments
2. Transcribe each segment separately
3. Merge results

**Option 2: Post-diarization**
1. Transcribe entire audio → text segments
2. Diarize audio → speaker segments
3. Align text segments with speaker segments (overlap-based matching)

**Option 3: Joint Model** (research)
End-to-end model for simultaneous transcription + diarization.

---

## 7. Timestamp Precision

### Challenges

**Frame-Level Alignment**: Models operate on frames (~10-25ms).

**Word Boundaries**: Depends on:
- Phoneme duration
- Speech rate
- Co-articulation (sounds blend)

**Typical Precision**:
- Phrase-level: ±100-500ms
- Word-level: ±50-100ms
- Phoneme-level: ±10-50ms

### Improving Precision

**1. Higher Temporal Resolution**
- Smaller frame shifts (10ms vs 25ms)
- Finer-grained attention

**2. Boundary Refinement**
- Acoustic-based energy detection
- Pitch/formant tracking

**3. Cross-Attention Weights**
Whisper uses attention weights to assign word timestamps.

### Whisper Word-Level Timestamps

**Algorithm** (simplified):
1. Decode transcript token-by-token
2. For each token, examine cross-attention weights
3. Find peak attention on audio frames
4. Convert frame indices to time

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe(
    "audio.wav",
    word_timestamps=True,  # Enable word-level
    prepend_punctuations="\"'"¿([{-",
    append_punctuations="\"'.。,，!！?？:：")]}、"
)

for segment in result["segments"]:
    print(f"Segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
    for word_info in segment.get("words", []):
        print(f"  {word_info['word']}: {word_info['start']:.2f}s")
```

---

## 8. Practical Implementation

### Segment Creation from VAD

```python
def vad_to_segments(audio, sr, vad_fn, min_duration=0.3, max_duration=30):
    """Convert VAD output to speech segments."""
    frame_duration = 10  # ms
    frame_size = int(sr * frame_duration / 1000)
    
    vad_flags = vad_fn(audio, frame_size)
    segments = []
    start = None
    
    for i, is_speech in enumerate(vad_flags):
        time = i * frame_duration / 1000
        
        if is_speech and start is None:
            start = time  # Speech started
        elif not is_speech and start is not None:
            # Speech ended
            duration = time - start
            if min_duration <= duration <= max_duration:
                end_sample = int(time * sr)
                start_sample = int(start * sr)
                segments.append({
                    'start': start,
                    'end': time,
                    'audio': audio[start_sample:end_sample]
                })
            start = None
    
    return segments
```

### Merging Overlapping Segments

```python
def merge_segments(segments, gap_threshold=0.5):
    """Merge segments with small gaps."""
    if not segments:
        return []
    
    merged = [segments[0]]
    
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg['start'] - last['end']
        
        if gap < gap_threshold:
            # Merge
            merged[-1] = {
                'start': last['start'],
                'end': seg['end'],
                'text': last['text'] + ' ' + seg['text']
            }
        else:
            merged.append(seg)
    
    return merged
```

### Exporting Segments to SRT (Subtitles)

```python
def segments_to_srt(segments, output_path):
    """Export segments as SRT subtitle file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp_srt(seg['start'])
            end = format_timestamp_srt(seg['end'])
            text = seg['text'].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

def format_timestamp_srt(seconds):
    """Convert seconds to SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
```

---

## Key Takeaways

1. **VAD** is critical: Use Silero or WebRTC for production systems.
2. **Alignment** methods:
   - Classical: HMM + Viterbi (MFA)
   - Modern: Neural CTC forced alignment
   - Whisper: Cross-attention weights
3. **CTC** enables training without frame-level labels.
4. **Attention** provides soft, learnable alignment (more flexible than CTC).
5. **Diarization** requires embedding models + clustering (pyannote.audio).
6. **Timestamp precision**: Word-level ±50-100ms is state-of-the-art.
7. **Segment merging** improves readability (avoid fragmented output).

---

## Further Reading

### Papers
- "Connectionist Temporal Classification" (Graves et al., 2006)
- "Listen, Attend and Spell" (Chan et al., 2016)
- "Attention Is All You Need" (Vaswani et al., 2017)
- "wav2vec 2.0: Self-Supervised Learning of Speech Representations" (Baevski et al., 2020)
- "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)
- "ECAPA-TDNN for Speaker Verification" (Desplanques et al., 2020)

### Tools
- **pyannote.audio**: Speaker diarization
- **Montreal Forced Aligner (MFA)**: HMM-based alignment
- **Gentle**: Forced alignment tool
- **Silero VAD**: Neural VAD
- **WebRTC VAD**: Production VAD

### Datasets
- **AMI Corpus**: Meeting recordings with diarization
- **VoxCeleb**: Speaker recognition
- **LibriTTS**: Multi-speaker audiobooks with alignments
