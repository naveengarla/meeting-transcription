# Automatic Speech Recognition (ASR) - Theory & Practice

## Table of Contents
1. [Evolution of ASR](#1-evolution-of-asr)
2. [Classical ASR: HMM-GMM Pipeline](#2-classical-asr-hmm-gmm-pipeline)
3. [Deep Learning Era: End-to-End Neural ASR](#3-deep-learning-era-end-to-end-neural-asr)
4. [Transformer-Based ASR](#4-transformer-based-asr)
5. [Whisper Architecture Deep Dive](#5-whisper-architecture-deep-dive)
6. [Preprocessing & Feature Extraction](#6-preprocessing--feature-extraction)
7. [Decoding Strategies](#7-decoding-strategies)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Practical Implementation](#9-practical-implementation)

---

## 1. Evolution of ASR

### Timeline of ASR Milestones

**1952**: Bell Labs - "Audrey" system (digits 0-9)
**1970s**: DARPA SUR project, dynamic programming
**1980s**: **Hidden Markov Models (HMM)** become dominant
**1990s**: **Gaussian Mixture Models (GMM)** for acoustic modeling
**2000s**: Discriminative training, large vocabulary systems
**2009**: **Deep Neural Networks (DNN)** replace GMMs
**2014**: **Recurrent Neural Networks (RNN)** - sequence modeling
**2015**: **End-to-end models** (CTC, Attention) - no phoneme alignment needed
**2017**: **Transformers** - "Attention Is All You Need"
**2019**: **Self-supervised learning** - wav2vec, HuBERT
**2022**: **Whisper** - large-scale weakly supervised training

### Paradigm Shifts

| Era             | Approach             | Key Innovation                 |
| --------------- | -------------------- | ------------------------------ |
| Classical       | HMM-GMM + lexicon    | Statistical modeling           |
| Hybrid DNN      | DNN-HMM              | Neural acoustic models         |
| **End-to-End**  | **CTC, Attention**   | **No forced alignment needed** |
| Transformer     | Self-attention       | Parallelization, long context  |
| Self-supervised | wav2vec 2.0, Whisper | Massive unlabeled data         |

---

## 2. Classical ASR: HMM-GMM Pipeline

### Architecture

```
Audio → Feature Extraction (MFCC) → Acoustic Model (HMM-GMM)
                                          ↓
                                    Language Model (n-gram)
                                          ↓
                                    Pronunciation Lexicon
                                          ↓
                                      Decoder (Viterbi)
                                          ↓
                                       Transcript
```

### Components

**1. Feature Extraction**: Audio → MFCCs (13-40 coefficients)

**2. Acoustic Model (AM)**: P(features | phoneme sequence)
- **HMM**: Models phoneme duration and transitions
- **GMM**: Models feature distribution for each HMM state

**3. Pronunciation Lexicon**: "hello" → /h ə l oʊ/

**4. Language Model (LM)**: P(word sequence)
- **n-gram**: P(w_i | w_{i-1}, ..., w_{i-n+1})
- Trained on text corpora

**5. Decoder**: Find best word sequence using:
```
W* = argmax_W P(W|X) = argmax_W P(X|W) · P(W)
                           ↑           ↑
                       Acoustic    Language
                        Model       Model
```

### Kaldi: Open-Source Classical ASR

```bash
# Kaldi recipe structure
data/
  train/
    wav.scp     # Audio file list
    text        # Transcripts
    utt2spk     # Utterance to speaker mapping
  test/

# Training pipeline
steps/make_mfcc.sh          # Extract MFCCs
steps/train_mono.sh         # Monophone HMM
steps/align_si.sh           # Forced alignment
steps/train_deltas.sh       # Triphone HMM
steps/train_sat.sh          # Speaker adaptation (fMLLR)
steps/nnet3/train_tdnn.sh   # DNN-HMM hybrid

# Decoding
steps/decode.sh
```

**Pros**:
- Modular (easy to swap components)
- Interpretable (phoneme-level debugging)
- Data-efficient (can train on ~10h labeled speech)

**Cons**:
- Complex pipeline (many moving parts)
- Requires linguistic resources (lexicon, phonemes)
- Language-specific tuning

---

## 3. Deep Learning Era: End-to-End Neural ASR

### Listen, Attend, and Spell (LAS)

**Architecture** (2016, Google):
```
Audio → Listener (Encoder) → Listen to all
            ↓
        Attention
            ↓
        Speller (Decoder) → Output characters
```

**Listener (Encoder)**:
- Pyramidal BiLSTM (reduce temporal resolution)
- Outputs hidden states h_1, ..., h_T

**Attention**:
- Learns soft alignment between encoder outputs and decoder states

**Speller (Decoder)**:
- Character-level RNN
- Generates text autoregressively

**Training**: Cross-entropy loss on character sequences.

**Breakthrough**: No phoneme dictionary, no forced alignment, end-to-end gradient flow.

### DeepSpeech (Baidu, 2014)

**Architecture**:
```
Audio → Spectrogram → RNN layers → CTC loss
```

**Key Ideas**:
- **CTC loss**: Train without alignment
- **Data augmentation**: Noise addition, speed perturbation
- **Massive data**: 5000h English + synthetic data

**Code** (simplified):
```python
import torch
import torch.nn as nn

class DeepSpeech(nn.Module):
    def __init__(self, n_mels=80, n_class=29):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2))
        self.rnn = nn.LSTM(input_size=n_mels*16, hidden_size=512, 
                           num_layers=5, bidirectional=True)
        self.fc = nn.Linear(1024, n_class)  # 26 letters + space + blank + apostrophe
    
    def forward(self, x):
        # x: [batch, 1, time, n_mels]
        x = self.conv(x)  # [batch, 32, time', mels']
        x = x.permute(0, 2, 1, 3).flatten(2)  # [batch, time', features]
        x, _ = self.rnn(x)  # [batch, time', 1024]
        x = self.fc(x)  # [batch, time', n_class]
        return torch.nn.functional.log_softmax(x, dim=-1)

# Training
model = DeepSpeech()
ctc_loss = nn.CTCLoss(blank=0)

# Forward
log_probs = model(spectrograms)  # [batch, time, classes]
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

### RNN-Transducer (RNN-T)

**Problem with CTC**: Conditional independence assumption (each frame independent).

**Solution**: RNN-T adds prediction network.

**Architecture**:
```
Audio → Encoder → Encoder states
                       ↓
                  Joint Network → Output distribution
                       ↓
         Prediction Network ← Previous outputs
```

**Advantage**: Models label dependencies (like language model).

**Widely used**: Google Voice Search, Siri.

---

## 4. Transformer-Based ASR

### Why Transformers for ASR?

**RNN Limitations**:
- Sequential processing (slow, can't parallelize)
- Vanishing gradients (long-range dependencies)

**Transformer Advantages**:
- Parallel processing (faster training)
- Self-attention (global context)
- Scalability (100M+ parameters)

### Conformer (2020, Google)

**Hybrid CNN-Transformer architecture** for speech.

```
Input → Convolution Subsampling (↓4x)
          ↓
    Conformer Blocks (×16)
    [
      Feed Forward (Macaron)
      ↓
      Multi-Head Self-Attention
      ↓
      Convolution Module (depthwise)
      ↓
      Feed Forward
      ↓
      Layer Norm
    ]
          ↓
        Linear → CTC/AED loss
```

**Key Innovation**: Combines:
- **Self-attention**: Global context
- **Convolution**: Local patterns (phonetic features)

**Performance**: State-of-the-art on LibriSpeech (1.9% WER).

### wav2vec 2.0 (2020, Meta)

**Self-Supervised Pretraining** on unlabeled audio.

**Idea**:
1. Mask random spans of input
2. Train model to predict masked parts
3. Fine-tune on labeled data (10x less than supervised)

**Architecture**:
```
Raw Audio → CNN Encoder → Transformer → Contextualized representations
                              ↓
                       Contrastive Loss (predict masked)
```

**Impact**: Achieves low WER with only 10 minutes of labeled data!

---

## 5. Whisper Architecture Deep Dive

### Overview

**Whisper** (2022, OpenAI) is a **Transformer encoder-decoder** trained on 680,000 hours of weakly labeled data from the web.

**Key Features**:
- Multi-task: Transcription, translation, language detection, voice activity detection
- Multilingual: 99 languages
- Robust: Handles accents, noise, technical terms
- Timestamps: Word-level alignment via cross-attention

### Architecture

```
Audio (30s max) → Log-Mel Spectrogram (80 bins)
                        ↓
                  Encoder (Transformer)
                  [24 layers, 512-dim, 8 heads] (base)
                        ↓
                  Encoder output (hidden states)
                        ↓
                  Decoder (Transformer)
                  [24 layers, cross-attention]
                        ↓
                  Token predictions (autoregressive)
```

### Encoder

**Input**: 80-channel log-mel spectrogram
- 25ms windows, 10ms stride
- 3000 frames for 30s audio

**Layers**: 
- Convolutional stem (2 layers, kernel=3)
- Sinusoidal positional encoding
- Transformer blocks (multi-head self-attention + FFN)

**Output**: Sequence of hidden states (contextual audio features)

### Decoder

**Input**: 
- Previous tokens (shifted right)
- Special tokens: `<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, `<|notimestamps|>`

**Cross-Attention**: Attends to encoder outputs (learns alignment).

**Output**: Next token probabilities (autoregressive).

### Special Tokens

```
<|startoftranscript|>     # Always first
<|en|>                    # Language ID (99 languages)
<|transcribe|>            # Task (or <|translate|> for X→EN)
<|notimestamps|>          # Disable timestamps (or <|0.00|> for word-level)
<|nospeech|>              # No speech detected (VAD)
```

**Example sequence**:
```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello, how are you? <|endoftext|>
```

### Multi-Task Training

Whisper is trained on multiple tasks simultaneously:

1. **Transcription**: Audio → Text (same language)
2. **Translation**: Audio → English text (X→EN)
3. **Language Detection**: Audio → Language ID
4. **Voice Activity Detection**: Audio → `<|nospeech|>` if silent

**Training Data**:
- 680,000 hours (287,000 hours after filtering)
- Weakly labeled (from video subtitles, not professional transcripts)
- 117 languages (99 in final model)

### Model Sizes

| Model  | Parameters | Layers | Width | Heads | VRAM  | Speed (CPU) |
| ------ | ---------- | ------ | ----- | ----- | ----- | ----------- |
| tiny   | 39M        | 4/4    | 384   | 6     | ~1GB  | ~10-12x RT  |
| base   | 74M        | 6/6    | 512   | 8     | ~1GB  | ~6-7x RT    |
| small  | 244M       | 12/12  | 768   | 12    | ~2GB  | ~4-5x RT    |
| medium | 769M       | 24/24  | 1024  | 16    | ~5GB  | ~2-3x RT    |
| large  | 1550M      | 32/32  | 1280  | 20    | ~10GB | ~1-2x RT    |

(Encoder layers / Decoder layers)

### Inference Pipeline

```python
import whisper

# 1. Load model (lazy, cached)
model = whisper.load_model("base")

# 2. Load & preprocess audio
audio = whisper.load_audio("file.wav")  # Resample to 16kHz, mono
audio = whisper.pad_or_trim(audio)      # Trim to 30s

# 3. Compute mel spectrogram
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# 4. Detect language (optional)
_, probs = model.detect_language(mel)
lang = max(probs, key=probs.get)

# 5. Decode
options = whisper.DecodingOptions(language=lang, without_timestamps=False)
result = whisper.decode(model, mel, options)

print(result.text)
```

### Word-Level Timestamps

**How it works**:
1. Cross-attention weights show which audio frames aligned to each token
2. Find peak attention for each word token
3. Convert frame indices to time

**Accuracy**: ±50-100ms (state-of-the-art).

**Code**:
```python
result = model.transcribe("audio.wav", word_timestamps=True)

for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
```

---

## 6. Preprocessing & Feature Extraction

### Audio Normalization

```python
import librosa
import numpy as np

# 1. Load audio
audio, sr = librosa.load("audio.wav", sr=16000, mono=True)

# 2. Peak normalization
audio = audio / np.max(np.abs(audio))

# 3. Optional: Pre-emphasis
audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
```

### Log-Mel Spectrogram

```python
import torch
import torchaudio

# Whisper's exact preprocessing
def log_mel_spectrogram(audio, n_mels=80):
    """
    Args:
        audio: float32 tensor, shape [samples], range [-1, 1]
        n_mels: number of mel bins (80 for Whisper)
    Returns:
        log_mel: shape [n_mels, time]
    """
    # STFT
    n_fft = 400      # 25ms window @ 16kHz
    hop_length = 160 # 10ms hop
    
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft),
        return_complex=True
    )
    
    # Magnitude
    magnitudes = stft.abs() ** 2
    
    # Mel filterbank
    mel_filters = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        n_mels=n_mels,
        f_min=0,
        f_max=8000,
        sample_rate=16000
    )
    
    mel_spec = mel_filters.T @ magnitudes
    
    # Log scale
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)  # Dynamic range
    log_spec = (log_spec + 4.0) / 4.0  # Normalize to ~[0, 1]
    
    return log_spec
```

### Data Augmentation

**Techniques** (training only):
- **SpecAugment**: Mask time/frequency bands in spectrogram
- **Speed Perturbation**: Resample to 90%, 100%, 110% speed
- **Noise Addition**: Add background noise (SNR 5-20 dB)
- **Reverberation**: Simulate room acoustics

```python
# SpecAugment
import torchaudio.transforms as T

spec_aug = T.SpecAugment(
    time_mask_param=70,
    freq_mask_param=15,
    n_time_masks=2,
    n_freq_masks=2
)

augmented_spec = spec_aug(mel_spectrogram)
```

---

## 7. Decoding Strategies

### Greedy Decoding

**Simple**: Pick most likely token at each step.

```python
def greedy_decode(model, mel):
    tokens = [sot_token]  # Start of transcript
    
    for i in range(max_length):
        logits = model.decoder(tokens, mel)
        next_token = logits[:, -1, :].argmax(dim=-1)
        tokens.append(next_token)
        
        if next_token == eot_token:  # End of transcript
            break
    
    return tokens
```

**Pros**: Fast
**Cons**: Suboptimal (myopic decisions)

### Beam Search

**Idea**: Maintain top-k hypotheses at each step.

```python
def beam_search(model, mel, beam_width=5):
    # Start with <|startoftranscript|>
    beams = [(0.0, [sot_token])]  # (log_prob, tokens)
    
    for step in range(max_length):
        new_beams = []
        
        for log_prob, tokens in beams:
            if tokens[-1] == eot_token:
                new_beams.append((log_prob, tokens))
                continue
            
            logits = model.decoder(tokens, mel)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            
            # Get top-k next tokens
            topk_probs, topk_ids = probs.topk(beam_width)
            
            for prob, token_id in zip(topk_probs, topk_ids):
                new_log_prob = log_prob + torch.log(prob)
                new_tokens = tokens + [token_id]
                new_beams.append((new_log_prob, new_tokens))
        
        # Keep top-k beams
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
    
    return beams[0][1]  # Return best hypothesis
```

**Pros**: Better quality than greedy
**Cons**: Slower (k× more compute)

### Temperature Sampling

**Control randomness**:
```python
logits = model.decoder(tokens, mel) / temperature
probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

- `temperature < 1`: More deterministic (focused)
- `temperature = 1`: Standard sampling
- `temperature > 1`: More random (creative)

### Language Model Fusion

**Shallow Fusion**:
```
score = λ · log P_AM(y|x) + (1-λ) · log P_LM(y)
```

**Deep Fusion**: Train joint model with LM features.

**External LM**: Use pretrained GPT/BERT for rescoring.

---

## 8. Evaluation Metrics

### Word Error Rate (WER)

**Definition**: Edit distance between hypothesis and reference.

```
WER = (S + D + I) / N

S = Substitutions (wrong word)
D = Deletions (missing word)
I = Insertions (extra word)
N = Total words in reference
```

**Example**:
```
Reference:  "the cat sat on the mat"
Hypothesis: "the cat sit on mat"

S=1 (sat→sit), D=1 (the), I=0
WER = (1+1+0) / 6 = 33.3%
```

**Code**:
```python
import jiwer

reference = "the cat sat on the mat"
hypothesis = "the cat sit on mat"

wer = jiwer.wer(reference, hypothesis)
print(f"WER: {wer:.1%}")  # 33.3%
```

### Character Error Rate (CER)

**Same as WER**, but at character level.

**Use**: Languages without clear word boundaries (Chinese, Japanese).

### BLEU Score

**Used for**: Translation tasks (Whisper X→EN).

**Idea**: N-gram overlap between hypothesis and reference.

### Real-Time Factor (RTF)

```
RTF = Processing Time / Audio Duration
```

**Examples**:
- RTF = 0.1 → 10× faster than real-time (can transcribe 1h in 6 min)
- RTF = 1.0 → Real-time processing
- RTF = 2.0 → 2× slower than real-time (bottleneck)

**Target**: RTF < 0.5 for streaming applications.

---

## 9. Practical Implementation

### faster-whisper (CTranslate2)

**Optimized Whisper inference** (6-7× faster than original).

```python
from faster_whisper import WhisperModel

# Load model (int8 quantization)
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    num_workers=8  # Parallel processing
)

# Transcribe
segments, info = model.transcribe(
    "audio.wav",
    language="en",
    vad_filter=True,       # Voice activity detection
    vad_parameters=dict(
        threshold=0.5,
        min_speech_duration_ms=250
    )
)

# Iterate segments
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

# Info
print(f"Detected language: {info.language} ({info.language_probability:.2f})")
```

### Performance Monitoring

```python
import psutil
import time

def transcribe_with_metrics(audio_path):
    # Start monitoring
    process = psutil.Process()
    cpu_start = process.cpu_percent()
    mem_start = process.memory_info().rss / 1024**2  # MB
    time_start = time.time()
    
    # Get audio duration
    import librosa
    duration = librosa.get_duration(path=audio_path)
    
    # Transcribe
    segments, info = model.transcribe(audio_path)
    segments = list(segments)  # Consume generator
    
    # End monitoring
    time_end = time.time()
    cpu_end = process.cpu_percent()
    mem_end = process.memory_info().rss / 1024**2
    
    # Metrics
    elapsed = time_end - time_start
    rtf = elapsed / duration
    speed_multiplier = duration / elapsed
    
    print(f"Audio: {duration:.1f}s")
    print(f"Processing: {elapsed:.1f}s")
    print(f"Speed: {speed_multiplier:.1f}× real-time")
    print(f"CPU: {(cpu_end-cpu_start)/psutil.cpu_count():.1f}%")
    print(f"Memory: {mem_end - mem_start:.1f} MB")
    
    return segments, info
```

---

## Key Takeaways

1. **Evolution**: HMM-GMM → DNN-HMM → End-to-End (CTC/Attention) → Transformers → Self-Supervised
2. **Whisper** is a **multi-task Transformer** trained on 680k hours weakly labeled data
3. **Encoder-Decoder** with **cross-attention** provides soft alignment (timestamps)
4. **Log-mel spectrograms** (80 bins) are standard input for neural ASR
5. **Beam search** improves quality over greedy decoding
6. **WER** is the primary metric; **RTF** measures speed
7. **faster-whisper** provides 6-7× speedup via CTranslate2 + int8 quantization
8. **VAD** + **language hint** significantly improve accuracy and speed

---

## Further Reading

### Papers
- "A Tutorial on Hidden Markov Models" (Rabiner, 1989)
- "Listen, Attend and Spell" (Chan et al., 2016)
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Conformer: Convolution-augmented Transformer" (Gulati et al., 2020)
- "wav2vec 2.0" (Baevski et al., 2020)
- "Whisper: Robust Speech Recognition" (Radford et al., 2022)

### Frameworks
- **Kaldi**: Classical ASR toolkit
- **ESPnet**: End-to-end toolkit (supports CTC, AED, RNN-T, Transducer)
- **NeMo** (NVIDIA): Production-grade ASR
- **Hugging Face Transformers**: Whisper, wav2vec 2.0, etc.
- **faster-whisper**: Optimized Whisper (CTranslate2)

### Datasets
- **LibriSpeech**: 1000h English audiobooks (clean)
- **Common Voice**: Multilingual crowdsourced
- **GigaSpeech**: 10,000h English (diverse domains)
- **Multilingual LibriSpeech**: 6 languages
