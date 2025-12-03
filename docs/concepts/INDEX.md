# Technical Foundations for Speech Recognition Applications

**Purpose**: This folder provides in-depth technical tutorials on the fundamental concepts required to build production-grade speech recognition and transcription systems. These guides go beyond this specific project to teach the underlying theory, mathematics, and engineering patterns used across the industry.

**Target Audience**: Software engineers, ML practitioners, and researchers wanting to master the technical foundations of audio processing, automatic speech recognition (ASR), neural networks, and real-time systems.

---

## Learning Path

### 1. **Digital Speech Processing** (`speech.md`)
Master the fundamentals of digital audio:
- **Signal Theory**: Sampling theorem, Nyquist frequency, aliasing
- **Audio Encoding**: PCM, quantization, bit depth, dynamic range
- **Frequency Domain**: FFT, spectrograms, MFCCs, mel-scale
- **Audio Processing**: Filtering, normalization, VAD, noise reduction
- **Codecs & Formats**: WAV, FLAC, MP3, Opus - compression vs quality
- **Psychoacoustics**: Human hearing, masking, perceptual coding

### 2. **Segmentation & Alignment** (`segments.md`)
Learn how audio is chunked and aligned with text:
- **Voice Activity Detection (VAD)**: Energy-based, model-based (WebRTC, Silero)
- **Forced Alignment**: HMM-based, neural alignment (Montreal Forced Aligner)
- **Connectionist Temporal Classification (CTC)**: Loss function for sequence labeling
- **Attention Mechanisms**: Soft alignment in encoder-decoder models
- **Speaker Diarization**: Clustering, embeddings (x-vectors, d-vectors)
- **Timestamp Precision**: Frame-level vs word-level alignment

### 3. **Automatic Speech Recognition (ASR)** (`transcription.md`)
Understand the evolution and architecture of ASR systems:
- **Classical ASR**: HMM-GMM pipeline, Kaldi, pronunciation dictionaries
- **End-to-End Neural ASR**: Listen, Attend, and Spell (LAS), DeepSpeech
- **Transformer-based ASR**: Conformer, wav2vec 2.0, Whisper architecture
- **Preprocessing**: Feature extraction (log-mel spectrograms), data augmentation
- **Decoding Strategies**: Greedy, beam search, language model fusion
- **Evaluation Metrics**: WER, CER, BLEU, alignment accuracy

### 4. **Neural Network Models & Optimization** (`models.md`)
Deep dive into model architecture and deployment:
- **Encoder-Decoder Architecture**: Convolutional layers, self-attention, cross-attention
- **Whisper Internals**: Multi-task training, language detection, timestamp prediction
- **Model Quantization**: FP32 → FP16 → INT8 → INT4 (PTQ vs QAT)
- **Inference Optimization**: ONNX Runtime, TensorRT, CTranslate2, OpenVINO
- **Model Compression**: Pruning, distillation, knowledge transfer
- **Hardware Acceleration**: CPU (AVX2), GPU (CUDA), NPU, edge deployment
- **Benchmarking**: Latency, throughput, memory footprint, accuracy trade-offs

### 5. **Concurrency & Real-Time Systems** (`multithreading.md`)
Master concurrent programming for audio applications:
- **Python GIL**: Limitations, workarounds, when to use threads vs processes
- **Threading Patterns**: Producer-consumer, thread pools, locks, semaphores
- **Async Programming**: asyncio, coroutines, event loops, non-blocking I/O
- **Process-based Parallelism**: multiprocessing, shared memory, IPC
- **Audio Streaming**: Ring buffers, latency management, real-time constraints
- **GUI Threading**: Qt signals/slots, thread-safe UI updates, background workers
- **Profiling & Debugging**: cProfile, py-spy, threading debug tools, race conditions

---

## How to Use These Guides

**Sequential Learning**: Follow 1→5 if you're new to ASR systems.

**Reference Material**: Use as a technical reference when implementing specific features.

**Beyond This Project**: These concepts apply to any speech/audio ML system (voice assistants, podcast transcription, call center analytics, etc.).

**Further Reading**: Each guide includes references to academic papers, open-source projects, and production systems.

---

## Prerequisites

- **Mathematics**: Linear algebra, probability, basic calculus
- **Programming**: Python proficiency, understanding of OOP
- **Signal Processing**: Basic DSP (helpful but not required)
- **Machine Learning**: Familiarity with neural networks (CNNs, RNNs, Transformers)

---

## Practical Applications in This Project

| Concept                  | Implementation in `speech2text`          |
| ------------------------ | ---------------------------------------- |
| PCM Audio (16kHz, 16bit) | `audio_capture.py` - sounddevice config  |
| Mel Spectrograms         | faster-whisper preprocessing             |
| VAD Filtering            | `WhisperEngine` - `vad_filter=True`      |
| Encoder-Decoder          | Whisper model architecture               |
| INT8 Quantization        | CTranslate2 `compute_type="int8"`        |
| Threading                | `RecordingThread`, `TranscriptionThread` |
| Real-time Streaming      | Chrome Web Speech API (WebSocket)        |
| Forced Alignment         | Whisper word-level timestamps (future)   |
| Speaker Diarization      | Planned (pyannote.audio integration)     |

---

## Additional Resources

- **Books**: "Speech and Language Processing" (Jurafsky & Martin), "Deep Learning" (Goodfellow et al.)
- **Courses**: Stanford CS224S, CMU 11-785 (Deep Learning for Speech)
- **Papers**: "Attention Is All You Need", "Whisper: Robust Speech Recognition", "wav2vec 2.0"
- **Tools**: Kaldi, ESPnet, NeMo, Hugging Face Transformers, NVIDIA Riva
