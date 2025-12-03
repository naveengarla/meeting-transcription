# Neural Network Models & Optimization for ASR

## Table of Contents
1. [Neural Network Fundamentals](#1-neural-network-fundamentals)
2. [Encoder-Decoder Architecture](#2-encoder-decoder-architecture)
3. [Whisper Model Internals](#3-whisper-model-internals)
4. [Model Quantization](#4-model-quantization)
5. [Inference Optimization](#5-inference-optimization)
6. [Model Compression Techniques](#6-model-compression-techniques)
7. [Hardware Acceleration](#7-hardware-acceleration)
8. [Benchmarking & Profiling](#8-benchmarking--profiling)

---

## 1. Neural Network Fundamentals

### Basic Building Blocks

**Linear Layer** (Fully Connected):
```python
y = Wx + b

# PyTorch
linear = nn.Linear(in_features=512, out_features=256)
```

**Activation Functions**:
- **ReLU**: f(x) = max(0, x) - most common
- **GELU**: f(x) = x·Φ(x) - smoother, used in Transformers
- **Sigmoid**: f(x) = 1/(1+e^(-x)) - output gates
- **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) - recurrent networks

**Convolutional Layer** (extract local patterns):
```python
# 1D conv for audio
conv1d = nn.Conv1d(in_channels=80, out_channels=128, kernel_size=3)

# 2D conv for spectrograms
conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
```

**Normalization**:
- **BatchNorm**: Normalize over batch dimension
- **LayerNorm**: Normalize over feature dimension (Transformers)
- **GroupNorm**: Middle ground (stable for small batches)

```python
layer_norm = nn.LayerNorm(512)  # Used in Transformers
```

**Dropout**: Random neuron dropout (prevent overfitting)
```python
dropout = nn.Dropout(p=0.1)  # Drop 10% of neurons during training
```

### Recurrent Networks

**LSTM** (Long Short-Term Memory):
```python
lstm = nn.LSTM(
    input_size=512,
    hidden_size=1024,
    num_layers=4,
    bidirectional=True,
    dropout=0.1
)

output, (h_n, c_n) = lstm(x)
# output: [seq_len, batch, hidden*2]
```

**GRU** (Gated Recurrent Unit): Simpler than LSTM, faster.

**Problem**: Sequential processing (slow), vanishing gradients.

### Transformer Basics

**Self-Attention**:
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, seq_len, d_model]
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (for causality or padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
```

**Multi-Head Attention**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections + split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_k
        )
        
        # Final linear
        output = self.W_o(attn_output)
        return output
```

**Feed-Forward Network**:
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))
```

---

## 2. Encoder-Decoder Architecture

### General Structure

```
Input → Encoder → Latent Representation → Decoder → Output
```

**Encoder**: Compress input into fixed/variable-length representation.

**Decoder**: Generate output sequence from representation.

### Speech Recognition Encoder-Decoder

```
Audio → Mel Spectrogram → Encoder → Hidden States
                                        ↓
                                  Cross-Attention
                                        ↓
Special Tokens + Previous Text → Decoder → Next Token
```

### Whisper Encoder

**Input**: Log-mel spectrogram [80, 3000] (80 mel bins, 3000 frames for 30s)

**Architecture**:
```python
class WhisperEncoder(nn.Module):
    def __init__(self, n_mels=80, n_ctx=1500, n_state=512, n_head=8, n_layer=6):
        super().__init__()
        
        # Convolutional stem (downsample 2×)
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        
        # Positional encoding
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_state, n_head) for _ in range(n_layer)
        ])
        
        self.ln_post = nn.LayerNorm(n_state)
    
    def forward(self, x):
        # x: [batch, n_mels, time]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))  # [batch, n_state, time/2]
        
        x = x.permute(0, 2, 1)  # [batch, time, n_state]
        
        # Add positional encoding
        x = x + self.positional_embedding[:x.shape[1]]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_post(x)
        return x
```

**Key Points**:
- 2× temporal downsampling via strided conv
- Sinusoidal positional encoding (learned in practice)
- Standard Transformer blocks with LayerNorm

### Whisper Decoder

**Autoregressive** generation with **cross-attention** to encoder.

```python
class WhisperDecoder(nn.Module):
    def __init__(self, n_vocab=51865, n_ctx=448, n_state=512, n_head=8, n_layer=6):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        
        # Transformer blocks (with cross-attention)
        self.blocks = nn.ModuleList([
            DecoderBlock(n_state, n_head) for _ in range(n_layer)
        ])
        
        self.ln = nn.LayerNorm(n_state)
        
        # Output projection
        self.token_out = nn.Linear(n_state, n_vocab, bias=False)
    
    def forward(self, tokens, encoder_output):
        # tokens: [batch, seq_len]
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding[:tokens.shape[1]]
        
        # Causal mask (prevent attending to future)
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1]), diagonal=1).bool()
        
        for block in self.blocks:
            x = block(x, encoder_output, mask)
        
        x = self.ln(x)
        logits = self.token_out(x)  # [batch, seq_len, n_vocab]
        
        return logits
```

**DecoderBlock**:
```python
class DecoderBlock(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)          # Self-attention
        self.cross_attn = MultiHeadAttention(n_state, n_head)    # Cross-attention
        self.mlp = PositionwiseFeedForward(n_state, n_state * 4)
        
        self.ln1 = nn.LayerNorm(n_state)
        self.ln2 = nn.LayerNorm(n_state)
        self.ln3 = nn.LayerNorm(n_state)
    
    def forward(self, x, encoder_output, mask):
        # Self-attention (masked)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        
        # Cross-attention (attend to encoder)
        x = x + self.cross_attn(
            self.ln2(x),              # Query from decoder
            encoder_output,           # Key from encoder
            encoder_output,           # Value from encoder
            mask=None
        )
        
        # Feed-forward
        x = x + self.mlp(self.ln3(x))
        
        return x
```

---

## 3. Whisper Model Internals

### Model Configurations

| Model  | Encoder Layers | Decoder Layers | Width | Heads | Parameters |
| ------ | -------------- | -------------- | ----- | ----- | ---------- |
| tiny   | 4              | 4              | 384   | 6     | 39M        |
| base   | 6              | 6              | 512   | 8     | 74M        |
| small  | 12             | 12             | 768   | 12    | 244M       |
| medium | 24             | 24             | 1024  | 16    | 769M       |
| large  | 32             | 32             | 1280  | 20    | 1550M      |

**Width** = d_model (hidden dimension)
**Heads** = Number of attention heads per layer

### Parameter Breakdown (base model)

```
Total: 74M parameters

Encoder:
- Conv layers: 80*512 + 512*512 = 0.3M
- Positional embedding: 1500*512 = 0.77M
- 6 Transformer blocks:
  - Self-attention: 4*512*512 = 1M per layer → 6M total
  - FFN: 2*512*2048 = 2M per layer → 12M total
- Total encoder: ~20M

Decoder:
- Token embedding: 51865*512 = 26.6M
- Positional embedding: 448*512 = 0.23M
- 6 Transformer blocks:
  - Self-attention: 1M per layer → 6M
  - Cross-attention: 1M per layer → 6M
  - FFN: 2M per layer → 12M
- Output projection: 512*51865 = 26.6M (tied with embedding)
- Total decoder: ~54M

Total: 74M ✓
```

### Vocabulary & Tokenization

**Vocabulary Size**: 51,865 tokens

**Components**:
- 50,257 tokens (GPT-2 BPE vocabulary)
- 99 language tokens: `<|en|>`, `<|es|>`, etc.
- Special tokens: `<|startoftranscript|>`, `<|translate|>`, `<|transcribe|>`, `<|notimestamps|>`, `<|nospeech|>`
- Timestamp tokens: `<|0.00|>`, `<|0.02|>`, ..., `<|30.00|>` (1501 tokens, 20ms resolution)

**Tokenizer**: Byte-Pair Encoding (BPE)
- Subword tokenization
- Language-agnostic (character fallback)

```python
import tiktoken

# Whisper uses GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, how are you?"
tokens = tokenizer.encode(text)
# [15496, 11, 703, 389, 345, 30]

decoded = tokenizer.decode(tokens)
# "Hello, how are you?"
```

### Training Details

**Loss Function**: Cross-entropy on next token prediction.

**Optimization**:
- AdamW optimizer
- Learning rate: 5e-5 (with warmup + decay)
- Batch size: ~256 examples (30s each)
- Mixed precision (FP16)

**Multi-Task Training**:
```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> TRANSCRIPT <|endoftext|>  # English transcription
<|startoftranscript|><|es|><|translate|><|notimestamps|> TRANSLATION <|endoftext|>   # Spanish→English translation
<|startoftranscript|><|fr|><|transcribe|><|0.00|> Bonjour <|2.50|> comment <|4.80|> # French with timestamps
```

---

## 4. Model Quantization

### Precision Formats

| Format   | Bits  | Range           | Precision   | Size (1M params) |
| -------- | ----- | --------------- | ----------- | ---------------- |
| FP32     | 32    | ±3.4×10^38      | ~7 digits   | 4 MB             |
| FP16     | 16    | ±6.5×10^4       | ~3 digits   | 2 MB             |
| **INT8** | **8** | **-128 to 127** | **Integer** | **1 MB**         |
| INT4     | 4     | -8 to 7         | Integer     | 0.5 MB           |

**Quantization**: Map floating point to lower precision (int8/int4).

### Why Quantization?

**Benefits**:
- **4× smaller** model size (FP32 → INT8)
- **2-4× faster** inference (SIMD integer ops)
- **Lower memory bandwidth** (critical for mobile/edge)

**Drawbacks**:
- Slight accuracy loss (~1-3% relative WER increase)
- Requires calibration data

### Post-Training Quantization (PTQ)

**Static Quantization**:
1. Collect activation statistics (min/max) on calibration data
2. Compute scale/zero-point for each layer
3. Quantize weights and activations

```python
# PyTorch static quantization
import torch.quantization

model_fp32 = WhisperModel()
model_fp32.eval()

# Fuse layers (conv+bn+relu)
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'bn', 'relu']])

# Prepare for quantization
model_fp32_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# Calibrate (forward pass on representative data)
for data in calibration_loader:
    model_fp32_prepared(data)

# Convert to INT8
model_int8 = torch.quantization.convert(model_fp32_prepared)

# Size comparison
def get_model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size_mb = os.path.getsize("temp.pth") / 1e6
    os.remove("temp.pth")
    return size_mb

print(f"FP32: {get_model_size(model_fp32):.1f} MB")
print(f"INT8: {get_model_size(model_int8):.1f} MB")  # ~4× smaller
```

**Dynamic Quantization** (weights-only):
```python
# Simpler: Quantize weights, keep activations in FP32
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear, nn.LSTM},  # Quantize these layers
    dtype=torch.qint8
)
```

### Quantization-Aware Training (QAT)

**Simulate quantization during training** → Better accuracy.

```python
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32)

# Train with fake quantization
for epoch in range(num_epochs):
    train(model_fp32_prepared)

# Convert to actual INT8
model_int8 = torch.quantization.convert(model_fp32_prepared)
```

### CTranslate2 INT8 Quantization

**faster-whisper** uses **CTranslate2** for optimized inference.

```python
from faster_whisper import WhisperModel

# Load quantized model
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"  # Options: float32, float16, int8, int8_float16
)

# Quantization happens transparently
segments, info = model.transcribe("audio.wav")
```

**Performance** (base model on CPU):
- **FP32**: ~1-2× real-time
- **INT8**: ~6-7× real-time (4-6× speedup!)

**Accuracy**: <1% relative WER degradation.

---

## 5. Inference Optimization

### CTranslate2

**Optimizations**:
- Layer fusion (conv+activation, linear+bias)
- GEMM (matrix multiplication) optimization (MKL, oneDNN)
- Memory-efficient attention (FlashAttention-style)
- Dynamic batching
- Quantization (INT8, INT16, FP16)

**Usage**:
```python
import ctranslate2

# Convert Whisper to CTranslate2 format
ct2-transformers-converter --model openai/whisper-base --output_dir whisper-base-ct2

# Load
model = ctranslate2.models.Whisper("whisper-base-ct2", device="cpu", compute_type="int8")

# Inference
features = whisper.log_mel_spectrogram(audio)
results = model.generate(features, beam_size=5)
```

### ONNX Runtime

**ONNX**: Open Neural Network Exchange (cross-framework format).

**Benefits**:
- Framework-agnostic (PyTorch → ONNX → TensorFlow/TFLite)
- Optimized kernels (ONNX Runtime)
- Hardware acceleration (DirectML, TensorRT)

**Export Whisper to ONNX**:
```python
import torch
import whisper

model = whisper.load_model("base")
model.eval()

# Dummy inputs
mel = torch.randn(1, 80, 3000)
tokens = torch.tensor([[50258]])  # <|startoftranscript|>

# Export encoder
torch.onnx.export(
    model.encoder,
    mel,
    "whisper_encoder.onnx",
    input_names=["mel"],
    output_names=["encoder_output"],
    dynamic_axes={"mel": {2: "time"}}
)

# Run with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("whisper_encoder.onnx")
encoder_output = session.run(None, {"mel": mel.numpy()})
```

### TensorRT (NVIDIA GPUs)

**NVIDIA's inference optimizer** for GPUs.

**Optimizations**:
- Kernel auto-tuning
- Layer fusion
- Precision calibration (INT8, FP16)
- TensorCore acceleration (mixed precision)

**Expected Speedup**: 2-5× over PyTorch on GPU.

### OpenVINO (Intel CPUs/iGPUs)

**Intel's toolkit** for edge deployment.

**Supports**: x86 CPUs, Intel GPUs, Movidius VPUs.

```bash
# Convert to OpenVINO IR
mo --input_model whisper_encoder.onnx --output_dir openvino/

# Run inference
from openvino.inference_engine import IECore

ie = IECore()
net = ie.read_network("openvino/whisper_encoder.xml")
exec_net = ie.load_network(net, "CPU")

output = exec_net.infer(inputs={"mel": mel_numpy})
```

---

## 6. Model Compression Techniques

### Pruning

**Remove unimportant weights** (set to zero).

**Methods**:
- **Magnitude Pruning**: Remove smallest weights
- **Structured Pruning**: Remove entire channels/neurons
- **Lottery Ticket Hypothesis**: Find sparse subnetworks that train well

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights in linear layer
prune.l1_unstructured(model.encoder.blocks[0].attn.W_q, name="weight", amount=0.3)

# Make pruning permanent
prune.remove(model.encoder.blocks[0].attn.W_q, 'weight')
```

**Typical Result**: 50-70% sparsity with <2% accuracy loss.

### Knowledge Distillation

**Train small model to mimic large model**.

**Teacher**: Large pretrained Whisper (e.g., large-v2)
**Student**: Smaller Whisper (e.g., tiny)

**Loss**:
```
L = α · CE(student_logits, ground_truth) + (1-α) · KL(student_logits, teacher_logits)
```

**Example**:
```python
teacher = WhisperModel("large")
student = WhisperModel("tiny")

teacher.eval()

for audio, text in dataloader:
    # Teacher outputs (frozen)
    with torch.no_grad():
        teacher_logits = teacher.decoder(text, teacher.encoder(audio))
    
    # Student outputs
    student_logits = student.decoder(text, student.encoder(audio))
    
    # Distillation loss
    ce_loss = F.cross_entropy(student_logits, text)
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * T**2
    
    loss = 0.5 * ce_loss + 0.5 * kl_loss
    loss.backward()
```

**Result**: Student learns faster, achieves better accuracy than training from scratch.

### Low-Rank Factorization

**Approximate weight matrix** W ∈ R^(m×n) as **W ≈ UV^T** where U ∈ R^(m×r), V ∈ R^(n×r), r ≪ min(m,n).

**Benefits**: Reduce parameters from m×n to (m+n)×r.

**Example** (reduce 512×512 to rank-64):
```python
W = model.layer.weight  # [512, 512]

# SVD
U, S, V = torch.svd(W)

# Keep top-64 singular values
r = 64
U_r = U[:, :r]  # [512, 64]
S_r = S[:r]
V_r = V[:, :r]  # [512, 64]

# Factorized weights
W_factorized = U_r @ torch.diag(S_r) @ V_r.T

# Replace layer with two smaller layers
model.layer_U = nn.Linear(512, 64, bias=False)
model.layer_V = nn.Linear(64, 512, bias=False)
model.layer_U.weight.data = (V_r * S_r).T
model.layer_V.weight.data = U_r.T
```

**Compression**: 512×512 = 262k params → 2×(512×64) = 65k params (4× reduction).

---

## 7. Hardware Acceleration

### CPU Optimization

**SIMD Instructions**:
- **AVX-2** (256-bit): 8× FP32 or 16× INT16 ops per cycle
- **AVX-512** (512-bit): 16× FP32 or 32× INT16 ops per cycle

**Libraries**:
- **Intel MKL** (Math Kernel Library): Optimized BLAS/LAPACK
- **oneDNN** (Deep Neural Network): Optimized primitives (conv, matmul)

**Threading**:
```python
import os
os.environ["OMP_NUM_THREADS"] = "8"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "8"  # MKL threads

import torch
torch.set_num_threads(8)
```

### GPU Acceleration

**CUDA Cores**: Thousands of small cores for parallel ops.

**Tensor Cores** (NVIDIA Ampere/Hopper): Specialized for matrix multiply (FP16/INT8).

**Mixed Precision Training** (FP16 + FP32):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    # FP16 forward/backward
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Scale loss, backward, unscale gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Result**: 2-3× speedup, 50% memory reduction.

### Edge Deployment (Mobile/Raspberry Pi)

**TensorFlow Lite**:
```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # PTQ
converter.target_spec.supported_types = [tf.float16]  # FP16
tflite_model = converter.convert()

# Save
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Inference (Android/iOS/RPi)
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_index)
```

**CoreML** (Apple Silicon):
```python
import coremltools as ct

model = ct.convert(
    torch_model,
    inputs=[ct.TensorType(shape=(1, 80, 3000))],
    compute_precision=ct.precision.FLOAT16
)

model.save("Whisper.mlmodel")
```

**ONNX → NCNN** (mobile CPUs):
- Optimized for ARM NEON (SIMD)
- Used in WeChat, TikTok

---

## 8. Benchmarking & Profiling

### Speed Benchmark

```python
import time
import numpy as np

def benchmark(model, audio, n_runs=10):
    """Benchmark transcription speed."""
    # Warmup
    for _ in range(3):
        model.transcribe(audio)
    
    # Measure
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = model.transcribe(audio)
        elapsed = time.time() - start
        times.append(elapsed)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    # Get audio duration
    import librosa
    duration = librosa.get_duration(filename=audio)
    
    rtf = mean_time / duration
    
    print(f"Audio Duration: {duration:.2f}s")
    print(f"Avg Processing Time: {mean_time:.2f}s ± {std_time:.2f}s")
    print(f"Real-Time Factor: {rtf:.2f}× ({1/rtf:.1f}× faster than real-time)")
```

### Memory Profiling

```python
import psutil
import os

def profile_memory(func):
    """Profile peak memory usage."""
    process = psutil.Process(os.getpid())
    
    mem_before = process.memory_info().rss / 1024**2  # MB
    
    result = func()
    
    mem_after = process.memory_info().rss / 1024**2
    mem_peak = mem_after - mem_before
    
    print(f"Memory Before: {mem_before:.1f} MB")
    print(f"Memory After: {mem_after:.1f} MB")
    print(f"Memory Peak: {mem_peak:.1f} MB")
    
    return result

# Usage
result = profile_memory(lambda: model.transcribe("audio.wav"))
```

### PyTorch Profiler

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model.transcribe("audio.wav")

# Print results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Export Chrome trace
prof.export_chrome_trace("trace.json")
```

### Accuracy Benchmark

```python
import jiwer

def evaluate_wer(model, test_set):
    """Compute WER on test set."""
    references = []
    hypotheses = []
    
    for audio_path, reference_text in test_set:
        result = model.transcribe(audio_path)
        hypothesis_text = result["text"]
        
        references.append(reference_text)
        hypotheses.append(hypothesis_text)
    
    wer = jiwer.wer(references, hypotheses)
    
    print(f"Word Error Rate: {wer:.2%}")
    return wer
```

---

## Key Takeaways

1. **Whisper** uses **encoder-decoder Transformer** with 74M-1550M parameters
2. **INT8 quantization** provides **4× size reduction** and **2-4× speedup** with minimal accuracy loss
3. **CTranslate2** is the fastest Whisper backend (6-7× real-time on CPU with int8)
4. **Optimization stack**: Quantization → Inference framework (CT2/ONNX) → Hardware accel (AVX-512/CUDA)
5. **Model compression**: Pruning (50-70% sparsity), Distillation (small model learns from large), Low-rank factorization
6. **Benchmarking**: Track RTF (real-time factor), WER, memory, latency
7. **Production tips**: Use int8, batch processing, VAD, language hints

---

## Further Reading

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Whisper: Robust Speech Recognition" (Radford et al., 2022)
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2018)
- "DistilBERT" (Sanh et al., 2019) - Knowledge distillation
- "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019) - Pruning

### Tools
- **CTranslate2**: Optimized Transformer inference
- **ONNX Runtime**: Cross-platform inference
- **TensorRT**: NVIDIA GPU optimization
- **OpenVINO**: Intel CPU/GPU optimization
- **PyTorch Profiler**: Performance analysis

### Frameworks
- **Hugging Face Optimum**: Model optimization (quantization, pruning, distillation)
- **Neural Compressor** (Intel): PTQ/QAT toolkit
- **TensorFlow Lite**: Mobile deployment
