# Multithreading & Concurrency in Python

## Table of Contents
1. [Concurrency Fundamentals](#1-concurrency-fundamentals)
2. [Python Global Interpreter Lock (GIL)](#2-python-global-interpreter-lock-gil)
3. [Threading in Python](#3-threading-in-python)
4. [Multiprocessing](#4-multiprocessing)
5. [Async/Await (Asyncio)](#5-asyncawait-asyncio)
6. [GUI Threading (PyQt/Tkinter)](#6-gui-threading-pyqttkinter)
7. [Real-Time Audio Streaming](#7-real-time-audio-streaming)
8. [Profiling & Debugging](#8-profiling--debugging)

---

## 1. Concurrency Fundamentals

### Concurrency vs Parallelism

**Concurrency**: Multiple tasks **interleaved** (share CPU time).
- Example: Single-core CPU switching between threads

**Parallelism**: Multiple tasks **simultaneously** (different CPUs).
- Example: Multi-core CPU running threads on separate cores

```
Concurrency:     |----Task A----|----Task B----|----Task A----|
                 Time →

Parallelism:     |------------Task A------------|
                 |------------Task B------------|
                 Time →
```

### Types of Concurrency

| Model               | Python Module     | Use Case                             | GIL Impact |
| ------------------- | ----------------- | ------------------------------------ | ---------- |
| **Threading**       | `threading`       | I/O-bound (network, disk, audio)     | Limited    |
| **Multiprocessing** | `multiprocessing` | CPU-bound (ML inference, processing) | Bypassed   |
| **Async/Await**     | `asyncio`         | I/O-bound, many connections          | N/A        |

### I/O-Bound vs CPU-Bound

**I/O-Bound**: Waiting for external resources (network, disk, audio device).
- **Solution**: Threading, asyncio (overlap waiting time)

**CPU-Bound**: Heavy computation (matrix multiply, FFT, model inference).
- **Solution**: Multiprocessing (utilize multiple cores)

**Example**:
- Audio capture: **I/O-bound** (waiting for sounddevice buffers)
- Whisper inference: **CPU-bound** (neural network computation)

---

## 2. Python Global Interpreter Lock (GIL)

### What is the GIL?

**GIL**: Mutex that allows only **one thread to execute Python bytecode** at a time.

**Implication**: Python threads **cannot** achieve true parallelism for CPU-bound tasks.

```python
import threading

counter = 0

def increment():
    global counter
    for _ in range(1000000):
        counter += 1

# Two threads
t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)

t1.start()
t2.start()
t1.join()
t2.join()

print(counter)  # NOT 2000000 due to race condition!
```

### Why GIL Exists

- **Memory management**: Simplifies reference counting (CPython's GC)
- **C extensions**: Easier to write thread-safe C extensions

### When GIL is Released

**Good news**: GIL is released during **I/O operations** and **C library calls**.

**Examples**:
- `time.sleep()` → GIL released
- `sounddevice` audio capture → GIL released (WASAPI runs in C)
- `faster-whisper` (CTranslate2) → GIL released during inference

**Practical Implication**: Threading works well for audio + ASR in our app!

### Bypassing the GIL

**Option 1**: Use **multiprocessing** (separate Python processes, separate GILs)

**Option 2**: Use libraries that release GIL:
- NumPy, SciPy (C/Fortran backends)
- CTranslate2, ONNX Runtime (C++ inference)

**Option 3**: Use alternative Python implementations:
- **Jython** (JVM-based, no GIL)
- **IronPython** (.NET-based, no GIL)
- **PyPy** (JIT compiler, still has GIL but faster)

---

## 3. Threading in Python

### Basic Threading

```python
import threading
import time

def worker(name, delay):
    """Simulate I/O-bound task."""
    print(f"{name} starting")
    time.sleep(delay)  # GIL released during sleep
    print(f"{name} finished")

# Create threads
t1 = threading.Thread(target=worker, args=("Thread-1", 2))
t2 = threading.Thread(target=worker, args=("Thread-2", 1))

# Start threads
t1.start()
t2.start()

# Wait for completion
t1.join()
t2.join()

print("All threads completed")
```

**Output**:
```
Thread-1 starting
Thread-2 starting
Thread-2 finished  # After 1s
Thread-1 finished  # After 2s
All threads completed
```

### Thread Synchronization

#### **Lock** (Mutex)
Prevent race conditions.

```python
import threading

counter = 0
lock = threading.Lock()

def increment_safe():
    global counter
    for _ in range(1000000):
        with lock:  # Acquire lock, auto-release on exit
            counter += 1

t1 = threading.Thread(target=increment_safe)
t2 = threading.Thread(target=increment_safe)

t1.start()
t2.start()
t1.join()
t2.join()

print(counter)  # 2000000 ✓
```

#### **RLock** (Reentrant Lock)
Same thread can acquire multiple times.

```python
rlock = threading.RLock()

def recursive_function(n):
    with rlock:
        if n > 0:
            print(n)
            recursive_function(n - 1)  # Can re-acquire lock
```

#### **Semaphore**
Limit concurrent access (e.g., max 3 threads).

```python
semaphore = threading.Semaphore(3)

def access_resource(name):
    with semaphore:
        print(f"{name} accessing resource")
        time.sleep(2)
        print(f"{name} releasing resource")

threads = [threading.Thread(target=access_resource, args=(f"T{i}",)) for i in range(10)]
for t in threads:
    t.start()
```

#### **Event**
Signal between threads.

```python
event = threading.Event()

def waiter():
    print("Waiting for event...")
    event.wait()  # Block until event.set()
    print("Event received!")

def setter():
    time.sleep(2)
    print("Setting event")
    event.set()

threading.Thread(target=waiter).start()
threading.Thread(target=setter).start()
```

### Producer-Consumer Pattern

```python
import threading
import queue
import time

# Thread-safe queue
q = queue.Queue(maxsize=10)

def producer():
    for i in range(20):
        item = f"item-{i}"
        q.put(item)  # Blocks if queue is full
        print(f"Produced {item}")
        time.sleep(0.1)
    
    # Signal end
    q.put(None)

def consumer():
    while True:
        item = q.get()  # Blocks if queue is empty
        if item is None:
            break
        print(f"  Consumed {item}")
        time.sleep(0.2)
        q.task_done()

threading.Thread(target=producer).start()
threading.Thread(target=consumer).start()
```

### Thread Pool

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * n

# Pool of 4 threads
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks
    futures = [executor.submit(task, i) for i in range(10)]
    
    # Collect results
    results = [f.result() for f in futures]
    print(results)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

**Alternative**:
```python
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(task, range(10)))
```

---

## 4. Multiprocessing

### Why Multiprocessing?

**Bypass GIL**: Each process has its own Python interpreter and GIL.

**Use Cases**:
- CPU-bound tasks (image processing, ML training)
- Parallel batch inference (transcribe 100 files)

### Basic Multiprocessing

```python
from multiprocessing import Process
import os

def worker(name):
    print(f"{name} running in process {os.getpid()}")

if __name__ == "__main__":  # Required on Windows
    processes = []
    
    for i in range(4):
        p = Process(target=worker, args=(f"Process-{i}",))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
```

### Process Pool

```python
from multiprocessing import Pool
import time

def cpu_bound_task(n):
    # Simulate heavy computation
    total = sum(i*i for i in range(n))
    return total

if __name__ == "__main__":
    # Pool of 4 processes
    with Pool(processes=4) as pool:
        results = pool.map(cpu_bound_task, [10**6, 10**6, 10**6, 10**6])
        print(results)
```

**Speedup**: ~4× on 4-core CPU (true parallelism).

### Inter-Process Communication (IPC)

#### **Queue**
```python
from multiprocessing import Process, Queue

def producer(q):
    for i in range(5):
        q.put(f"item-{i}")

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Got {item}")

if __name__ == "__main__":
    q = Queue()
    
    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))
    
    p1.start()
    p2.start()
    
    p1.join()
    q.put(None)  # Signal end
    p2.join()
```

#### **Pipe**
```python
from multiprocessing import Process, Pipe

def sender(conn):
    conn.send("Hello from sender")
    conn.close()

def receiver(conn):
    msg = conn.recv()
    print(msg)

if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    
    p1 = Process(target=sender, args=(child_conn,))
    p2 = Process(target=receiver, args=(parent_conn,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
```

#### **Shared Memory**
```python
from multiprocessing import Process, Value, Array

def increment(shared_val, shared_arr):
    shared_val.value += 1
    for i in range(len(shared_arr)):
        shared_arr[i] *= 2

if __name__ == "__main__":
    # Shared integer
    val = Value('i', 0)
    
    # Shared array
    arr = Array('i', [1, 2, 3, 4, 5])
    
    processes = [Process(target=increment, args=(val, arr)) for _ in range(4)]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(val.value)  # 4
    print(list(arr))  # [16, 32, 48, 64, 80]
```

### Multiprocessing for Batch Transcription

```python
from multiprocessing import Pool
from faster_whisper import WhisperModel

def transcribe_file(audio_path):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path)
    text = " ".join([seg.text for seg in segments])
    return audio_path, text

if __name__ == "__main__":
    audio_files = ["file1.wav", "file2.wav", "file3.wav", "file4.wav"]
    
    # 4 processes, each loads own model copy
    with Pool(processes=4) as pool:
        results = pool.map(transcribe_file, audio_files)
    
    for path, text in results:
        print(f"{path}: {text}")
```

**Note**: Each process loads its own model (~300MB RAM × 4 = 1.2GB total).

---

## 5. Async/Await (Asyncio)

### Asyncio Basics

**Cooperative multitasking**: Tasks voluntarily yield control (no preemption).

**Use Case**: Thousands of I/O-bound tasks (network requests, database queries).

```python
import asyncio

async def fetch_data(name, delay):
    print(f"{name} fetching...")
    await asyncio.sleep(delay)  # Yield control to event loop
    print(f"{name} done")
    return f"Data from {name}"

async def main():
    # Run concurrently
    results = await asyncio.gather(
        fetch_data("API-1", 2),
        fetch_data("API-2", 1),
        fetch_data("API-3", 3)
    )
    print(results)

# Run event loop
asyncio.run(main())
```

**Output**:
```
API-1 fetching...
API-2 fetching...
API-3 fetching...
API-2 done       # After 1s
API-1 done       # After 2s
API-3 done       # After 3s
['Data from API-1', 'Data from API-2', 'Data from API-3']
```

### Async File I/O

```python
import aiofiles
import asyncio

async def read_file(path):
    async with aiofiles.open(path, 'r') as f:
        content = await f.read()
        return content

async def main():
    tasks = [read_file(f"file{i}.txt") for i in range(10)]
    results = await asyncio.gather(*tasks)
```

### Async HTTP Requests

```python
import aiohttp
import asyncio

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [f"https://api.example.com/{i}" for i in range(100)]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    print(f"Fetched {len(results)} URLs")

asyncio.run(main())
```

**Performance**: 100 requests in ~1s (vs 100s sequentially).

### Async WebSocket (Chrome Speech API)

```python
import asyncio
import websockets

async def handle_client(websocket, path):
    async for message in websocket:
        print(f"Received: {message}")
        await websocket.send(f"Echo: {message}")

async def main():
    server = await websockets.serve(handle_client, "localhost", 8765)
    await server.wait_closed()

asyncio.run(main())
```

### Mixing Asyncio with Threading

```python
import asyncio
import concurrent.futures

def cpu_bound_task(n):
    return sum(i*i for i in range(n))

async def main():
    loop = asyncio.get_event_loop()
    
    # Run CPU-bound task in thread pool
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_bound_task, 10**7)
    
    print(result)

asyncio.run(main())
```

---

## 6. GUI Threading (PyQt/Tkinter)

### The Golden Rule

**NEVER block the GUI thread** → UI freezes.

**Solution**: Run long tasks in background threads, update GUI via signals/callbacks.

### PyQt Threading (Our Project)

#### **QThread** (Object-Oriented)

```python
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit
import time

class WorkerThread(QThread):
    # Signals (thread-safe communication)
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    
    def run(self):
        """This runs in background thread."""
        for i in range(10):
            time.sleep(1)
            self.progress.emit(i * 10)  # Update progress
        
        self.finished.emit("Task complete!")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.button = QPushButton("Start Task")
        self.button.clicked.connect(self.start_task)
        
        self.text_edit = QTextEdit()
        
        # Layout...
        self.setCentralWidget(self.text_edit)
    
    def start_task(self):
        self.worker = WorkerThread()
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
    
    def update_progress(self, value):
        """Called in GUI thread (via signal)."""
        self.text_edit.append(f"Progress: {value}%")
    
    def on_finished(self, message):
        self.text_edit.append(message)
```

**Key Points**:
- `run()` executes in background thread
- **Signals** are thread-safe (Qt handles synchronization)
- **Slots** (connected functions) run in GUI thread

#### **Real Example: Recording Thread**

```python
from PyQt6.QtCore import QThread, pyqtSignal
from audio_capture import AudioCapture

class RecordingThread(QThread):
    status_update = pyqtSignal(str)
    recording_complete = pyqtSignal(object)  # Audio data
    
    def __init__(self, mic_device, speaker_device):
        super().__init__()
        self.mic_device = mic_device
        self.speaker_device = speaker_device
        self.audio_capture = AudioCapture()
    
    def run(self):
        try:
            self.status_update.emit("Recording started...")
            
            # Start recording (I/O-bound, GIL released)
            self.audio_capture.start_recording(
                self.mic_device,
                self.speaker_device
            )
            
            # Wait for stop (in main thread via stop_recording())
            
        except Exception as e:
            self.status_update.emit(f"Error: {e}")
    
    def stop_recording(self):
        """Called from main thread."""
        audio_data = self.audio_capture.stop_recording()
        self.recording_complete.emit(audio_data)

# In MainWindow
self.recording_thread = RecordingThread(mic_idx, speaker_idx)
self.recording_thread.status_update.connect(self.update_status)
self.recording_thread.recording_complete.connect(self.on_recording_done)
self.recording_thread.start()
```

#### **Transcription Thread**

```python
class TranscriptionThread(QThread):
    transcription_complete = pyqtSignal(list)  # List[TranscriptSegment]
    error = pyqtSignal(str)
    
    def __init__(self, audio_data, sample_rate, engine):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.engine = engine
    
    def run(self):
        try:
            # CPU-bound (but GIL released in CTranslate2)
            segments = self.engine.transcribe(
                self.audio_data,
                self.sample_rate
            )
            self.transcription_complete.emit(segments)
        
        except Exception as e:
            self.error.emit(str(e))

# Usage
self.transcription_thread = TranscriptionThread(audio, sr, whisper_engine)
self.transcription_thread.transcription_complete.connect(self.display_transcript)
self.transcription_thread.error.connect(self.show_error)
self.transcription_thread.start()
```

### Tkinter Threading

```python
import tkinter as tk
import threading

def long_task():
    # This runs in background thread
    import time
    for i in range(10):
        time.sleep(1)
        # Update GUI (thread-safe via after())
        root.after(0, update_label, f"Progress: {i*10}%")

def update_label(text):
    label.config(text=text)

def start_task():
    threading.Thread(target=long_task, daemon=True).start()

root = tk.Tk()
label = tk.Label(root, text="Ready")
button = tk.Button(root, text="Start", command=start_task)

label.pack()
button.pack()

root.mainloop()
```

**Key**: Use `root.after(0, callback, *args)` to schedule GUI updates from threads.

---

## 7. Real-Time Audio Streaming

### Ring Buffer

**Problem**: Audio callback runs in separate thread, needs fast, lock-free buffer.

**Solution**: Circular buffer with atomic read/write pointers.

```python
import numpy as np
import threading

class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()
    
    def write(self, data):
        """Write data to buffer (called from audio callback)."""
        with self.lock:
            n = len(data)
            end_pos = (self.write_pos + n) % self.size
            
            if end_pos > self.write_pos:
                self.buffer[self.write_pos:end_pos] = data
            else:
                # Wrap around
                first_part = self.size - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:end_pos] = data[first_part:]
            
            self.write_pos = end_pos
    
    def read(self, n):
        """Read n samples (called from processing thread)."""
        with self.lock:
            end_pos = (self.read_pos + n) % self.size
            
            if end_pos > self.read_pos:
                data = self.buffer[self.read_pos:end_pos].copy()
            else:
                first_part = self.size - self.read_pos
                data = np.concatenate([
                    self.buffer[self.read_pos:],
                    self.buffer[:end_pos]
                ])
            
            self.read_pos = end_pos
            return data
    
    def available(self):
        """Number of samples available."""
        with self.lock:
            if self.write_pos >= self.read_pos:
                return self.write_pos - self.read_pos
            else:
                return self.size - self.read_pos + self.write_pos
```

### Real-Time Transcription Pipeline

```python
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue

class RealtimeTranscriber:
    def __init__(self):
        self.buffer = RingBuffer(16000 * 30)  # 30s buffer
        self.model = WhisperModel("base", compute_type="int8")
        self.chunk_queue = queue.Queue()
        self.running = False
    
    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice (separate thread)."""
        if status:
            print(status)
        
        # Write to ring buffer
        self.buffer.write(indata[:, 0])  # Mono
    
    def transcription_worker(self):
        """Background thread for transcription."""
        while self.running:
            # Check if enough data
            if self.buffer.available() >= 16000 * 5:  # 5s chunks
                # Read chunk
                chunk = self.buffer.read(16000 * 5)
                
                # Transcribe (CPU-bound, GIL released)
                segments, _ = self.model.transcribe(chunk)
                
                for seg in segments:
                    print(f"[{seg.start:.1f}s] {seg.text}")
    
    def start(self):
        self.running = True
        
        # Start transcription thread
        threading.Thread(target=self.transcription_worker, daemon=True).start()
        
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=16000,
            blocksize=4000  # ~250ms chunks
        )
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()

# Usage
transcriber = RealtimeTranscriber()
transcriber.start()

# Let it run...
import time
time.sleep(60)

transcriber.stop()
```

### Latency Optimization

**Target**: <500ms end-to-end latency.

**Strategies**:
- **Small chunks**: Process 2-3s segments (vs 30s)
- **Streaming VAD**: Silero VAD can process 250ms chunks
- **Incremental decoding**: Some models support prefix caching
- **GPU acceleration**: ~10× faster than CPU

---

## 8. Profiling & Debugging

### Thread Profiling

```python
import cProfile
import pstats
import threading

def task():
    total = sum(i*i for i in range(10**6))

# Profile threaded code
profiler = cProfile.Profile()
profiler.enable()

threads = [threading.Thread(target=task) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)
```

### py-spy (Sampling Profiler)

```bash
# Profile running Python process
py-spy top --pid 12345

# Generate flame graph
py-spy record -o profile.svg -- python my_script.py
```

### Threading Debug

```python
import threading

# List all threads
for t in threading.enumerate():
    print(f"Thread: {t.name}, Daemon: {t.daemon}, Alive: {t.is_alive()}")

# Main thread name
print(f"Main thread: {threading.main_thread().name}")

# Current thread
print(f"Current: {threading.current_thread().name}")
```

### Deadlock Detection

```python
import threading
import time

lock1 = threading.Lock()
lock2 = threading.Lock()

def task_a():
    with lock1:
        print("Task A acquired lock1")
        time.sleep(0.1)
        with lock2:  # Deadlock if task_b holds lock2
            print("Task A acquired lock2")

def task_b():
    with lock2:
        print("Task B acquired lock2")
        time.sleep(0.1)
        with lock1:  # Deadlock if task_a holds lock1
            print("Task B acquired lock1")

# This will deadlock!
threading.Thread(target=task_a).start()
threading.Thread(target=task_b).start()
```

**Solution**: Always acquire locks in same order.

```python
# Fixed version
def task_a():
    with lock1, lock2:  # Acquire in order
        print("Task A has both locks")

def task_b():
    with lock1, lock2:  # Same order
        print("Task B has both locks")
```

### Race Condition Detector

```python
import threading

# ThreadSanitizer (C/C++ tool, but concept applies)
# For Python: Use logging to detect unexpected interleavings

import logging

logging.basicConfig(level=logging.DEBUG, format='%(threadName)s: %(message)s')

counter = 0

def increment():
    global counter
    for _ in range(100000):
        logging.debug(f"Reading counter: {counter}")
        temp = counter
        temp += 1
        counter = temp
        logging.debug(f"Writing counter: {counter}")

# Run and inspect logs for unexpected orderings
```

---

## Key Takeaways

1. **GIL**: Python threads can't parallelize CPU-bound tasks, but work great for I/O-bound (audio, network)
2. **Threading**: Use for I/O-bound tasks, always synchronize shared state (locks, queues)
3. **Multiprocessing**: Use for CPU-bound tasks, bypasses GIL, separate memory space (IPC required)
4. **Asyncio**: Best for thousands of I/O-bound tasks (network servers, scrapers)
5. **GUI Threading**: NEVER block GUI thread, use QThread + signals for PyQt
6. **Audio Streaming**: Ring buffers + background threads, minimize latency (<500ms)
7. **Profiling**: Use cProfile, py-spy, thread enumeration to find bottlenecks
8. **Debugging**: Watch for deadlocks (lock order), race conditions (shared state), thread leaks

---

## Further Reading

### Books
- *Python Concurrency with asyncio* - Matthew Fowler
- *High Performance Python* - Micha Gorelick, Ian Ozsvald
- *Effective Python* - Brett Slatkin (Item 37-40 on concurrency)

### Documentation
- [Python threading docs](https://docs.python.org/3/library/threading.html)
- [Python multiprocessing docs](https://docs.python.org/3/library/multiprocessing.html)
- [Python asyncio docs](https://docs.python.org/3/library/asyncio.html)
- [PyQt6 QThread docs](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html)

### Tools
- **py-spy**: Sampling profiler for Python
- **ThreadSanitizer**: Detect data races (C/C++)
- **cProfile**: Built-in Python profiler
- **line_profiler**: Line-by-line profiling

### Papers
- "The Problem with Threads" - Edward A. Lee (2006)
- "An Analysis of Linux Scalability to Many Cores" - Boyd-Wickizer et al. (2010)
