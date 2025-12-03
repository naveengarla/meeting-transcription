"""
Transcription Module
Supports multiple transcription engines:
1. Chrome Web Speech API (free, cloud-based, real-time)
2. OpenAI Whisper (local, offline, high-quality)
3. Azure Speech Service (cloud-based, enterprise)
"""
import numpy as np
import wave
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable
from abc import ABC, abstractmethod
import config


class TranscriptSegment:
    """Represents a segment of transcribed text with metadata"""
    def __init__(self, text: str, start_time: float, end_time: float, 
                 speaker: str = "Unknown", confidence: float = 0.0):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.speaker = speaker
        self.confidence = confidence
    
    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'speaker': self.speaker,
            'confidence': self.confidence
        }
    
    def __repr__(self):
        return f"[{self.start_time:.2f}s - {self.end_time:.2f}s] {self.speaker}: {self.text}"


class TranscriptionEngine(ABC):
    """Abstract base class for transcription engines"""
    
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> List[TranscriptSegment]:
        """Transcribe audio data and return segments"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is available and configured"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name"""
        pass


class ChromeSpeechEngine(TranscriptionEngine):
    """
    Chrome Web Speech API transcription engine.
    Uses a local web server to interface with Chrome's speech recognition.
    """
    
    def __init__(self):
        self.server_port = 8765
        self.is_running = False
        self._last_transcript = []
        
    @property
    def name(self) -> str:
        return "Chrome Web Speech API"
    
    def is_available(self) -> bool:
        """Check if Chrome/Edge browser is available"""
        try:
            import webbrowser
            # Check if we can launch a browser
            return True
        except:
            return False
    
    def start_server(self, callback: Callable[[str, float], None] = None):
        """
        Start a local WebSocket server for real-time transcription.
        callback: Function to call with (text, timestamp) for each recognized phrase
        """
        try:
            import asyncio
            import websockets
            from threading import Thread
            
            self._callback = callback
            
            async def handle_client(websocket, path):
                """Handle WebSocket messages from browser"""
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        if data.get('type') == 'transcript':
                            text = data.get('text', '')
                            timestamp = data.get('timestamp', 0)
                            confidence = data.get('confidence', 0)
                            is_final = data.get('is_final', False)
                            
                            if self._callback and text:
                                self._callback(text, timestamp, confidence, is_final)
                                
                except Exception as e:
                    print(f"WebSocket error: {e}")
            
            async def start_ws_server():
                async with websockets.serve(handle_client, "localhost", self.server_port):
                    await asyncio.Future()  # run forever
            
            def run_server():
                asyncio.run(start_ws_server())
            
            # Start server in background thread
            server_thread = Thread(target=run_server, daemon=True)
            server_thread.start()
            self.is_running = True
            print(f"✓ WebSocket server started on ws://localhost:{self.server_port}")
            
        except ImportError:
            print("⚠ websockets module not installed. Install with: pip install websockets")
            return False
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
    
    def get_html_page(self) -> str:
        """Generate HTML page with Web Speech API integration"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Speech Recognition</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .status {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            font-weight: bold;
        }}
        .status.listening {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .status.stopped {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .controls {{
            margin: 20px 0;
        }}
        button {{
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
        }}
        button.start {{
            background: #28a745;
            color: white;
        }}
        button.stop {{
            background: #dc3545;
            color: white;
        }}
        button:hover {{
            opacity: 0.8;
        }}
        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        #transcript {{
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            min-height: 200px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .interim {{
            color: #999;
            font-style: italic;
        }}
        .final {{
            color: #333;
            margin: 5px 0;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.85em;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Chrome Speech Recognition</h1>
        <div id="status" class="status stopped">Status: Not Connected</div>
        
        <div class="controls">
            <button id="startBtn" class="start" onclick="startRecognition()">Start Listening</button>
            <button id="stopBtn" class="stop" onclick="stopRecognition()" disabled>Stop</button>
        </div>
        
        <div id="transcript"></div>
    </div>

    <script>
        let recognition = null;
        let ws = null;
        let isRecognizing = false;
        let startTime = Date.now();

        // Initialize WebSocket connection
        function connectWebSocket() {{
            ws = new WebSocket('ws://localhost:{self.server_port}');
            
            ws.onopen = () => {{
                console.log('Connected to Python app');
                updateStatus('Connected - Ready to start', 'stopped');
            }};
            
            ws.onerror = (error) => {{
                console.error('WebSocket error:', error);
                updateStatus('Connection error - Check Python app', 'stopped');
            }};
            
            ws.onclose = () => {{
                console.log('Disconnected from Python app');
                updateStatus('Disconnected', 'stopped');
            }};
        }}

        // Initialize Speech Recognition
        function initRecognition() {{
            if (!('webkitSpeechRecognition' in window)) {{
                alert('Speech recognition not supported in this browser. Please use Chrome or Edge.');
                return false;
            }}

            recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = () => {{
                isRecognizing = true;
                startTime = Date.now();
                updateStatus('Listening...', 'listening');
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
            }};

            recognition.onend = () => {{
                isRecognizing = false;
                updateStatus('Stopped', 'stopped');
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            }};

            recognition.onerror = (event) => {{
                console.error('Recognition error:', event.error);
                updateStatus('Error: ' + event.error, 'stopped');
            }};

            recognition.onresult = (event) => {{
                let interimTranscript = '';
                let finalTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {{
                    const transcript = event.results[i][0].transcript;
                    const confidence = event.results[i][0].confidence;
                    
                    if (event.results[i].isFinal) {{
                        finalTranscript += transcript + ' ';
                        
                        // Send to Python app
                        if (ws && ws.readyState === WebSocket.OPEN) {{
                            const timestamp = (Date.now() - startTime) / 1000;
                            ws.send(JSON.stringify({{
                                type: 'transcript',
                                text: transcript,
                                timestamp: timestamp,
                                confidence: confidence || 0,
                                is_final: true
                            }}));
                        }}
                        
                        // Display final transcript
                        displayTranscript(transcript, confidence, true);
                    }} else {{
                        interimTranscript += transcript;
                    }}
                }}

                // Show interim results
                if (interimTranscript) {{
                    displayInterim(interimTranscript);
                }}
            }};

            return true;
        }}

        function startRecognition() {{
            if (!recognition && !initRecognition()) {{
                return;
            }}
            
            if (!isRecognizing) {{
                recognition.start();
            }}
        }}

        function stopRecognition() {{
            if (recognition && isRecognizing) {{
                recognition.stop();
            }}
        }}

        function updateStatus(message, state) {{
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = 'Status: ' + message;
            statusDiv.className = 'status ' + state;
        }}

        function displayTranscript(text, confidence, isFinal) {{
            const transcriptDiv = document.getElementById('transcript');
            const timestamp = new Date().toLocaleTimeString();
            
            if (isFinal) {{
                const p = document.createElement('p');
                p.className = 'final';
                p.innerHTML = `<span class="timestamp">${{timestamp}}</span>${{text}}`;
                transcriptDiv.appendChild(p);
                
                // Remove interim display
                const interim = transcriptDiv.querySelector('.interim');
                if (interim) interim.remove();
                
                // Auto-scroll
                transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
            }}
        }}

        function displayInterim(text) {{
            const transcriptDiv = document.getElementById('transcript');
            let interim = transcriptDiv.querySelector('.interim');
            
            if (!interim) {{
                interim = document.createElement('p');
                interim.className = 'interim';
                transcriptDiv.appendChild(interim);
            }}
            
            interim.textContent = text;
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
        }}

        // Connect on page load
        window.onload = () => {{
            connectWebSocket();
            initRecognition();
        }};
    </script>
</body>
</html>"""
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> List[TranscriptSegment]:
        """
        Note: Chrome Web Speech API works best with real-time audio.
        For file-based transcription, use Whisper or Azure instead.
        """
        print("⚠ Chrome Speech API is designed for real-time transcription")
        print("  For file-based transcription, use Whisper or Azure engine")
        return []


class WhisperEngine(TranscriptionEngine):
    """OpenAI Whisper local transcription engine"""
    
    def __init__(self, model_size: str = "base", language: str = "auto", task: str = "transcribe"):
        self.model_size = model_size
        self.language = language if language != "auto" else None
        self.task = task  # 'transcribe' or 'translate' (to English)
        self.model = None
        
    @property
    def name(self) -> str:
        return f"Whisper ({self.model_size})"
    
    def is_available(self) -> bool:
        """Check if Whisper is available"""
        try:
            import whisper
            return True
        except ImportError:
            return False
    
    def load_model(self):
        """Load Whisper model (lazy loading)"""
        if self.model is None:
            try:
                import whisper
                print(f"Loading Whisper model '{self.model_size}'...")
                self.model = whisper.load_model(self.model_size, download_root=str(config.MODELS_DIR))
                print(f"✓ Whisper model loaded")
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
                raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> List[TranscriptSegment]:
        """Transcribe audio using Whisper"""
        if not self.is_available():
            raise RuntimeError("Whisper is not available. Install with: pip install openai-whisper")
        
        self.load_model()
        
        try:
            # Whisper expects float32 normalized to [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Flatten if stereo
            if len(audio_float.shape) > 1:
                audio_float = audio_float.mean(axis=1)
            
            # Transcribe
            language_str = self.language if self.language else "auto-detect"
            print(f"🎤 Transcribing with Whisper (language: {language_str}, task: {self.task})...")
            print(f"   WhisperEngine.language = {self.language}")
            
            # Build transcription parameters
            transcribe_params = {
                'verbose': False,
                'task': self.task
            }
            
            # Add language if specified (not auto-detect)
            if self.language:
                transcribe_params['language'] = self.language
                print(f"   Using language parameter: {self.language}")
            
            result = self.model.transcribe(audio_float, **transcribe_params)
            
            # Log detected language
            detected_lang = result.get('language', 'unknown')
            print(f"✓ Detected/Used language: {detected_lang}")
            
            # Convert to TranscriptSegment objects
            segments = []
            for seg in result.get('segments', []):
                segment = TranscriptSegment(
                    text=seg['text'].strip(),
                    start_time=seg['start'],
                    end_time=seg['end'],
                    speaker="Unknown",
                    confidence=seg.get('confidence', 0.0)
                )
                segments.append(segment)
            
            print(f"✓ Transcription complete: {len(segments)} segments")
            return segments
            
        except Exception as e:
            print(f"Error during Whisper transcription: {e}")
            raise


class AzureSpeechEngine(TranscriptionEngine):
    """Azure Speech Service transcription engine"""
    
    def __init__(self, speech_key: str = None, region: str = None):
        self.speech_key = speech_key or config.AZURE_SPEECH_KEY
        self.region = region or config.AZURE_SPEECH_REGION
        
    @property
    def name(self) -> str:
        return "Azure Speech Service"
    
    def is_available(self) -> bool:
        """Check if Azure Speech is configured"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            return bool(self.speech_key and self.region)
        except ImportError:
            return False
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> List[TranscriptSegment]:
        """Transcribe audio using Azure Speech Service"""
        if not self.is_available():
            raise RuntimeError("Azure Speech not configured. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")
        
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                with wave.open(tmp_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data.tobytes())
                
                tmp_path = tmp_file.name
            
            # Configure Azure Speech
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.region
            )
            speech_config.speech_recognition_language = "en-US"
            
            audio_config = speechsdk.AudioConfig(filename=tmp_path)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Recognize
            print("Transcribing with Azure Speech Service...")
            result = speech_recognizer.recognize_once()
            
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
            # Process result
            segments = []
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                segment = TranscriptSegment(
                    text=result.text,
                    start_time=0.0,
                    end_time=len(audio_data) / sample_rate,
                    speaker="Unknown",
                    confidence=1.0
                )
                segments.append(segment)
                print(f"✓ Transcription complete")
            else:
                print(f"⚠ Recognition failed: {result.reason}")
            
            return segments
            
        except Exception as e:
            print(f"Error during Azure transcription: {e}")
            raise


class TranscriptionManager:
    """Manages multiple transcription engines and provides unified interface"""
    
    def __init__(self):
        self.engines: Dict[str, TranscriptionEngine] = {}
        self.active_engine: Optional[str] = None
        
        # Initialize engines
        self._register_engines()
    
    def _register_engines(self):
        """Register all available transcription engines"""
        # Chrome Web Speech
        chrome_engine = ChromeSpeechEngine()
        if chrome_engine.is_available():
            self.engines['chrome'] = chrome_engine
        
        # Whisper
        whisper_engine = WhisperEngine(
            model_size=config.WHISPER_MODEL,
            language=config.WHISPER_LANGUAGE,
            task=config.WHISPER_TASK
        )
        if whisper_engine.is_available():
            self.engines['whisper'] = whisper_engine
        
        # Azure Speech
        azure_engine = AzureSpeechEngine()
        if azure_engine.is_available():
            self.engines['azure'] = azure_engine
        
        # Set default active engine
        if config.TRANSCRIPTION_MODE in self.engines:
            self.active_engine = config.TRANSCRIPTION_MODE
        elif self.engines:
            self.active_engine = list(self.engines.keys())[0]
    
    def list_engines(self) -> List[str]:
        """List available transcription engines"""
        return list(self.engines.keys())
    
    def set_engine(self, engine_name: str) -> bool:
        """Set active transcription engine"""
        if engine_name in self.engines:
            self.active_engine = engine_name
            print(f"✓ Switched to {self.engines[engine_name].name}")
            return True
        else:
            print(f"✗ Engine '{engine_name}' not available")
            return False
    
    def get_engine(self, engine_name: str = None) -> Optional[TranscriptionEngine]:
        """Get transcription engine by name (or active engine)"""
        name = engine_name or self.active_engine
        return self.engines.get(name)
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int, 
                   engine_name: str = None) -> List[TranscriptSegment]:
        """Transcribe audio using specified or active engine"""
        engine = self.get_engine(engine_name)
        if not engine:
            raise RuntimeError(f"No transcription engine available")
        
        return engine.transcribe(audio_data, sample_rate)
    
    def export_transcript(self, segments: List[TranscriptSegment], 
                         output_path: Path, format: str = 'markdown') -> Path:
        """
        Export transcript to file.
        
        Args:
            segments: List of transcript segments
            output_path: Output file path
            format: 'txt' or 'markdown'
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format == 'markdown':
                    # Markdown format with title and metadata
                    f.write(f"# Meeting Transcript\n\n")
                    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"**Duration:** {segments[-1].end_time:.2f} seconds\n\n" if segments else "")
                    f.write(f"**Segments:** {len(segments)}\n\n")
                    f.write("---\n\n")
                    
                    for i, seg in enumerate(segments, 1):
                        timestamp = str(timedelta(seconds=int(seg.start_time)))
                        f.write(f"### Segment {i} [{timestamp}]\n\n")
                        f.write(f"{seg.text}\n\n")
                else:
                    # Plain text format
                    f.write(f"Meeting Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 70 + "\n\n")
                    
                    for seg in segments:
                        timestamp = str(timedelta(seconds=int(seg.start_time)))
                        f.write(f"[{timestamp}] {seg.text}\n")
            
            print(f"✓ Transcript exported: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error exporting transcript: {e}")
            raise


# Test functionality
if __name__ == "__main__":
    print("=== Transcription Module Test ===\n")
    
    manager = TranscriptionManager()
    
    print("Available Engines:")
    for engine_name in manager.list_engines():
        engine = manager.get_engine(engine_name)
        print(f"  ✓ {engine.name}")
    
    print(f"\nActive Engine: {manager.active_engine}")
