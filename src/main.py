"""
Main Application - Meeting Transcription & MoM Generator
Desktop GUI for Windows using PyQt6
"""
import sys
import webbrowser
import tempfile
from pathlib import Path
from datetime import datetime
from queue import Queue
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QComboBox, QGroupBox, 
    QFileDialog, QMessageBox, QCheckBox, QStatusBar, QProgressBar,
    QListWidget, QDialog, QDialogButtonBox, QListWidgetItem, QSlider
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QIcon
import pygame
import numpy as np

import config
from audio_capture import AudioCapture, AudioDevice
from transcription import TranscriptionManager, TranscriptSegment


class RecordingThread(QThread):
    """Background thread for audio recording"""
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, audio_capture: AudioCapture, mic_device, speaker_device, 
                 record_mic: bool, record_speaker: bool):
        super().__init__()
        self.audio_capture = audio_capture
        self.mic_device = mic_device
        self.speaker_device = speaker_device
        self.record_mic = record_mic
        self.record_speaker = record_speaker
    
    def run(self):
        try:
            success = self.audio_capture.start_recording(
                mic_device=self.mic_device,
                speaker_device=self.speaker_device,
                record_mic=self.record_mic,
                record_speaker=self.record_speaker
            )
            if success:
                self.status_update.emit("Recording...")
            else:
                self.error_occurred.emit("Failed to start recording")
        except Exception as e:
            self.error_occurred.emit(str(e))


class TranscriptionThread(QThread):
    """Background thread for transcription"""
    transcription_complete = pyqtSignal(list)
    progress_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, transcription_manager: TranscriptionManager, 
                 audio_data: np.ndarray, sample_rate: int, engine: str):
        super().__init__()
        self.transcription_manager = transcription_manager
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.engine = engine
    
    def run(self):
        try:
            self.progress_update.emit(f"Transcribing with {self.engine}...")
            segments = self.transcription_manager.transcribe(
                self.audio_data,
                self.sample_rate,
                engine_name=self.engine
            )
            self.transcription_complete.emit(segments)
        except Exception as e:
            self.error_occurred.emit(str(e))


class TranscriptionQueueItem:
    """Item in the transcription queue"""
    def __init__(self, audio_data: np.ndarray, sample_rate: int, engine: str, filename: str):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.engine = engine
        self.filename = filename


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setGeometry(100, 100, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        
        # Initialize modules
        self.audio_capture = AudioCapture()
        self.transcription_manager = TranscriptionManager()
        
        # State variables
        self.is_recording = False
        self.recorded_audio = None
        self.current_segments = []
        self.recording_thread = None
        self.transcription_thread = None
        self.chrome_browser_open = False
        self.last_recording_file = None
        
        # Transcription queue
        self.transcription_queue = Queue()
        self.is_transcribing = False
        self.queue_counter = 0
        
        # Setup UI
        self.init_ui()
        
        # Load devices
        self.load_audio_devices()
        
    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("🎤 Meeting Transcription & MoM Generator")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Audio Settings Group
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QVBoxLayout()
        
        # Microphone selection
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("Microphone:"))
        self.mic_combo = QComboBox()
        mic_layout.addWidget(self.mic_combo)
        self.record_mic_check = QCheckBox("Record")
        self.record_mic_check.setChecked(True)
        mic_layout.addWidget(self.record_mic_check)
        audio_layout.addLayout(mic_layout)
        
        # Speaker selection
        speaker_layout = QHBoxLayout()
        speaker_layout.addWidget(QLabel("System Audio:"))
        self.speaker_combo = QComboBox()
        speaker_layout.addWidget(self.speaker_combo)
        self.record_speaker_check = QCheckBox("Record")
        self.record_speaker_check.setChecked(True)  # Enable system audio recording by default
        speaker_layout.addWidget(self.record_speaker_check)
        audio_layout.addLayout(speaker_layout)
        
        # Buttons row for devices
        device_buttons_layout = QHBoxLayout()
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Devices")
        refresh_btn.clicked.connect(self.load_audio_devices)
        device_buttons_layout.addWidget(refresh_btn)
        
        # Sound Settings button
        self.sound_settings_btn = QPushButton("🔊 Open Sound Settings")
        self.sound_settings_btn.clicked.connect(self.open_sound_settings)
        self.sound_settings_btn.setStyleSheet("background-color: #17a2b8; color: white;")
        device_buttons_layout.addWidget(self.sound_settings_btn)
        
        audio_layout.addLayout(device_buttons_layout)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # Transcription Settings Group
        trans_group = QGroupBox("Transcription Settings")
        trans_layout = QVBoxLayout()
        
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        
        # Populate engines
        available_engines = self.transcription_manager.list_engines()
        for engine in available_engines:
            engine_obj = self.transcription_manager.get_engine(engine)
            self.engine_combo.addItem(f"{engine_obj.name}", engine)
        
        # Set active engine
        if self.transcription_manager.active_engine:
            active_idx = available_engines.index(self.transcription_manager.active_engine)
            self.engine_combo.setCurrentIndex(active_idx)
        
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        engine_layout.addWidget(self.engine_combo)
        trans_layout.addLayout(engine_layout)
        
        trans_group.setLayout(trans_layout)
        main_layout.addWidget(trans_group)
        
        # Recording Controls
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Recording")
        self.start_btn.setStyleSheet("background-color: #28a745; color: white; font-size: 14px; padding: 10px;")
        self.start_btn.clicked.connect(self.start_recording)
        controls_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop Recording")
        self.stop_btn.setStyleSheet("background-color: #dc3545; color: white; font-size: 14px; padding: 10px;")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        self.transcribe_btn = QPushButton("📝 Transcribe")
        self.transcribe_btn.setStyleSheet("background-color: #007bff; color: white; font-size: 14px; padding: 10px;")
        self.transcribe_btn.clicked.connect(self.transcribe_audio)
        self.transcribe_btn.setEnabled(False)
        controls_layout.addWidget(self.transcribe_btn)
        
        main_layout.addLayout(controls_layout)
        
        # Chrome Web Speech button (shown only when chrome engine selected)
        self.chrome_btn = QPushButton("🌐 Open Chrome Speech Recognition")
        self.chrome_btn.setStyleSheet("background-color: #17a2b8; color: white; font-size: 12px; padding: 8px;")
        self.chrome_btn.clicked.connect(self.open_chrome_speech)
        self.chrome_btn.setVisible(False)
        main_layout.addWidget(self.chrome_btn)
        
        # Transcript Display
        transcript_label = QLabel("Transcript:")
        transcript_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        main_layout.addWidget(transcript_label)
        
        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setPlaceholderText("Transcript will appear here...")
        main_layout.addWidget(self.transcript_text)
        
        # Export Controls
        export_layout = QHBoxLayout()
        
        self.save_txt_btn = QPushButton("💾 Save as TXT")
        self.save_txt_btn.clicked.connect(lambda: self.save_transcript('txt'))
        self.save_txt_btn.setEnabled(False)
        export_layout.addWidget(self.save_txt_btn)
        
        self.save_md_btn = QPushButton("📄 Save as Markdown")
        self.save_md_btn.clicked.connect(lambda: self.save_transcript('markdown'))
        self.save_md_btn.setEnabled(False)
        export_layout.addWidget(self.save_md_btn)
        
        self.clear_btn = QPushButton("🗑 Clear")
        self.clear_btn.clicked.connect(self.clear_transcript)
        export_layout.addWidget(self.clear_btn)
        
        main_layout.addLayout(export_layout)
        
        # Recording Management Controls
        recording_mgmt_layout = QHBoxLayout()
        
        self.view_recordings_btn = QPushButton("📁 Recording Manager")
        self.view_recordings_btn.setStyleSheet("background-color: #6c757d; color: white; font-size: 13px; padding: 10px;")
        self.view_recordings_btn.clicked.connect(self.view_recordings)
        recording_mgmt_layout.addWidget(self.view_recordings_btn)
        
        main_layout.addLayout(recording_mgmt_layout)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def load_audio_devices(self):
        """Load available audio devices into combo boxes"""
        self.mic_combo.clear()
        self.speaker_combo.clear()
        
        input_devices, loopback_devices = self.audio_capture.list_devices()
        
        # Populate microphone combo
        for dev in input_devices:
            self.mic_combo.addItem(dev.name, dev)
        
        # Populate speaker combo
        if loopback_devices:
            for dev in loopback_devices:
                self.speaker_combo.addItem(dev.name, dev)
        else:
            self.speaker_combo.addItem("No system audio device available", None)
            self.record_speaker_check.setEnabled(False)
            self.record_speaker_check.setChecked(False)
            self.status_bar.showMessage("⚠ Enable 'Stereo Mix' in Windows Sound Settings to record system audio", 5000)
        
        # Try to select defaults
        default_mic, default_speaker = self.audio_capture.get_default_devices()
        if default_mic:
            for i in range(self.mic_combo.count()):
                if self.mic_combo.itemData(i).index == default_mic.index:
                    self.mic_combo.setCurrentIndex(i)
                    break
        
        self.status_bar.showMessage(f"Loaded {len(input_devices)} input devices, "
                                   f"{len(loopback_devices)} loopback devices")
    
    def on_engine_changed(self, index):
        """Handle transcription engine change"""
        engine_key = self.engine_combo.itemData(index)
        self.transcription_manager.set_engine(engine_key)
        
        # Show/hide Chrome button
        self.chrome_btn.setVisible(engine_key == 'chrome')
    
    def start_recording(self):
        """Start audio recording"""
        if not self.record_mic_check.isChecked() and not self.record_speaker_check.isChecked():
            QMessageBox.warning(self, "Warning", "Please select at least one audio source to record.")
            return
        
        # Get selected devices
        mic_device = self.mic_combo.currentData() if self.record_mic_check.isChecked() else None
        speaker_device = self.speaker_combo.currentData() if self.record_speaker_check.isChecked() else None
        
        if speaker_device is None and self.record_speaker_check.isChecked():
            reply = QMessageBox.warning(
                self,
                "System Audio Not Available",
                "⚠️ System audio (Stereo Mix) is not available!\n\n"
                "This means you will ONLY record YOUR voice through the microphone.\n"
                "Meeting participants' voices will NOT be recorded.\n\n"
                "To record meeting audio:\n"
                "1. Click '🔊 Open Sound Settings' button\n"
                "2. Enable 'Stereo Mix' device\n"
                "3. Restart recording\n\n"
                "Continue with microphone only?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                return
            
            self.record_speaker_check.setChecked(False)
        
        # Start recording in background thread
        self.recording_thread = RecordingThread(
            self.audio_capture,
            mic_device,
            speaker_device,
            self.record_mic_check.isChecked(),
            self.record_speaker_check.isChecked()
        )
        self.recording_thread.status_update.connect(self.on_recording_started)
        self.recording_thread.error_occurred.connect(self.on_error)
        self.recording_thread.start()
    
    def on_recording_started(self, message):
        """Handle recording started event"""
        self.is_recording = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.transcribe_btn.setEnabled(False)
        self.status_bar.showMessage(message)
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            return
        
        mic_audio, speaker_audio = self.audio_capture.stop_recording()
        
        # Merge audio channels if both recorded
        if mic_audio is not None or speaker_audio is not None:
            self.recorded_audio = self.audio_capture.merge_audio_channels(mic_audio, speaker_audio)
            
            # Save recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.audio_capture.save_audio(self.recorded_audio, f"recording_{timestamp}")
            
            self.status_bar.showMessage(f"Recording stopped. Duration: {len(self.recorded_audio) / config.SAMPLE_RATE:.2f}s")
            self.transcribe_btn.setEnabled(True)
            
            # Store the last recording filename for playback
            self.last_recording_file = config.RECORDINGS_DIR / f"recording_{timestamp}.wav"
        else:
            self.status_bar.showMessage("Recording stopped (no audio captured)")
        
        self.is_recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def transcribe_audio(self):
        """Transcribe recorded audio"""
        if self.recorded_audio is None:
            QMessageBox.warning(self, "Warning", "No audio to transcribe. Please record first.")
            return
        
        engine_key = self.engine_combo.currentData()
        
        # Check if using Chrome engine
        if engine_key == 'chrome':
            QMessageBox.information(
                self, 
                "Chrome Web Speech", 
                "Chrome Web Speech API is designed for real-time transcription.\n\n"
                "For file-based transcription, please:\n"
                "1. Switch to Whisper or Azure engine, OR\n"
                "2. Use 'Open Chrome Speech Recognition' before recording"
            )
            return
        
        # Add to transcription queue
        self.queue_counter += 1
        queue_item = TranscriptionQueueItem(
            audio_data=self.recorded_audio.copy(),
            sample_rate=config.SAMPLE_RATE,
            engine=engine_key,
            filename=f"Recording #{self.queue_counter}"
        )
        self.transcription_queue.put(queue_item)
        
        queue_size = self.transcription_queue.qsize()
        if queue_size > 1:
            self.status_bar.showMessage(f"Added to queue. Position: {queue_size}")
        
        # Process queue if not already transcribing
        if not self.is_transcribing:
            self.process_transcription_queue()
    
    def process_transcription_queue(self):
        """Process next item in transcription queue"""
        if self.transcription_queue.empty():
            self.is_transcribing = False
            self.transcribe_btn.setEnabled(True)
            self.status_bar.showMessage("All transcriptions complete")
            return
        
        self.is_transcribing = True
        queue_item = self.transcription_queue.get()
        
        remaining = self.transcription_queue.qsize()
        status_msg = f"Transcribing {queue_item.filename}"
        if remaining > 0:
            status_msg += f" ({remaining} in queue)"
        self.status_bar.showMessage(status_msg)
        
        # Start transcription in background
        self.transcription_thread = TranscriptionThread(
            self.transcription_manager,
            queue_item.audio_data,
            queue_item.sample_rate,
            queue_item.engine
        )
        self.transcription_thread.transcription_complete.connect(self.on_transcription_complete)
        self.transcription_thread.progress_update.connect(self.on_progress_update)
        self.transcription_thread.error_occurred.connect(self.on_error)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.transcribe_btn.setEnabled(False)
        
        self.transcription_thread.start()
    
    def transcribe_from_file(self, item, dialog):
        """Transcribe audio from a selected recording file"""
        if not item:
            QMessageBox.warning(self, "No Selection", "Please select a recording to transcribe.")
            return
        
        recording_path = item.data(Qt.ItemDataRole.UserRole)
        
        try:
            # Load audio file
            import wave
            import numpy as np
            
            with wave.open(str(recording_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Close the recording manager dialog
            dialog.accept()
            
            # Get selected engine
            engine_key = self.engine_combo.currentData()
            
            # Check if using Chrome engine
            if engine_key == 'chrome':
                QMessageBox.information(
                    self, 
                    "Chrome Web Speech", 
                    "Chrome Web Speech API is designed for real-time transcription.\n\n"
                    "For file-based transcription, please switch to Whisper or Azure engine."
                )
                return
            
            # Add to transcription queue
            self.queue_counter += 1
            queue_item = TranscriptionQueueItem(
                audio_data=audio_data.copy(),
                sample_rate=sample_rate,
                engine=engine_key,
                filename=recording_path.name
            )
            self.transcription_queue.put(queue_item)
            
            queue_size = self.transcription_queue.qsize()
            if queue_size > 1:
                self.status_bar.showMessage(f"Added '{recording_path.name}' to queue. Position: {queue_size}")
            else:
                self.status_bar.showMessage(f"Transcribing '{recording_path.name}'...")
            
            # Process queue if not already transcribing
            if not self.is_transcribing:
                self.process_transcription_queue()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{e}")
    
    def on_transcription_complete(self, segments):
        """Handle transcription completion"""
        self.current_segments.extend(segments)
        
        # Display transcript (append new segments)
        for seg in segments:
            self.transcript_text.append(f"[{seg.start_time:.1f}s] {seg.text}\n")
        
        self.progress_bar.setVisible(False)
        self.save_txt_btn.setEnabled(True)
        self.save_md_btn.setEnabled(True)
        
        # Process next item in queue
        self.process_transcription_queue()
    
    def on_progress_update(self, message):
        """Handle progress update"""
        self.status_bar.showMessage(message)
    
    def on_error(self, error_message):
        """Handle error"""
        QMessageBox.critical(self, "Error", error_message)
        self.status_bar.showMessage(f"Error: {error_message}")
        self.progress_bar.setVisible(False)
        
        # Process next item in queue even after error
        self.process_transcription_queue()
    
    def open_chrome_speech(self):
        """Open Chrome Web Speech recognition page"""
        try:
            # Get Chrome engine
            chrome_engine = self.transcription_manager.get_engine('chrome')
            if not chrome_engine:
                QMessageBox.warning(self, "Warning", "Chrome engine not available")
                return
            
            # Start WebSocket server if not running
            if not chrome_engine.is_running:
                # Set up callback for real-time transcription
                def on_transcript(text, timestamp, confidence, is_final):
                    if is_final:
                        self.transcript_text.append(f"[{timestamp:.1f}s] {text}")
                        # Create segment
                        segment = TranscriptSegment(
                            text=text,
                            start_time=timestamp,
                            end_time=timestamp + 2.0,  # Estimate
                            confidence=confidence
                        )
                        self.current_segments.append(segment)
                        self.save_txt_btn.setEnabled(True)
                        self.save_md_btn.setEnabled(True)
                
                chrome_engine.start_server(callback=on_transcript)
            
            # Create temporary HTML file
            html_content = chrome_engine.get_html_page()
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
                f.write(html_content)
                html_path = f.name
            
            # Open in browser
            webbrowser.open(f'file:///{html_path}')
            self.chrome_browser_open = True
            self.status_bar.showMessage("Chrome Speech Recognition opened in browser")
            
            QMessageBox.information(
                self,
                "Chrome Speech Recognition",
                "Browser page opened!\n\n"
                "1. Click 'Start Listening' in the browser\n"
                "2. Grant microphone permissions if asked\n"
                "3. Speak and watch the transcript appear here\n\n"
                "Transcript will sync to this app in real-time."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Chrome Speech: {e}")
    
    def save_transcript(self, format_type):
        """Save transcript to file"""
        if not self.current_segments:
            QMessageBox.warning(self, "Warning", "No transcript to save")
            return
        
        # Get save location
        default_name = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_extension = "txt" if format_type == "txt" else "md"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcript",
            str(config.TRANSCRIPTS_DIR / f"{default_name}.{file_extension}"),
            f"{format_type.upper()} Files (*.{file_extension})"
        )
        
        if file_path:
            try:
                output_path = Path(file_path)
                self.transcription_manager.export_transcript(
                    self.current_segments,
                    output_path,
                    format=format_type
                )
                QMessageBox.information(self, "Success", f"Transcript saved to:\n{output_path}")
                self.status_bar.showMessage(f"Transcript saved: {output_path.name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save transcript: {e}")
    
    def clear_transcript(self):
        """Clear transcript display"""
        reply = QMessageBox.question(
            self,
            "Clear Transcript",
            "Are you sure you want to clear the transcript?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.transcript_text.clear()
            self.current_segments = []
            self.save_txt_btn.setEnabled(False)
            self.save_md_btn.setEnabled(False)
            self.status_bar.showMessage("Transcript cleared")
    
    def view_recordings(self):
        """Open Recording Manager window"""
        recordings_dir = config.RECORDINGS_DIR
        
        if not recordings_dir.exists():
            QMessageBox.information(self, "No Recordings", "No recordings folder found.")
            return
        
        # Get all WAV files
        recordings = list(recordings_dir.glob("*.wav"))
        
        if not recordings:
            QMessageBox.information(self, "No Recordings", "No recordings found.")
            return
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("📁 Recording Manager")
        dialog.setMinimumWidth(700)
        dialog.setMinimumHeight(500)
        
        layout = QVBoxLayout(dialog)
        
        # Header
        header_layout = QHBoxLayout()
        info_label = QLabel(f"📊 Total Recordings: {len(recordings)}")
        info_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        header_layout.addWidget(info_label)
        header_layout.addStretch()
        
        folder_label = QLabel(f"📂 {recordings_dir}")
        folder_label.setStyleSheet("color: gray;")
        header_layout.addWidget(folder_label)
        layout.addLayout(header_layout)
        
        # List widget
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        
        for recording in sorted(recordings, reverse=True):
            size_mb = recording.stat().st_size / (1024 * 1024)
            modified_time = datetime.fromtimestamp(recording.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            item_text = f"🎵 {recording.name}  |  {size_mb:.2f} MB  |  {modified_time}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, recording)
            list_widget.addItem(item)
        
        # Double-click to play
        list_widget.itemDoubleClicked.connect(lambda item: self.play_recording(item))
        
        layout.addWidget(list_widget)
        
        # Action Buttons Row 1
        button_layout1 = QHBoxLayout()
        
        play_btn = QPushButton("▶️ Play Selected")
        play_btn.setStyleSheet("background-color: #28a745; color: white; font-size: 12px; padding: 8px;")
        play_btn.clicked.connect(lambda: self.play_recording(list_widget.currentItem()))
        button_layout1.addWidget(play_btn)
        
        transcribe_btn = QPushButton("📝 Transcribe Selected")
        transcribe_btn.setStyleSheet("background-color: #007bff; color: white; font-size: 12px; padding: 8px; font-weight: bold;")
        transcribe_btn.clicked.connect(lambda: self.transcribe_from_file(list_widget.currentItem(), dialog))
        button_layout1.addWidget(transcribe_btn)
        
        delete_selected_btn = QPushButton("🗑 Delete Selected")
        delete_selected_btn.setStyleSheet("background-color: #dc3545; color: white; font-size: 12px; padding: 8px;")
        delete_selected_btn.clicked.connect(lambda: self.delete_selected_recording(list_widget, dialog))
        button_layout1.addWidget(delete_selected_btn)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("background-color: #17a2b8; color: white; font-size: 12px; padding: 8px;")
        refresh_btn.clicked.connect(lambda: [dialog.close(), self.view_recordings()])
        button_layout1.addWidget(refresh_btn)
        
        layout.addLayout(button_layout1)
        
        # Action Buttons Row 2
        button_layout2 = QHBoxLayout()
        
        open_folder_btn = QPushButton("📂 Open Folder")
        open_folder_btn.clicked.connect(lambda: self.open_folder(recordings_dir))
        button_layout2.addWidget(open_folder_btn)
        
        delete_all_btn = QPushButton("🗑 Delete All Recordings")
        delete_all_btn.setStyleSheet("background-color: #bd2130; color: white; font-weight: bold; font-size: 12px; padding: 8px;")
        delete_all_btn.clicked.connect(lambda: [self.delete_all_recordings_from_manager(dialog), dialog.close()])
        button_layout2.addWidget(delete_all_btn)
        
        layout.addLayout(button_layout2)
        
        # Info text
        info_text = QLabel("💡 Tip: Double-click to play | Select and click 'Transcribe' to transcribe old recordings")
        info_text.setStyleSheet("color: gray; font-style: italic; margin-top: 5px;")
        layout.addWidget(info_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("margin-top: 10px;")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def delete_recordings(self):
        """Delete all recordings (compliance)"""
        recordings_dir = config.RECORDINGS_DIR
        
        if not recordings_dir.exists():
            QMessageBox.information(self, "No Recordings", "No recordings folder found.")
            return
        
        recordings = list(recordings_dir.glob("*.wav"))
        
        if not recordings:
            QMessageBox.information(self, "No Recordings", "No recordings found.")
            return
        
        # Confirm deletion
        reply = QMessageBox.warning(
            self,
            "Delete All Recordings",
            f"⚠ WARNING: This will permanently delete ALL {len(recordings)} recording(s).\n\n"
            "This action cannot be undone.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            failed_count = 0
            
            for recording in recordings:
                try:
                    recording.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {recording.name}: {e}")
                    failed_count += 1
            
            if failed_count == 0:
                QMessageBox.information(
                    self,
                    "Success",
                    f"✓ Successfully deleted {deleted_count} recording(s)."
                )
                self.status_bar.showMessage(f"Deleted {deleted_count} recordings")
            else:
                QMessageBox.warning(
                    self,
                    "Partial Success",
                    f"Deleted {deleted_count} recording(s).\n"
                    f"Failed to delete {failed_count} file(s)."
                )
    
    def delete_selected_recording(self, list_widget, dialog):
        """Delete selected recording from list"""
        current_item = list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(dialog, "No Selection", "Please select a recording to delete.")
            return
        
        recording_path = current_item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.warning(
            dialog,
            "Delete Recording",
            f"Delete this recording?\n\n{recording_path.name}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                recording_path.unlink()
                row = list_widget.row(current_item)
                list_widget.takeItem(row)
                QMessageBox.information(dialog, "Deleted", f"Deleted: {recording_path.name}")
            except Exception as e:
                QMessageBox.critical(dialog, "Error", f"Failed to delete:\n{e}")
    
    def open_folder(self, folder_path):
        """Open folder in file explorer"""
        import subprocess
        import os
        
        if sys.platform == 'win32':
            os.startfile(folder_path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', folder_path])
        else:
            subprocess.Popen(['xdg-open', folder_path])
    
    def play_recording(self, item):
        """Play selected recording using built-in player"""
        if not item:
            return
        
        recording_path = item.data(Qt.ItemDataRole.UserRole)
        self.open_audio_player(recording_path)

    def open_audio_player(self, audio_path):
        """Open built-in audio player dialog using pygame"""
        player_dialog = QDialog(self)
        player_dialog.setWindowTitle(f"🎵 Audio Player - {audio_path.name}")
        player_dialog.setMinimumWidth(450)
        player_dialog.setMinimumHeight(220)
        
        layout = QVBoxLayout(player_dialog)
        
        # File info
        file_size = audio_path.stat().st_size / (1024 * 1024)
        info_label = QLabel(f"📂 {audio_path.name}\n📊 Size: {file_size:.2f} MB")
        info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px; font-size: 11px;")
        layout.addWidget(info_label)
        
        # Initialize pygame mixer
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(str(audio_path))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{e}")
            player_dialog.reject()
            return
        
        # Status label
        status_label = QLabel("▶️ Playing...")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #28a745; padding: 10px;")
        layout.addWidget(status_label)
        
        # Control buttons
        controls = QHBoxLayout()
        
        is_playing = {'state': False}
        
        play_pause_btn = QPushButton("▶️ Play")
        play_pause_btn.setStyleSheet("background-color: #28a745; color: white; font-size: 14px; padding: 10px; min-width: 100px;")
        
        def toggle_play_pause():
            if is_playing['state']:
                pygame.mixer.music.pause()
                is_playing['state'] = False
                play_pause_btn.setText("▶️ Play")
                play_pause_btn.setStyleSheet("background-color: #28a745; color: white; font-size: 14px; padding: 10px; min-width: 100px;")
                status_label.setText("⏸ Paused")
                status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffc107; padding: 10px;")
            else:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.unpause()
                else:
                    pygame.mixer.music.play()
                is_playing['state'] = True
                play_pause_btn.setText("⏸ Pause")
                play_pause_btn.setStyleSheet("background-color: #ffc107; color: white; font-size: 14px; padding: 10px; min-width: 100px;")
                status_label.setText("▶️ Playing...")
                status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #28a745; padding: 10px;")
        
        play_pause_btn.clicked.connect(toggle_play_pause)
        controls.addWidget(play_pause_btn)
        
        stop_btn = QPushButton("⏹ Stop")
        stop_btn.setStyleSheet("background-color: #dc3545; color: white; font-size: 14px; padding: 10px; min-width: 100px;")
        
        def stop_playback():
            pygame.mixer.music.stop()
            is_playing['state'] = False
            play_pause_btn.setText("▶️ Play")
            play_pause_btn.setStyleSheet("background-color: #28a745; color: white; font-size: 14px; padding: 10px; min-width: 100px;")
            status_label.setText("⏹ Stopped")
            status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #dc3545; padding: 10px;")
        
        stop_btn.clicked.connect(stop_playback)
        controls.addWidget(stop_btn)
        
        layout.addLayout(controls)
        
        # Volume control
        volume_layout = QHBoxLayout()
        volume_label = QLabel("🔊 Volume:")
        volume_layout.addWidget(volume_label)
        
        volume_slider = QSlider(Qt.Orientation.Horizontal)
        volume_slider.setRange(0, 100)
        volume_slider.setValue(70)
        volume_slider.valueChanged.connect(lambda v: pygame.mixer.music.set_volume(v / 100))
        pygame.mixer.music.set_volume(0.7)
        volume_layout.addWidget(volume_slider)
        
        volume_value_label = QLabel("70%")
        volume_slider.valueChanged.connect(lambda v: volume_value_label.setText(f"{v}%"))
        volume_layout.addWidget(volume_value_label)
        
        layout.addLayout(volume_layout)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("margin-top: 10px; padding: 8px;")
        close_btn.clicked.connect(lambda: [stop_playback(), player_dialog.accept()])
        layout.addWidget(close_btn)
        
        # Auto-play
        pygame.mixer.music.play()
        is_playing['state'] = True
        play_pause_btn.setText("⏸ Pause")
        play_pause_btn.setStyleSheet("background-color: #ffc107; color: white; font-size: 14px; padding: 10px; min-width: 100px;")
        
        player_dialog.exec()
        pygame.mixer.music.stop()
    
    def play_current_recording(self):
        """Play the most recent recording (kept for backward compatibility)"""
        if self.last_recording_file and self.last_recording_file.exists():
            try:
                self.open_audio_player(self.last_recording_file)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to play recording:\n{e}")
        else:
            QMessageBox.warning(self, "No Recording", "No recording available to play.")

    def delete_all_recordings(self):
        """Delete all recordings in the recordings folder"""
        recordings = list(config.RECORDINGS_DIR.glob("*.wav"))
        
        if not recordings:
            QMessageBox.information(self, "No Recordings", "No recordings to delete.")
            return
        
        reply = QMessageBox.warning(
            self,
            "Delete All Recordings",
            f"Are you sure you want to delete ALL {len(recordings)} recordings?\n\n"
            "This action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            failed_count = 0
            
            for recording in recordings:
                try:
                    recording.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {recording.name}: {e}")
                    failed_count += 1
            
            if failed_count > 0:
                QMessageBox.warning(
                    self,
                    "Deletion Complete",
                    f"Deleted {deleted_count} recordings.\n{failed_count} failed to delete."
                )
            else:
                QMessageBox.information(
                    self,
                    "Deletion Complete",
                    f"Successfully deleted all {deleted_count} recordings."
                )
            
    def delete_all_recordings_from_manager(self, dialog):
        """Delete all recordings from the Recording Manager window"""
        recordings = list(config.RECORDINGS_DIR.glob("*.wav"))
        
        if not recordings:
            QMessageBox.information(self, "No Recordings", "No recordings to delete.")
            return
        
        reply = QMessageBox.warning(
            self,
            "Delete All Recordings",
            f"Are you sure you want to delete ALL {len(recordings)} recordings?\n\n"
            "This action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            failed_count = 0
            
            for recording in recordings:
                try:
                    recording.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {recording.name}: {e}")
                    failed_count += 1
            
            if failed_count > 0:
                QMessageBox.warning(
                    self,
                    "Deletion Complete",
                    f"Deleted {deleted_count} recordings.\n{failed_count} failed to delete."
                )
            else:
                QMessageBox.information(
                    self,
                    "Deletion Complete",
                    f"Successfully deleted all {deleted_count} recordings."
                )
            
            self.status_bar.showMessage(f"Deleted {deleted_count} recordings")

    def open_sound_settings(self):
        """Open Windows Sound Settings to enable Stereo Mix"""
        import subprocess
        try:
            # Open classic Sound Control Panel (has Recording tab)
            subprocess.Popen(['control', 'mmsys.cpl', 'sounds', ',1'])
            
            QMessageBox.information(
                self,
                "Enable Stereo Mix",
                "To record system audio (what you hear):\n\n"
                "1. In the 'Recording' tab that just opened\n"
                "2. Right-click in empty space → 'Show Disabled Devices'\n"
                "3. Find 'Stereo Mix' in the list\n"
                "4. Right-click 'Stereo Mix' → 'Enable'\n"
                "5. Click 'OK'\n"
                "6. Click '🔄 Refresh Devices' in this app\n\n"
                "Then you'll be able to record what you hear in your headphones!"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open Sound Settings:\n{e}")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
