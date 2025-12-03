"""
Main Application - Meeting Transcription & MoM Generator
Desktop GUI for Windows using PyQt6
"""
import sys
import webbrowser
import tempfile
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QComboBox, QGroupBox, 
    QFileDialog, QMessageBox, QCheckBox, QStatusBar, QProgressBar,
    QListWidget, QDialog, QDialogButtonBox, QListWidgetItem
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QIcon
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
        self.record_speaker_check.setChecked(False)
        speaker_layout.addWidget(self.record_speaker_check)
        audio_layout.addLayout(speaker_layout)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Devices")
        refresh_btn.clicked.connect(self.load_audio_devices)
        audio_layout.addWidget(refresh_btn)
        
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
        
        self.view_recordings_btn = QPushButton("📁 View Recordings")
        self.view_recordings_btn.clicked.connect(self.view_recordings)
        recording_mgmt_layout.addWidget(self.view_recordings_btn)
        
        self.delete_recordings_btn = QPushButton("🗑 Delete Recordings")
        self.delete_recordings_btn.setStyleSheet("background-color: #dc3545; color: white;")
        self.delete_recordings_btn.clicked.connect(self.delete_recordings)
        recording_mgmt_layout.addWidget(self.delete_recordings_btn)
        
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
            self.speaker_combo.addItem("No loopback device found", None)
            self.record_speaker_check.setEnabled(False)
        
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
            QMessageBox.warning(self, "Warning", "No system audio device available. Recording microphone only.")
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
        
        # Start transcription in background
        self.transcription_thread = TranscriptionThread(
            self.transcription_manager,
            self.recorded_audio,
            config.SAMPLE_RATE,
            engine_key
        )
        self.transcription_thread.transcription_complete.connect(self.on_transcription_complete)
        self.transcription_thread.progress_update.connect(self.on_progress_update)
        self.transcription_thread.error_occurred.connect(self.on_error)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.transcribe_btn.setEnabled(False)
        
        self.transcription_thread.start()
    
    def on_transcription_complete(self, segments):
        """Handle transcription completion"""
        self.current_segments = segments
        
        # Display transcript
        self.transcript_text.clear()
        for seg in segments:
            self.transcript_text.append(f"[{seg.start_time:.1f}s] {seg.text}\n")
        
        self.progress_bar.setVisible(False)
        self.transcribe_btn.setEnabled(True)
        self.save_txt_btn.setEnabled(True)
        self.save_md_btn.setEnabled(True)
        
        self.status_bar.showMessage(f"Transcription complete: {len(segments)} segments")
    
    def on_progress_update(self, message):
        """Handle progress update"""
        self.status_bar.showMessage(message)
    
    def on_error(self, error_message):
        """Handle error"""
        QMessageBox.critical(self, "Error", error_message)
        self.status_bar.showMessage(f"Error: {error_message}")
        self.progress_bar.setVisible(False)
        self.transcribe_btn.setEnabled(True)
    
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
        """View all recorded audio files"""
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
        dialog.setWindowTitle("Recorded Files")
        dialog.setMinimumWidth(500)
        dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout(dialog)
        
        # Info label
        info_label = QLabel(f"Found {len(recordings)} recording(s) in:\n{recordings_dir}")
        layout.addWidget(info_label)
        
        # List widget
        list_widget = QListWidget()
        
        for recording in sorted(recordings, reverse=True):
            size_mb = recording.stat().st_size / (1024 * 1024)
            item_text = f"{recording.name} ({size_mb:.2f} MB)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, recording)
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        open_folder_btn = QPushButton("📂 Open Folder")
        open_folder_btn.clicked.connect(lambda: self.open_folder(recordings_dir))
        button_layout.addWidget(open_folder_btn)
        
        play_btn = QPushButton("▶ Play Selected")
        play_btn.clicked.connect(lambda: self.play_recording(list_widget.currentItem()))
        button_layout.addWidget(play_btn)
        
        delete_selected_btn = QPushButton("🗑 Delete Selected")
        delete_selected_btn.setStyleSheet("background-color: #dc3545; color: white;")
        delete_selected_btn.clicked.connect(lambda: self.delete_selected_recording(list_widget, dialog))
        button_layout.addWidget(delete_selected_btn)
        
        layout.addLayout(button_layout)
        
        # Close button
        close_btn = QPushButton("Close")
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
        """Play selected recording"""
        if not item:
            return
        
        recording_path = item.data(Qt.ItemDataRole.UserRole)
        
        try:
            import os
            if sys.platform == 'win32':
                os.startfile(recording_path)
            else:
                QMessageBox.information(
                    self,
                    "Play Recording",
                    f"Please open the file manually:\n{recording_path}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to play recording:\n{e}")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
