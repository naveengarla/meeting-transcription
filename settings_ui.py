"""
Settings UI Dialog
PyQt6 GUI for configuring application settings (replaces .env file editing).
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QComboBox, QPushButton, QSpinBox, QLineEdit, QFileDialog,
    QGroupBox, QFormLayout, QCheckBox, QMessageBox, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class SettingsDialog(QDialog):
    """Main settings dialog window."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Meeting Transcription - Settings")
        self.setMinimumSize(700, 600)
        
        # Load current settings
        self.settings = self.load_settings()
        
        # Create UI
        self.init_ui()
        
        # Populate with current values
        self.populate_settings()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Application Settings")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_transcription_tab(), "Transcription")
        tabs.addTab(self.create_audio_tab(), "Audio Devices")
        tabs.addTab(self.create_performance_tab(), "Performance")
        tabs.addTab(self.create_advanced_tab(), "Advanced")
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)
        self.save_btn.setDefault(True)
        button_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def create_transcription_tab(self) -> QWidget:
        """Create transcription settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Engine selection
        engine_group = QGroupBox("Transcription Engine")
        engine_layout = QFormLayout()
        
        self.engine_combo = QComboBox()
        self.engine_combo.addItems([
            "faster-whisper (Local, Offline, Recommended)",
            "whisper (Local, Offline, Original OpenAI)",
            "azure (Cloud, Requires API Key)",
            "web-speech (Browser-based, Microphone only)"
        ])
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        engine_layout.addRow("Engine:", self.engine_combo)
        
        engine_group.setLayout(engine_layout)
        layout.addWidget(engine_group)
        
        # Whisper model settings
        self.whisper_group = QGroupBox("Whisper Model Settings")
        whisper_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "tiny (75 MB, Very Fast, Basic Quality)",
            "base (142 MB, Fast, Good Quality) - Recommended",
            "small (466 MB, Medium, Better Quality)",
            "medium (1.5 GB, Slow, Best Quality)"
        ])
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        whisper_layout.addRow("Model Size:", self.model_combo)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "auto (Auto-detect)",
            "en (English)",
            "es (Spanish)",
            "fr (French)",
            "de (German)",
            "it (Italian)",
            "pt (Portuguese)",
            "zh (Chinese)",
            "ja (Japanese)",
            "ko (Korean)"
        ])
        whisper_layout.addRow("Language:", self.language_combo)
        
        self.task_combo = QComboBox()
        self.task_combo.addItems([
            "transcribe (Keep original language)",
            "translate (Translate to English)"
        ])
        whisper_layout.addRow("Task:", self.task_combo)
        
        # Model info
        self.model_info = QLabel()
        self.model_info.setWordWrap(True)
        whisper_layout.addRow("Info:", self.model_info)
        
        self.whisper_group.setLayout(whisper_layout)
        layout.addWidget(self.whisper_group)
        
        # Azure settings
        self.azure_group = QGroupBox("Azure Speech Service Settings")
        azure_layout = QFormLayout()
        
        self.azure_key_edit = QLineEdit()
        self.azure_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.azure_key_edit.setPlaceholderText("Enter your Azure Speech API key")
        azure_layout.addRow("API Key:", self.azure_key_edit)
        
        self.azure_region_edit = QLineEdit()
        self.azure_region_edit.setPlaceholderText("e.g., eastus, westus2")
        azure_layout.addRow("Region:", self.azure_region_edit)
        
        self.azure_group.setLayout(azure_layout)
        self.azure_group.setVisible(False)
        layout.addWidget(self.azure_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_audio_tab(self) -> QWidget:
        """Create audio device settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Detect devices button
        detect_btn = QPushButton("🔍 Detect Audio Devices")
        detect_btn.clicked.connect(self.detect_audio_devices)
        layout.addWidget(detect_btn)
        
        # Microphone settings
        mic_group = QGroupBox("Microphone (Your Voice)")
        mic_layout = QFormLayout()
        
        self.mic_combo = QComboBox()
        mic_layout.addRow("Device:", self.mic_combo)
        
        mic_group.setLayout(mic_layout)
        layout.addWidget(mic_group)
        
        # Speaker/Stereo Mix settings
        speaker_group = QGroupBox("System Audio / Stereo Mix (Meeting Audio)")
        speaker_layout = QFormLayout()
        
        self.speaker_combo = QComboBox()
        speaker_layout.addRow("Device:", self.speaker_combo)
        
        self.stereo_mix_info = QLabel()
        self.stereo_mix_info.setWordWrap(True)
        speaker_layout.addRow("Info:", self.stereo_mix_info)
        
        help_btn = QPushButton("❓ How to enable Stereo Mix")
        help_btn.clicked.connect(self.show_stereo_mix_help)
        speaker_layout.addRow("", help_btn)
        
        speaker_group.setLayout(speaker_layout)
        layout.addWidget(speaker_group)
        
        # Audio format
        format_group = QGroupBox("Audio Format")
        format_layout = QFormLayout()
        
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["16000 (Recommended)", "44100", "48000"])
        format_layout.addRow("Sample Rate:", self.sample_rate_combo)
        
        self.channels_combo = QComboBox()
        self.channels_combo.addItems(["1 (Mono)", "2 (Stereo)"])
        format_layout.addRow("Channels:", self.channels_combo)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_performance_tab(self) -> QWidget:
        """Create performance settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # CPU settings
        cpu_group = QGroupBox("CPU Settings")
        cpu_layout = QFormLayout()
        
        self.cpu_threads_spin = QSpinBox()
        self.cpu_threads_spin.setMinimum(1)
        self.cpu_threads_spin.setMaximum(32)
        
        # Auto-detect CPU cores
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True)
            self.cpu_threads_spin.setValue(max(1, cpu_count - 2))
            cpu_info = QLabel(f"System has {cpu_count} CPU threads")
        except:
            self.cpu_threads_spin.setValue(4)
            cpu_info = QLabel("Unable to detect CPU cores")
        
        cpu_layout.addRow("Worker Threads:", self.cpu_threads_spin)
        cpu_layout.addRow("", cpu_info)
        
        cpu_group.setLayout(cpu_layout)
        layout.addWidget(cpu_group)
        
        # Memory settings
        mem_group = QGroupBox("Memory Settings")
        mem_layout = QFormLayout()
        
        self.auto_fallback_check = QCheckBox("Automatically use smaller model on MemoryError")
        self.auto_fallback_check.setChecked(True)
        mem_layout.addRow("", self.auto_fallback_check)
        
        # Show system memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            mem_gb = mem.total / (1024**3)
            mem_info = QLabel(f"System has {mem_gb:.1f} GB RAM\n"
                            f"Recommended: tiny/base for <8GB, small for 8-16GB, medium for 16GB+")
        except:
            mem_info = QLabel("Unable to detect system memory")
        
        mem_info.setWordWrap(True)
        mem_layout.addRow("System Info:", mem_info)
        
        mem_group.setLayout(mem_layout)
        layout.addWidget(mem_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()
        
        output_path_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("transcripts/")
        output_path_layout.addWidget(self.output_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output_folder)
        output_path_layout.addWidget(browse_btn)
        
        output_layout.addRow("Output Folder:", output_path_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Logging settings
        log_group = QGroupBox("Logging")
        log_layout = QFormLayout()
        
        self.log_performance_check = QCheckBox("Log performance metrics")
        self.log_performance_check.setChecked(True)
        log_layout.addRow("", self.log_performance_check)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Reset button
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def load_settings(self) -> dict:
        """Load settings from .env file."""
        settings = {
            'TRANSCRIPTION_MODE': 'whisper',
            'WHISPER_MODEL': 'base',
            'WHISPER_LANGUAGE': 'en',
            'WHISPER_TASK': 'transcribe',
            'AZURE_SPEECH_KEY': '',
            'AZURE_SPEECH_REGION': '',
            'PREFERRED_MICROPHONE': '',
            'PREFERRED_SPEAKER': '',
            'SAMPLE_RATE': '16000',
            'CHANNELS': '2',
            'NUM_WORKERS': '4',
            'OUTPUT_DIR': 'transcripts',
        }
        
        # Load from .env if exists
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key in settings:
                            settings[key] = value
        
        return settings
    
    def populate_settings(self):
        """Populate UI with current settings."""
        # Transcription engine
        mode = self.settings.get('TRANSCRIPTION_MODE', 'whisper')
        if mode == 'whisper':
            self.engine_combo.setCurrentIndex(0)  # faster-whisper
        elif mode == 'azure':
            self.engine_combo.setCurrentIndex(2)
        elif mode == 'web-speech':
            self.engine_combo.setCurrentIndex(3)
        
        # Model
        model = self.settings.get('WHISPER_MODEL', 'base')
        model_index = {'tiny': 0, 'base': 1, 'small': 2, 'medium': 3}.get(model, 1)
        self.model_combo.setCurrentIndex(model_index)
        
        # Language
        language = self.settings.get('WHISPER_LANGUAGE', 'en')
        for i in range(self.language_combo.count()):
            if self.language_combo.itemText(i).startswith(language):
                self.language_combo.setCurrentIndex(i)
                break
        
        # Task
        task = self.settings.get('WHISPER_TASK', 'transcribe')
        self.task_combo.setCurrentIndex(0 if task == 'transcribe' else 1)
        
        # Azure
        self.azure_key_edit.setText(self.settings.get('AZURE_SPEECH_KEY', ''))
        self.azure_region_edit.setText(self.settings.get('AZURE_SPEECH_REGION', ''))
        
        # Audio format
        sample_rate = self.settings.get('SAMPLE_RATE', '16000')
        for i in range(self.sample_rate_combo.count()):
            if sample_rate in self.sample_rate_combo.itemText(i):
                self.sample_rate_combo.setCurrentIndex(i)
                break
        
        channels = self.settings.get('CHANNELS', '2')
        self.channels_combo.setCurrentIndex(0 if channels == '1' else 1)
        
        # Performance
        num_workers = int(self.settings.get('NUM_WORKERS', '4'))
        self.cpu_threads_spin.setValue(num_workers)
        
        # Output
        self.output_edit.setText(self.settings.get('OUTPUT_DIR', 'transcripts'))
        
        # Detect audio devices
        self.detect_audio_devices()
        
        # Update UI state
        self.on_engine_changed()
        self.on_model_changed()
    
    def detect_audio_devices(self):
        """Detect and populate audio device lists."""
        try:
            import sounddevice as sd
            
            devices = sd.query_devices()
            
            # Clear combos
            self.mic_combo.clear()
            self.speaker_combo.clear()
            
            # Populate microphones
            mic_devices = []
            stereo_mix_devices = []
            
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    name = device['name']
                    
                    # Check if it's stereo mix / loopback
                    if any(keyword in name.lower() for keyword in 
                           ['stereo mix', 'what u hear', 'wave out mix', 'loopback']):
                        stereo_mix_devices.append((idx, name))
                    else:
                        mic_devices.append((idx, name))
            
            # Add microphones
            for idx, name in mic_devices:
                self.mic_combo.addItem(name, idx)
            
            # Add stereo mix devices
            for idx, name in stereo_mix_devices:
                self.speaker_combo.addItem(name, idx)
            
            # Set current selections
            pref_mic = self.settings.get('PREFERRED_MICROPHONE', '')
            pref_speaker = self.settings.get('PREFERRED_SPEAKER', '')
            
            if pref_mic:
                index = self.mic_combo.findText(pref_mic, Qt.MatchFlag.MatchContains)
                if index >= 0:
                    self.mic_combo.setCurrentIndex(index)
            
            if pref_speaker:
                index = self.speaker_combo.findText(pref_speaker, Qt.MatchFlag.MatchContains)
                if index >= 0:
                    self.speaker_combo.setCurrentIndex(index)
            
            # Update stereo mix info
            if stereo_mix_devices:
                self.stereo_mix_info.setText(f"✅ {len(stereo_mix_devices)} Stereo Mix device(s) detected")
                self.stereo_mix_info.setStyleSheet("color: green;")
            else:
                self.stereo_mix_info.setText("⚠️ No Stereo Mix found. Cannot record meeting audio.")
                self.stereo_mix_info.setStyleSheet("color: orange;")
            
        except Exception as e:
            self.mic_combo.addItem(f"Error: {str(e)}")
            self.speaker_combo.addItem(f"Error: {str(e)}")
    
    def on_engine_changed(self):
        """Handle engine selection change."""
        index = self.engine_combo.currentIndex()
        
        # Show/hide relevant groups
        self.whisper_group.setVisible(index in [0, 1])  # whisper engines
        self.azure_group.setVisible(index == 2)  # azure
    
    def on_model_changed(self):
        """Handle model selection change."""
        index = self.model_combo.currentIndex()
        
        info_texts = [
            "Tiny: Fast and lightweight. Good for testing or low-end hardware.",
            "Base: Recommended for most users. Good balance of speed and accuracy.",
            "Small: Better accuracy, requires more memory (4GB+ RAM recommended).",
            "Medium: Best accuracy, slow. Requires powerful hardware (8GB+ RAM, good CPU)."
        ]
        
        self.model_info.setText(info_texts[index])
    
    def show_stereo_mix_help(self):
        """Show help for enabling Stereo Mix."""
        help_text = """
<h3>How to Enable Stereo Mix on Windows:</h3>
<ol>
<li>Right-click the Speaker icon in system tray</li>
<li>Click "Sound settings" or "Sounds"</li>
<li>Go to "Recording" tab</li>
<li>Right-click in empty area → Show Disabled Devices</li>
<li>Find "Stereo Mix" → Right-click → Enable</li>
<li>Set as Default Device (optional)</li>
</ol>

<p><b>If Stereo Mix is not available:</b></p>
<ul>
<li>Update your audio driver</li>
<li>Use virtual audio cable (VB-Cable, Voicemeeter)</li>
<li>Run: <code>python audio_setup_helper.py</code> for detailed guide</li>
</ul>
        """
        
        QMessageBox.information(self, "Stereo Mix Setup Help", help_text)
    
    def browse_output_folder(self):
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self, 
            "Reset Settings",
            "Reset all settings to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.settings = {
                'TRANSCRIPTION_MODE': 'whisper',
                'WHISPER_MODEL': 'base',
                'WHISPER_LANGUAGE': 'en',
                'WHISPER_TASK': 'transcribe',
                'AZURE_SPEECH_KEY': '',
                'AZURE_SPEECH_REGION': '',
                'PREFERRED_MICROPHONE': '',
                'PREFERRED_SPEAKER': '',
                'SAMPLE_RATE': '16000',
                'CHANNELS': '2',
                'NUM_WORKERS': '4',
                'OUTPUT_DIR': 'transcripts',
            }
            self.populate_settings()
    
    def save_settings(self):
        """Save settings to .env file."""
        # Get current values from UI
        engine_index = self.engine_combo.currentIndex()
        mode_map = {0: 'whisper', 1: 'whisper', 2: 'azure', 3: 'web-speech'}
        mode = mode_map[engine_index]
        
        model_index = self.model_combo.currentIndex()
        model = ['tiny', 'base', 'small', 'medium'][model_index]
        
        language_text = self.language_combo.currentText()
        language = language_text.split('(')[0].strip() if '(' in language_text else language_text.split()[0]
        
        task = 'transcribe' if self.task_combo.currentIndex() == 0 else 'translate'
        
        sample_rate_text = self.sample_rate_combo.currentText()
        sample_rate = sample_rate_text.split()[0]
        
        channels = '1' if self.channels_combo.currentIndex() == 0 else '2'
        
        # Build settings dict
        new_settings = {
            'TRANSCRIPTION_MODE': mode,
            'WHISPER_MODEL': model,
            'WHISPER_LANGUAGE': language,
            'WHISPER_TASK': task,
            'AZURE_SPEECH_KEY': self.azure_key_edit.text(),
            'AZURE_SPEECH_REGION': self.azure_region_edit.text(),
            'PREFERRED_MICROPHONE': self.mic_combo.currentText(),
            'PREFERRED_SPEAKER': self.speaker_combo.currentText(),
            'SAMPLE_RATE': sample_rate,
            'CHANNELS': channels,
            'CHUNK_SIZE': '1024',
            'NUM_WORKERS': str(self.cpu_threads_spin.value()),
            'OUTPUT_DIR': self.output_edit.text(),
        }
        
        # Write to .env file
        try:
            with open('.env', 'w') as f:
                f.write("# Meeting Transcription Settings\n")
                f.write("# Auto-generated by Settings UI\n\n")
                
                for key, value in new_settings.items():
                    f.write(f"{key}={value}\n")
            
            QMessageBox.information(
                self,
                "Settings Saved",
                "Settings have been saved successfully!\n\nRestart the application for changes to take effect."
            )
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Settings",
                f"Failed to save settings:\n{str(e)}"
            )


def main():
    """Run settings dialog."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    dialog = SettingsDialog()
    result = dialog.exec()
    
    sys.exit(result)


if __name__ == "__main__":
    main()
