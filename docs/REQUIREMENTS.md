# Requirements Specification
## Meeting Transcription & MoM Generator

### Document Information
- **Version**: 1.2 (Updated for v0.0.2)
- **Date**: December 3, 2025
- **Project**: Meeting Transcription & MoM Generator
- **Platform**: Windows Desktop Application

---

## 1. Executive Summary

### 1.1 Purpose
This document defines the functional and non-functional requirements for the Meeting Transcription & MoM Generator application. The system enables users to record meetings (both microphone and system audio), transcribe speech to text using AI, and export formatted Minutes of Meeting (MoM) documents.

### 1.2 Scope
- **In Scope**: Desktop application for Windows, audio recording, speech transcription, export capabilities
- **Out of Scope**: Mobile applications, real-time collaboration, cloud storage (v1.0)

### 1.3 Stakeholders
- **Primary Users**: Professionals conducting meetings (Teams, Zoom, in-person)
- **Use Cases**: Meeting minutes, interview transcription, lecture notes, compliance documentation

---

## 2. Functional Requirements

### 2.1 Audio Capture (FR-AC)

#### FR-AC-01: Microphone Recording
- **Priority**: MUST HAVE
- **Description**: System shall capture audio from selected microphone input device
- **Acceptance Criteria**:
  - User can select microphone from available devices
  - Audio captured at 16 kHz, 16-bit mono
  - Recording indicator shows active status
  - Audio buffered continuously without gaps

#### FR-AC-02: System Audio Recording (Loopback)
- **Priority**: SHOULD HAVE
- **Description**: System shall capture audio from system output (speakers/headphones)
- **Acceptance Criteria**:
  - User can select loopback device (e.g., Stereo Mix)
  - System audio captured simultaneously with microphone
  - Audio format matches microphone settings
  - Clear indication when loopback not available

#### FR-AC-03: Dual-Channel Recording
- **Priority**: SHOULD HAVE
- **Description**: System shall record microphone and system audio simultaneously
- **Acceptance Criteria**:
  - Both channels recorded in parallel
  - Channels merged into single audio file
  - Volume balance maintained
  - No audio drift or desynchronization

#### FR-AC-04: Device Enumeration
- **Priority**: MUST HAVE
- **Description**: System shall list all available audio input devices
- **Acceptance Criteria**:
  - Microphone devices listed in dropdown
  - Loopback devices identified separately
  - Default devices pre-selected
  - Refresh capability for hot-plugged devices

#### FR-AC-05: Recording Controls
- **Priority**: MUST HAVE
- **Description**: System shall provide start/stop recording controls
- **Acceptance Criteria**:
  - Start button initiates recording
  - Stop button ends recording and saves file
  - Buttons disabled appropriately (no start while recording)
  - Visual feedback on recording state

#### FR-AC-06: Audio File Storage
- **Priority**: MUST HAVE
- **Description**: System shall save recordings as WAV files
- **Acceptance Criteria**:
  - Files saved to `recordings/` directory
  - Filename format: `recording_YYYYMMDD_HHMMSS.wav`
  - Automatic directory creation if missing
  - Error handling for disk full scenarios

---

### 2.2 Transcription (FR-TR)

#### FR-TR-01: Multi-Engine Support
- **Priority**: MUST HAVE
- **Description**: System shall support multiple transcription engines
- **Acceptance Criteria**:
  - Minimum 2 engines implemented (Whisper, one other)
  - User can select engine from dropdown
  - Engine availability checked before use
  - Clear error messages for unavailable engines

#### FR-TR-02: Whisper Integration (faster-whisper)
- **Priority**: MUST HAVE
- **Description**: System shall transcribe audio using faster-whisper (CTranslate2-optimized Whisper)
- **Performance**: 6-7x real-time speed with base model on CPU
- **Acceptance Criteria**:
  - Support model sizes: tiny, base, small, medium, large
  - Model selection configurable
  - Model cached after first download
  - Offline operation (no internet required)
  - Timestamp generation for segments
  - Confidence scores where available

#### FR-TR-03: Chrome Web Speech API
- **Priority**: SHOULD HAVE
- **Description**: System shall support real-time transcription via Chrome browser
- **Acceptance Criteria**:
  - WebSocket server on port 8765
  - HTML page generation for browser
  - Real-time transcript updates
  - Free to use (no API costs)

#### FR-TR-04: Azure Speech Service
- **Priority**: MAY HAVE
- **Description**: System shall optionally support Azure Speech transcription
- **Acceptance Criteria**:
  - Configurable via API key and region
  - High accuracy transcription
  - Error handling for API failures
  - Graceful degradation if not configured

#### FR-TR-05: Transcript Segmentation
- **Priority**: MUST HAVE
- **Description**: System shall break transcripts into timestamped segments
- **Acceptance Criteria**:
  - Each segment has start/end time
  - Segments display in chronological order
  - Timestamps formatted as `[XXs]` or `[MM:SS]`
  - Segments stored in structured format

#### FR-TR-06: Background Processing
- **Priority**: MUST HAVE
- **Description**: Transcription shall run in background thread
- **Acceptance Criteria**:
  - UI remains responsive during transcription
  - Progress indicator shown
  - User can cancel operation
  - Completion notification

---

### 2.3 User Interface (FR-UI)

#### FR-UI-01: Main Window
- **Priority**: MUST HAVE
- **Description**: System shall provide desktop GUI with all controls
- **Acceptance Criteria**:
  - Native Windows appearance
  - Minimum 800x600 resolution support
  - Resizable window
  - All features accessible without scrolling (at default size)

#### FR-UI-02: Device Selection
- **Priority**: MUST HAVE
- **Description**: Users shall select audio devices via dropdowns
- **Acceptance Criteria**:
  - Separate dropdowns for mic and speaker
  - Default devices pre-selected
  - Refresh button to reload devices
  - Clear labeling of device types

#### FR-UI-03: Recording Controls
- **Priority**: MUST HAVE
- **Description**: Large, clear buttons for recording operations
- **Acceptance Criteria**:
  - Start Recording (green)
  - Stop Recording (red)
  - Transcribe (blue)
  - Buttons appropriately enabled/disabled

#### FR-UI-04: Transcript Display
- **Priority**: MUST HAVE
- **Description**: System shall display transcript in readable text area
- **Acceptance Criteria**:
  - Scrollable text area
  - Timestamps visible for each segment
  - Auto-scroll to latest content
  - Copy/paste support

#### FR-UI-05: Engine Selection
- **Priority**: MUST HAVE
- **Description**: Users shall select transcription engine
- **Acceptance Criteria**:
  - Dropdown with available engines
  - Display engine names clearly
  - Show unavailable engines as disabled
  - Persist selection across sessions

#### FR-UI-06: Status Bar
- **Priority**: SHOULD HAVE
- **Description**: System shall show operation status
- **Acceptance Criteria**:
  - Current operation displayed
  - Recording duration shown
  - Error messages visible
  - Progress indicator for long operations

#### FR-UI-07: Export Controls
- **Priority**: MUST HAVE
- **Description**: Users shall export transcripts via buttons
- **Acceptance Criteria**:
  - Save as TXT button
  - Save as Markdown button
  - Clear transcript button
  - Buttons enabled only when transcript available

---

### 2.4 Export and Storage (FR-EX)

#### FR-EX-01: Plain Text Export
- **Priority**: MUST HAVE
- **Description**: System shall export transcripts as .txt files
- **Acceptance Criteria**:
  - File dialog for save location
  - Default filename: `transcript_YYYYMMDD_HHMMSS.txt`
  - Format: `[timestamp] text\n`
  - UTF-8 encoding

#### FR-EX-02: Markdown Export
- **Priority**: SHOULD HAVE
- **Description**: System shall export transcripts as Markdown files
- **Acceptance Criteria**:
  - File saved as .md
  - Formatted with headers
  - Timestamps in bold or code blocks
  - Meeting metadata (date, duration)

#### FR-EX-03: Directory Structure
- **Priority**: MUST HAVE
- **Description**: System shall organize files in directories
- **Acceptance Criteria**:
  - `recordings/` for audio files
  - `transcripts/` for exported documents
  - `models/` for AI models
  - Auto-creation of missing directories

---

### 2.5 Recording Management (FR-RM)

#### FR-RM-01: View Recordings
- **Priority**: SHOULD HAVE
- **Description**: System shall list all recorded audio files
- **Acceptance Criteria**:
  - Display filename, size, date
  - Sorted by date (newest first)
  - Open folder button
  - Play selected recording (via default app)

#### FR-RM-02: Delete Recordings
- **Priority**: MUST HAVE (Compliance)
- **Description**: Users shall permanently delete recording files
- **Acceptance Criteria**:
  - Delete all recordings button
  - Delete individual recording option
  - Confirmation dialog with warning
  - Cannot be undone message
  - Count of files to be deleted

#### FR-RM-03: Compliance Support
- **Priority**: MUST HAVE
- **Description**: System shall support data privacy compliance
- **Acceptance Criteria**:
  - Easy deletion of all recordings
  - No cloud upload without consent
  - Local storage only (default)
  - Clear data retention messaging

---

### 2.6 Configuration (FR-CF)

#### FR-CF-01: Environment Configuration
- **Priority**: MUST HAVE
- **Description**: System shall load settings from .env file
- **Acceptance Criteria**:
  - `.env` file for user settings
  - `.env.example` as template
  - Settings: transcription mode, model size, API keys
  - Fallback to defaults if missing

#### FR-CF-02: Preferred Devices
- **Priority**: SHOULD HAVE
- **Description**: Users shall configure preferred audio devices
- **Acceptance Criteria**:
  - `PREFERRED_MICROPHONE` setting
  - `PREFERRED_SPEAKER` setting
  - Auto-selection on startup
  - Fallback to system default

#### FR-CF-03: Whisper Model Selection
- **Priority**: MUST HAVE
- **Description**: Users shall choose Whisper model size
- **Acceptance Criteria**:
  - `WHISPER_MODEL` in config
  - Options: tiny, base, small, medium, large
  - Balance between speed and accuracy
  - Model downloaded on first use

---

## 3. Non-Functional Requirements

### 3.1 Performance (NFR-PF)

#### NFR-PF-01: Audio Latency
- **Priority**: MUST HAVE
- **Requirement**: Audio capture latency < 100ms
- **Measurement**: Time between speech and buffer write

#### NFR-PF-02: Transcription Speed
- **Priority**: SHOULD HAVE
- **Requirement**: Whisper tiny model transcribes at ≥1x realtime
- **Measurement**: Processing time / audio duration

#### NFR-PF-03: UI Responsiveness
- **Priority**: MUST HAVE
- **Requirement**: UI responds within 100ms to user input
- **Measurement**: Button click to visual feedback

#### NFR-PF-04: Memory Usage
- **Priority**: SHOULD HAVE
- **Requirement**: 
  - Base app: < 500 MB RAM
  - With Whisper tiny: < 2 GB RAM
  - With Whisper base: < 3 GB RAM

#### NFR-PF-05: Startup Time
- **Priority**: SHOULD HAVE
- **Requirement**: Application launches in < 3 seconds
- **Measurement**: main.py execution to window display

---

### 3.2 Usability (NFR-US)

#### NFR-US-01: Learning Curve
- **Priority**: MUST HAVE
- **Requirement**: First-time users can record and transcribe within 2 minutes
- **Measurement**: User testing with new users

#### NFR-US-02: Visual Clarity
- **Priority**: MUST HAVE
- **Requirement**: All text readable at 125% Windows scaling
- **Measurement**: Visual inspection and accessibility testing

#### NFR-US-03: Error Messages
- **Priority**: MUST HAVE
- **Requirement**: Error messages are clear, actionable, and non-technical
- **Measurement**: User comprehension testing

#### NFR-US-04: Documentation
- **Priority**: MUST HAVE
- **Requirement**: README and QUICKSTART guide included
- **Measurement**: Documentation completeness review

---

### 3.3 Reliability (NFR-RL)

#### NFR-RL-01: Crash Prevention
- **Priority**: MUST HAVE
- **Requirement**: Application handles errors gracefully without crashing
- **Measurement**: Exception handling coverage

#### NFR-RL-02: Data Integrity
- **Priority**: MUST HAVE
- **Requirement**: No audio data loss during recording
- **Measurement**: Audio file completeness verification

#### NFR-RL-03: Transcription Accuracy
- **Priority**: SHOULD HAVE
- **Requirement**: Whisper base model achieves >90% word accuracy (English)
- **Measurement**: WER (Word Error Rate) on test dataset

---

### 3.4 Compatibility (NFR-CP)

#### NFR-CP-01: Windows Support
- **Priority**: MUST HAVE
- **Requirement**: Works on Windows 10 (64-bit) and Windows 11
- **Measurement**: Testing on both OS versions

#### NFR-CP-02: Python Version
- **Priority**: MUST HAVE
- **Requirement**: Compatible with Python 3.8, 3.9, 3.10, 3.11, 3.12
- **Measurement**: CI testing across versions

#### NFR-CP-03: Audio Devices
- **Priority**: MUST HAVE
- **Requirement**: Supports standard USB and 3.5mm audio devices
- **Measurement**: Testing with multiple device types

---

### 3.5 Security (NFR-SC)

#### NFR-SC-01: API Key Storage
- **Priority**: MUST HAVE
- **Requirement**: API keys stored in .env file, not in code
- **Measurement**: Code review, no secrets in repository

#### NFR-SC-02: Local Processing
- **Priority**: SHOULD HAVE
- **Requirement**: Whisper engine processes audio locally (no network)
- **Measurement**: Network traffic monitoring during transcription

#### NFR-SC-03: File Permissions
- **Priority**: SHOULD HAVE
- **Requirement**: Recordings saved with user-only access permissions
- **Measurement**: File system permission verification

---

### 3.6 Maintainability (NFR-MT)

#### NFR-MT-01: Code Quality
- **Priority**: SHOULD HAVE
- **Requirement**: Code follows PEP 8 style guidelines
- **Measurement**: Linting with flake8 or ruff

#### NFR-MT-02: Documentation
- **Priority**: MUST HAVE
- **Requirement**: All modules have docstrings
- **Measurement**: Documentation coverage check

#### NFR-MT-03: Modularity
- **Priority**: MUST HAVE
- **Requirement**: Audio, transcription, UI separated into modules
- **Measurement**: Architectural review

---

## 4. System Constraints

### 4.1 Technical Constraints
- **TC-01**: Windows WASAPI required for system audio
- **TC-02**: GPU optional (CUDA) for Whisper acceleration
- **TC-03**: Internet required for model downloads and Azure/Chrome engines
- **TC-04**: Minimum 2GB free disk space for models

### 4.2 Business Constraints
- **BC-01**: Application must be free to use
- **BC-02**: Azure Speech is optional (user provides API key)
- **BC-03**: No telemetry or data collection

### 4.3 Regulatory Constraints
- **RC-01**: GDPR compliance: User controls all data
- **RC-02**: No audio uploaded to cloud without explicit consent
- **RC-03**: Easy data deletion for right to be forgotten

---

## 5. User Stories

### 5.1 Meeting Recording

**US-01**: As a remote worker, I want to record my Teams meetings so I can review discussions later.

**US-02**: As a manager, I want to capture both my voice and the meeting audio so I have complete context.

**US-03**: As a student, I want to record lectures so I can study from them.

### 5.2 Transcription

**US-04**: As a busy professional, I want automatic transcription so I don't have to type meeting notes.

**US-05**: As a non-native English speaker, I want accurate transcripts so I can review conversations at my own pace.

**US-06**: As a researcher, I want timestamped transcripts so I can reference specific moments.

### 5.3 Compliance

**US-07**: As a compliance officer, I want to easily delete all recordings so we meet data retention policies.

**US-08**: As a privacy-conscious user, I want local processing so my data never leaves my computer.

---

## 6. Acceptance Criteria

### 6.1 Minimum Viable Product (MVP)
- ✅ Record audio from microphone
- ✅ Transcribe with Whisper (base model)
- ✅ Export as TXT and Markdown
- ✅ Delete recordings for compliance
- ✅ Windows desktop GUI

### 6.2 Version 1.0 Complete
- ✅ All MVP features
- ✅ System audio (loopback) recording
- ✅ Multiple transcription engines (Whisper, Chrome, Azure)
- ✅ View and manage recordings
- ✅ Preferred device configuration
- ✅ Comprehensive documentation

---

## 7. Dependencies

### 7.1 External Services
- **Optional**: Azure Speech API (user-provided)
- **Optional**: Chrome browser for Web Speech API

### 7.2 Third-Party Libraries
- PyQt6 (GUI)
- sounddevice (audio)
- openai-whisper (transcription)
- torch (ML backend)

### 7.3 System Dependencies
- Windows Audio (WASAPI)
- Python 3.8+ runtime
- .NET Framework (for Azure SDK)

---

## 8. Assumptions

1. Users have basic computer literacy
2. Microphone access permissions granted
3. Sufficient disk space available
4. Windows audio drivers properly installed
5. Internet available for initial setup

---

## 9. Out of Scope (Version 1.0)

### Future Considerations
- ❌ Real-time transcription (live streaming)
- ❌ Speaker diarization (who said what)
- ❌ Multi-language support beyond English
- ❌ Mobile applications (iOS/Android)
- ❌ Cloud sync and storage
- ❌ Collaborative editing
- ❌ AI summarization of transcripts
- ❌ Integration with Slack/Teams/Zoom APIs
- ❌ Video recording
- ❌ Screen capture

---

## 10. Traceability Matrix

| Requirement | Priority | Implemented | Tested | Notes                     |
| ----------- | -------- | ----------- | ------ | ------------------------- |
| FR-AC-01    | MUST     | ✅           | ✅      | Microphone recording      |
| FR-AC-02    | SHOULD   | ✅           | ✅      | System audio (Stereo Mix) |
| FR-AC-03    | SHOULD   | ✅           | ✅      | Dual-channel merge        |
| FR-TR-01    | MUST     | ✅           | ✅      | 3 engines supported       |
| FR-TR-02    | MUST     | ✅           | ✅      | Whisper integration       |
| FR-UI-01    | MUST     | ✅           | ✅      | PyQt6 GUI                 |
| FR-EX-01    | MUST     | ✅           | ✅      | TXT export                |
| FR-EX-02    | SHOULD   | ✅           | ✅      | Markdown export           |
| FR-RM-01    | SHOULD   | ✅           | ⏳      | View recordings           |
| FR-RM-02    | MUST     | ✅           | ⏳      | Delete recordings         |

---

## Revision History

| Version | Date       | Author | Changes                            |
| ------- | ---------- | ------ | ---------------------------------- |
| 1.0     | 2025-12-03 | System | Initial requirements specification |
