"""
Audio Capture Module
Handles microphone and system audio (loopback) recording on Windows using WASAPI
"""
import sounddevice as sd
import numpy as np
import wave
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import config


class AudioDevice:
    """Represents an audio input/output device"""
    def __init__(self, index: int, name: str, channels: int, sample_rate: float, is_loopback: bool = False):
        self.index = index
        self.name = name
        self.channels = channels
        self.sample_rate = sample_rate
        self.is_loopback = is_loopback

    def __repr__(self):
        device_type = "Loopback" if self.is_loopback else "Input"
        return f"{device_type}: {self.name} ({self.channels}ch, {self.sample_rate}Hz)"


class AudioCapture:
    """
    Manages audio recording from microphone and/or system audio (loopback).
    Supports Windows WASAPI for system audio capture.
    """
    
    def __init__(self):
        self.is_recording = False
        self.mic_stream = None
        self.speaker_stream = None
        self.mic_data = []
        self.speaker_data = []
        self.sample_rate = config.SAMPLE_RATE
        self.channels = config.CHANNELS
        self.recording_thread = None
        self.lock = threading.Lock()
        
        # Callbacks for real-time processing
        self.on_audio_chunk: Optional[Callable] = None
        
    def list_devices(self) -> tuple[list[AudioDevice], list[AudioDevice]]:
        """
        List available audio input devices and loopback devices.
        Returns: (input_devices, loopback_devices)
        """
        input_devices = []
        loopback_devices = []
        
        try:
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                # Check if device supports input
                if device['max_input_channels'] > 0:
                    audio_dev = AudioDevice(
                        index=idx,
                        name=device['name'],
                        channels=device['max_input_channels'],
                        sample_rate=device['default_samplerate'],
                        is_loopback=False
                    )
                    input_devices.append(audio_dev)
                
                # Windows WASAPI loopback devices typically have "Stereo Mix" or similar in name
                # or we can use hostapi to identify WASAPI devices
                if 'Windows WASAPI' in str(device.get('hostapi', '')):
                    if device['max_output_channels'] > 0:
                        # Try to find loopback capability
                        loopback_dev = AudioDevice(
                            index=idx,
                            name=device['name'] + " (Loopback)",
                            channels=device['max_output_channels'],
                            sample_rate=device['default_samplerate'],
                            is_loopback=True
                        )
                        loopback_devices.append(loopback_dev)
                        
        except Exception as e:
            print(f"Error listing devices: {e}")
            
        return input_devices, loopback_devices
    
    def get_default_devices(self) -> tuple[Optional[AudioDevice], Optional[AudioDevice]]:
        """Get default microphone and speaker devices"""
        try:
            # Check for preferred microphone in config
            preferred_mic_name = config.PREFERRED_MICROPHONE
            mic_device = None
            
            if preferred_mic_name:
                # Try to find preferred microphone
                input_devices, _ = self.list_devices()
                for dev in input_devices:
                    if preferred_mic_name.lower() in dev.name.lower():
                        mic_device = dev
                        print(f"Using preferred microphone: {dev.name}")
                        break
            
            # Fallback to system default if preferred not found
            if not mic_device:
                default_input = sd.query_devices(kind='input')
                mic_device = AudioDevice(
                    index=sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device,
                    name=default_input['name'],
                    channels=min(self.channels, default_input['max_input_channels']),
                    sample_rate=default_input['default_samplerate'],
                    is_loopback=False
                )
            
            # For Windows, try to find preferred speaker or WASAPI loopback device
            speaker_device = None
            preferred_speaker_name = config.PREFERRED_SPEAKER
            devices = sd.query_devices()
            input_devices, _ = self.list_devices()
            
            # First try preferred speaker
            if preferred_speaker_name:
                for dev in input_devices:
                    if preferred_speaker_name.lower() in dev.name.lower():
                        speaker_device = dev
                        print(f"Using preferred speaker/loopback: {dev.name}")
                        break
            
            # Fallback: Look for WASAPI output devices that can be used for loopback
            if not speaker_device:
                for idx, device in enumerate(devices):
                    if 'WASAPI' in str(device) and device['max_output_channels'] > 0:
                        # This is a potential loopback device
                        speaker_device = AudioDevice(
                            index=idx,
                            name=device['name'],
                            channels=min(self.channels, device['max_output_channels']),
                            sample_rate=device['default_samplerate'],
                            is_loopback=True
                        )
                        break
            
            return mic_device, speaker_device
            
        except Exception as e:
            print(f"Error getting default devices: {e}")
            return None, None
    
    def start_recording(self, mic_device: Optional[AudioDevice] = None, 
                       speaker_device: Optional[AudioDevice] = None,
                       record_mic: bool = True,
                       record_speaker: bool = True) -> bool:
        """
        Start recording from selected devices.
        
        Args:
            mic_device: Microphone device (None for default)
            speaker_device: Speaker/loopback device (None for default)
            record_mic: Whether to record microphone
            record_speaker: Whether to record system audio
            
        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            print("Already recording")
            return False
        
        # Reset buffers
        self.mic_data = []
        self.speaker_data = []
        
        try:
            # Get default devices if not specified
            if mic_device is None or speaker_device is None:
                default_mic, default_speaker = self.get_default_devices()
                mic_device = mic_device or default_mic
                speaker_device = speaker_device or default_speaker
            
            self.is_recording = True
            
            # Start microphone recording
            if record_mic and mic_device:
                def mic_callback(indata, frames, time, status):
                    if status:
                        print(f"Mic status: {status}")
                    with self.lock:
                        self.mic_data.append(indata.copy())
                    if self.on_audio_chunk:
                        self.on_audio_chunk('mic', indata.copy())
                
                self.mic_stream = sd.InputStream(
                    device=mic_device.index,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=mic_callback,
                    dtype='int16'
                )
                self.mic_stream.start()
                print(f"✓ Recording from microphone: {mic_device.name}")
            
            # Start speaker recording (loopback)
            if record_speaker and speaker_device:
                def speaker_callback(indata, frames, time, status):
                    if status:
                        print(f"Speaker status: {status}")
                    with self.lock:
                        self.speaker_data.append(indata.copy())
                    if self.on_audio_chunk:
                        self.on_audio_chunk('speaker', indata.copy())
                
                # For loopback, we need to use input stream with loopback flag
                # This is Windows-specific WASAPI feature
                self.speaker_stream = sd.InputStream(
                    device=speaker_device.index,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=speaker_callback,
                    dtype='int16'
                )
                self.speaker_stream.start()
                print(f"✓ Recording system audio: {speaker_device.name}")
            
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            self.stop_recording()
            return False
    
    def stop_recording(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Stop recording and return captured audio data.
        
        Returns:
            (mic_audio, speaker_audio) as numpy arrays, or None if not recorded
        """
        if not self.is_recording:
            return None, None
        
        self.is_recording = False
        
        # Stop streams
        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
            self.mic_stream = None
        
        if self.speaker_stream:
            self.speaker_stream.stop()
            self.speaker_stream.close()
            self.speaker_stream = None
        
        # Concatenate recorded data
        with self.lock:
            mic_audio = np.concatenate(self.mic_data, axis=0) if self.mic_data else None
            speaker_audio = np.concatenate(self.speaker_data, axis=0) if self.speaker_data else None
        
        print(f"✓ Recording stopped")
        if mic_audio is not None:
            print(f"  Microphone: {len(mic_audio) / self.sample_rate:.2f} seconds")
        if speaker_audio is not None:
            print(f"  Speaker: {len(speaker_audio) / self.sample_rate:.2f} seconds")
        
        return mic_audio, speaker_audio
    
    def save_audio(self, audio_data: np.ndarray, filename: str, 
                   output_dir: Path = config.RECORDINGS_DIR) -> Path:
        """
        Save audio data to WAV file.
        
        Args:
            audio_data: Audio data as numpy array
            filename: Output filename (without extension)
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = output_dir / f"{filename}.wav"
        
        try:
            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            print(f"✓ Audio saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error saving audio: {e}")
            raise
    
    def merge_audio_channels(self, mic_audio: Optional[np.ndarray], 
                            speaker_audio: Optional[np.ndarray]) -> np.ndarray:
        """
        Merge microphone and speaker audio into a single stream.
        Uses mixing/averaging to combine both sources.
        
        Returns:
            Merged audio as numpy array
        """
        if mic_audio is None and speaker_audio is None:
            raise ValueError("No audio data to merge")
        
        if mic_audio is None:
            return speaker_audio
        
        if speaker_audio is None:
            return mic_audio
        
        # Ensure same length (pad shorter one with zeros)
        max_len = max(len(mic_audio), len(speaker_audio))
        
        if len(mic_audio) < max_len:
            mic_audio = np.pad(mic_audio, ((0, max_len - len(mic_audio)), (0, 0)), mode='constant')
        
        if len(speaker_audio) < max_len:
            speaker_audio = np.pad(speaker_audio, ((0, max_len - len(speaker_audio)), (0, 0)), mode='constant')
        
        # Mix both channels (simple average)
        # Convert to float to prevent overflow, then back to int16
        merged = (mic_audio.astype(np.float32) + speaker_audio.astype(np.float32)) / 2
        merged = merged.astype(np.int16)
        
        return merged


# Test functionality
if __name__ == "__main__":
    print("=== Audio Capture Module Test ===\n")
    
    capture = AudioCapture()
    
    # List available devices
    print("Available Input Devices:")
    input_devs, loopback_devs = capture.list_devices()
    for dev in input_devs:
        print(f"  {dev}")
    
    print("\nAvailable Loopback Devices:")
    for dev in loopback_devs:
        print(f"  {dev}")
    
    # Get default devices
    print("\nDefault Devices:")
    mic, speaker = capture.get_default_devices()
    print(f"  Microphone: {mic}")
    print(f"  Speaker: {speaker}")
    
    # Test recording (uncomment to test)
    # print("\nStarting 5-second test recording...")
    # capture.start_recording(record_mic=True, record_speaker=False)
    # import time
    # time.sleep(5)
    # mic_audio, speaker_audio = capture.stop_recording()
    # 
    # if mic_audio is not None:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     capture.save_audio(mic_audio, f"test_recording_{timestamp}")
