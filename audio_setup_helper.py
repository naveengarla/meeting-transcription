"""
Audio Setup Helper
Detects audio capabilities and provides guided setup instructions.
"""

import sys
import platform


def detect_audio_capabilities():
    """Detect available audio devices and capabilities."""
    print("=" * 70)
    print("Audio Device Detection")
    print("=" * 70)
    
    try:
        import sounddevice as sd
    except ImportError:
        print("\n❌ sounddevice not installed")
        print("   Run: pip install sounddevice")
        return None
    
    devices = sd.query_devices()
    
    capabilities = {
        'input_devices': [],
        'output_devices': [],
        'has_stereo_mix': False,
        'has_microphone': False,
        'stereo_mix_devices': [],
        'microphone_devices': [],
        'recommendations': [],
        'warnings': [],
    }
    
    # Analyze devices
    for idx, device in enumerate(devices):
        # Input devices
        if device['max_input_channels'] > 0:
            device_info = {
                'index': idx,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate'],
            }
            capabilities['input_devices'].append(device_info)
            
            # Check for stereo mix / loopback
            name_lower = device['name'].lower()
            if any(keyword in name_lower for keyword in 
                   ['stereo mix', 'what u hear', 'wave out mix', 'loopback', 'what you hear']):
                capabilities['has_stereo_mix'] = True
                capabilities['stereo_mix_devices'].append(device_info)
            
            # Check for microphone
            elif any(keyword in name_lower for keyword in 
                     ['microphone', 'mic', 'array']):
                capabilities['has_microphone'] = True
                capabilities['microphone_devices'].append(device_info)
        
        # Output devices
        if device['max_output_channels'] > 0:
            device_info = {
                'index': idx,
                'name': device['name'],
                'channels': device['max_output_channels'],
                'sample_rate': device['default_samplerate'],
            }
            capabilities['output_devices'].append(device_info)
    
    # Generate recommendations and warnings
    if not capabilities['has_microphone']:
        capabilities['warnings'].append("⚠️  No microphone detected - cannot record your voice")
    
    if not capabilities['has_stereo_mix']:
        capabilities['warnings'].append("⚠️  No Stereo Mix detected - cannot record system audio (Teams, Zoom, etc.)")
        capabilities['recommendations'].append("Enable Stereo Mix or install virtual audio cable")
    else:
        capabilities['recommendations'].append("✅ Stereo Mix available - can record meeting audio")
    
    return capabilities


def print_audio_report(capabilities):
    """Print detailed audio capabilities report."""
    if not capabilities:
        return
    
    print("\n📊 AUDIO DEVICE SUMMARY")
    print("-" * 70)
    print(f"Total Input Devices:  {len(capabilities['input_devices'])}")
    print(f"Total Output Devices: {len(capabilities['output_devices'])}")
    print(f"Microphones Found:    {len(capabilities['microphone_devices'])}")
    print(f"Stereo Mix Available: {'✅ Yes' if capabilities['has_stereo_mix'] else '❌ No'}")
    
    # Microphones
    if capabilities['microphone_devices']:
        print("\n🎤 MICROPHONES")
        print("-" * 70)
        for dev in capabilities['microphone_devices']:
            print(f"[{dev['index']}] {dev['name']}")
            print(f"    {dev['channels']} channels @ {dev['sample_rate']:.0f} Hz")
    
    # Stereo Mix / Loopback
    if capabilities['stereo_mix_devices']:
        print("\n🔊 STEREO MIX / LOOPBACK DEVICES")
        print("-" * 70)
        for dev in capabilities['stereo_mix_devices']:
            print(f"[{dev['index']}] {dev['name']}")
            print(f"    {dev['channels']} channels @ {dev['sample_rate']:.0f} Hz")
    
    # Warnings
    if capabilities['warnings']:
        print("\n⚠️  WARNINGS")
        print("-" * 70)
        for warning in capabilities['warnings']:
            print(f"   {warning}")
    
    # Recommendations
    if capabilities['recommendations']:
        print("\n💡 RECOMMENDATIONS")
        print("-" * 70)
        for rec in capabilities['recommendations']:
            print(f"   {rec}")


def show_stereo_mix_setup_guide():
    """Show step-by-step guide to enable Stereo Mix on Windows."""
    print("\n" + "=" * 70)
    print("HOW TO ENABLE STEREO MIX (Windows)")
    print("=" * 70)
    
    print("""
📋 STEP-BY-STEP INSTRUCTIONS:

1. Right-click the Speaker icon in the system tray (bottom-right corner)
   
2. Click "Sound settings" or "Sounds"

3. Scroll down and click "More sound settings" or go to "Recording" tab

4. In the Recording tab:
   • Right-click in an empty area
   • Check "Show Disabled Devices"
   • Check "Show Disconnected Devices"

5. Look for "Stereo Mix" in the list
   
6. If you see "Stereo Mix":
   • Right-click it
   • Click "Enable"
   • Right-click again
   • Click "Set as Default Device" (optional)
   • Click "OK"

7. If you DON'T see "Stereo Mix":
   ⚠️  Your audio driver may not support it. Options:
   
   a) Update audio driver:
      • Open Device Manager (Win + X → Device Manager)
      • Expand "Sound, video and game controllers"
      • Right-click your audio device (e.g., Realtek, Intel)
      • Click "Update driver"
      • Restart computer
   
   b) Install virtual audio cable (alternative):
      • VB-Audio Virtual Cable: https://vb-audio.com/Cable/
      • Voicemeeter: https://vb-audio.com/Voicemeeter/
      
      These create virtual audio devices that can capture system audio.

📝 NOTE: Some audio drivers (especially on laptops) may not support Stereo Mix.
         In that case, virtual audio cable software is recommended.
""")


def show_virtual_audio_cable_guide():
    """Show guide for installing virtual audio cable alternatives."""
    print("\n" + "=" * 70)
    print("VIRTUAL AUDIO CABLE ALTERNATIVES")
    print("=" * 70)
    
    print("""
If your system doesn't have Stereo Mix, use virtual audio cable software:

📦 OPTION 1: VB-Audio Virtual Cable (FREE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Download: https://vb-audio.com/Cable/

Features:
• Simple virtual audio device
• Captures system audio
• Lightweight and stable
• Works with all apps

Setup:
1. Download and install VB-CABLE Driver
2. Restart computer
3. Set "CABLE Output" as default playback device in Windows Sound settings
4. Set "CABLE Input" as recording device in this app
5. Audio from all apps will be captured


📦 OPTION 2: Voicemeeter (FREE - More Features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Download: https://vb-audio.com/Voicemeeter/

Features:
• Virtual audio mixer
• Mix microphone + system audio
• Multiple virtual devices
• Advanced routing options

Setup:
1. Download and install Voicemeeter
2. Run Voicemeeter application
3. Set "Voicemeeter Input" as default playback device
4. In Voicemeeter, route to your speakers/headphones
5. Use "Voicemeeter Output" as recording device in this app


📦 OPTION 3: OBS Virtual Camera Audio (FREE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Download: https://obsproject.com/

Features:
• Full screen recording suite
• Virtual audio device included
• Can also capture video if needed

Setup:
1. Install OBS Studio
2. Add "Audio Output Capture" source
3. Start Virtual Camera (includes audio)
4. Use OBS virtual device in this app


⚠️  IMPORTANT NOTES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• All solutions require administrator privileges to install
• You may need to restart your computer after installation
• Configure default audio devices in Windows Sound settings
• Test audio playback to ensure you can still hear audio

🎯 RECOMMENDED: Start with VB-Audio Virtual Cable (simplest solution)
""")


def test_audio_recording(device_index=None, duration=3):
    """Test audio recording from specified device."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("❌ Required packages not installed")
        return False
    
    try:
        print(f"\n🎤 Testing audio recording...")
        print(f"Duration: {duration} seconds")
        
        if device_index is not None:
            device = sd.query_devices(device_index)
            print(f"Device: {device['name']}")
            print(f"\n⏺️  Recording... (speak or play audio)")
        else:
            print(f"\n⏺️  Recording from default device...")
        
        # Record audio
        sample_rate = 16000
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_index,
            dtype='float32'
        )
        sd.wait()
        
        # Analyze recording
        max_amplitude = np.max(np.abs(recording))
        rms_level = np.sqrt(np.mean(recording**2))
        
        print(f"\n✅ Recording complete!")
        print(f"   Max amplitude: {max_amplitude:.4f}")
        print(f"   RMS level:     {rms_level:.4f}")
        
        if max_amplitude < 0.001:
            print("\n⚠️  WARNING: Very low audio level detected")
            print("   Possible issues:")
            print("   • Wrong device selected")
            print("   • Device is muted")
            print("   • No audio playing during test")
            return False
        elif max_amplitude < 0.01:
            print("\n⚠️  Audio level is low but detectable")
            print("   Consider increasing volume")
            return True
        else:
            print(f"\n✅ Good audio level detected!")
            return True
            
    except Exception as e:
        print(f"\n❌ Recording test failed: {str(e)}")
        return False


def interactive_audio_setup():
    """Interactive audio device setup wizard."""
    print("\n" + "=" * 70)
    print("INTERACTIVE AUDIO SETUP WIZARD")
    print("=" * 70)
    
    # Detect capabilities
    capabilities = detect_audio_capabilities()
    if not capabilities:
        return None
    
    print_audio_report(capabilities)
    
    # Check for Stereo Mix
    if not capabilities['has_stereo_mix']:
        print("\n" + "=" * 70)
        print("⚠️  STEREO MIX NOT DETECTED")
        print("=" * 70)
        print("\nYou won't be able to record meeting audio (Teams, Zoom, etc.)")
        print("without Stereo Mix or a virtual audio cable.\n")
        
        response = input("Would you like to see setup instructions? (y/n): ").strip().lower()
        if response == 'y':
            show_stereo_mix_setup_guide()
            
            response2 = input("\nWould you like to see virtual audio cable alternatives? (y/n): ").strip().lower()
            if response2 == 'y':
                show_virtual_audio_cable_guide()
    
    # Device selection
    print("\n" + "=" * 70)
    print("DEVICE SELECTION")
    print("=" * 70)
    
    selected_devices = {}
    
    # Select microphone
    if capabilities['microphone_devices']:
        print("\n🎤 Available Microphones:")
        for i, dev in enumerate(capabilities['microphone_devices']):
            print(f"   {i+1}. {dev['name']} (index {dev['index']})")
        
        if len(capabilities['microphone_devices']) == 1:
            selected_devices['microphone'] = capabilities['microphone_devices'][0]
            print(f"\n✅ Auto-selected: {selected_devices['microphone']['name']}")
        else:
            try:
                choice = int(input(f"\nSelect microphone (1-{len(capabilities['microphone_devices'])}): "))
                if 1 <= choice <= len(capabilities['microphone_devices']):
                    selected_devices['microphone'] = capabilities['microphone_devices'][choice-1]
                    print(f"✅ Selected: {selected_devices['microphone']['name']}")
            except:
                selected_devices['microphone'] = capabilities['microphone_devices'][0]
                print(f"⚠️  Invalid choice, using first device: {selected_devices['microphone']['name']}")
    
    # Select Stereo Mix
    if capabilities['stereo_mix_devices']:
        print("\n🔊 Available Stereo Mix Devices:")
        for i, dev in enumerate(capabilities['stereo_mix_devices']):
            print(f"   {i+1}. {dev['name']} (index {dev['index']})")
        
        if len(capabilities['stereo_mix_devices']) == 1:
            selected_devices['stereo_mix'] = capabilities['stereo_mix_devices'][0]
            print(f"\n✅ Auto-selected: {selected_devices['stereo_mix']['name']}")
        else:
            try:
                choice = int(input(f"\nSelect stereo mix (1-{len(capabilities['stereo_mix_devices'])}): "))
                if 1 <= choice <= len(capabilities['stereo_mix_devices']):
                    selected_devices['stereo_mix'] = capabilities['stereo_mix_devices'][choice-1]
                    print(f"✅ Selected: {selected_devices['stereo_mix']['name']}")
            except:
                selected_devices['stereo_mix'] = capabilities['stereo_mix_devices'][0]
                print(f"⚠️  Invalid choice, using first device: {selected_devices['stereo_mix']['name']}")
    
    # Test selected devices
    if selected_devices:
        print("\n" + "=" * 70)
        print("DEVICE TESTING")
        print("=" * 70)
        
        response = input("\nWould you like to test the selected devices? (y/n): ").strip().lower()
        if response == 'y':
            if 'microphone' in selected_devices:
                print(f"\n🎤 Testing microphone: {selected_devices['microphone']['name']}")
                test_audio_recording(selected_devices['microphone']['index'], duration=3)
            
            if 'stereo_mix' in selected_devices:
                print(f"\n🔊 Testing stereo mix: {selected_devices['stereo_mix']['name']}")
                print("   (Play some audio on your computer)")
                input("   Press Enter when ready to record...")
                test_audio_recording(selected_devices['stereo_mix']['index'], duration=3)
    
    return selected_devices


if __name__ == "__main__":
    print("\n🎧 Meeting Transcription - Audio Setup Helper\n")
    
    try:
        # Run interactive setup
        selected = interactive_audio_setup()
        
        if selected:
            print("\n" + "=" * 70)
            print("✅ SETUP COMPLETE")
            print("=" * 70)
            print("\nSelected Devices:")
            for device_type, device in selected.items():
                print(f"   {device_type}: {device['name']} (index {device['index']})")
            
            print("\n💡 Update your .env file with these device names:")
            if 'microphone' in selected:
                print(f"   PREFERRED_MICROPHONE={selected['microphone']['name']}")
            if 'stereo_mix' in selected:
                print(f"   PREFERRED_SPEAKER={selected['stereo_mix']['name']}")
            
            print("\n🚀 You're ready to start transcribing!")
        else:
            print("\n⚠️  Setup incomplete. Run this script again to configure audio devices.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
