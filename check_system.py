"""
System Prerequisites Checker
Validates system compatibility before installation.
"""

import sys
import platform
import subprocess
from pathlib import Path


def check_prerequisites():
    """Check all system prerequisites and return issues/warnings."""
    print("=" * 60)
    print("System Prerequisites Check")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # 1. Operating System Check
    print("\n[1/8] Checking operating system...")
    if platform.system() != "Windows":
        issues.append("❌ This application only runs on Windows")
        print(f"   ❌ Found: {platform.system()}")
    else:
        version = platform.version()
        build = int(platform.version().split('.')[2]) if '.' in platform.version() else 0
        
        # Windows 10 1809 (Build 17763) or later
        if build < 17763:
            issues.append(f"❌ Windows 10 (Build 17763) or later required (found build {build})")
            print(f"   ❌ Windows build {build} is too old")
        else:
            win_version = platform.win32_ver()[0]
            print(f"   ✅ {platform.system()} {win_version} (Build {build})")
    
    # 2. Python Version Check
    print("\n[2/8] Checking Python version...")
    py_version = sys.version_info
    if py_version < (3, 10):
        issues.append(f"❌ Python 3.10+ required (found {py_version.major}.{py_version.minor}.{py_version.micro})")
        print(f"   ❌ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        print("   Download from: https://www.python.org/downloads/")
    else:
        print(f"   ✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    # 3. CPU Check
    print("\n[3/8] Checking CPU...")
    try:
        import psutil
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        
        if cpu_count_physical < 2:
            warnings.append(f"⚠️  Only {cpu_count_physical} physical CPU cores - transcription may be slow")
            print(f"   ⚠️  {cpu_count_physical} physical cores (4+ recommended)")
        else:
            print(f"   ✅ {cpu_count_physical} physical cores, {cpu_count_logical} logical cores")
            if cpu_count_logical >= 8:
                print("   💡 Good performance expected")
    except ImportError:
        warnings.append("⚠️  Could not check CPU (psutil not installed)")
        print("   ⚠️  psutil not installed - will check after setup")
    
    # 4. RAM Check
    print("\n[4/8] Checking RAM...")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if ram_gb < 4:
            issues.append(f"❌ Minimum 4GB RAM required (found {ram_gb:.1f}GB)")
            print(f"   ❌ {ram_gb:.1f}GB RAM (minimum 4GB)")
        elif ram_gb < 8:
            warnings.append(f"⚠️  8GB+ RAM recommended for best performance (found {ram_gb:.1f}GB)")
            print(f"   ⚠️  {ram_gb:.1f}GB RAM (8GB+ recommended)")
        else:
            print(f"   ✅ {ram_gb:.1f}GB RAM")
    except ImportError:
        warnings.append("⚠️  Could not check RAM (psutil not installed)")
        print("   ⚠️  psutil not installed - will check after setup")
    
    # 5. Disk Space Check
    print("\n[5/8] Checking disk space...")
    try:
        import psutil
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        
        if free_gb < 3:
            issues.append(f"❌ Minimum 3GB disk space needed (found {free_gb:.1f}GB free)")
            print(f"   ❌ {free_gb:.1f}GB free (minimum 3GB for models)")
        elif free_gb < 5:
            warnings.append(f"⚠️  5GB+ disk space recommended (found {free_gb:.1f}GB)")
            print(f"   ⚠️  {free_gb:.1f}GB free (5GB+ recommended)")
        else:
            print(f"   ✅ {free_gb:.1f}GB free")
    except ImportError:
        warnings.append("⚠️  Could not check disk space (psutil not installed)")
        print("   ⚠️  psutil not installed - will check after setup")
    
    # 6. Audio Devices Check
    print("\n[6/8] Checking audio devices...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        if not input_devices:
            issues.append("❌ No audio input device (microphone) found")
            print("   ❌ No microphone detected")
        else:
            print(f"   ✅ {len(input_devices)} input device(s) found")
            for d in input_devices[:3]:  # Show first 3
                print(f"      - {d['name']}")
        
        if not output_devices:
            warnings.append("⚠️  No audio output device found")
            print("   ⚠️  No speakers/headphones detected")
        
        # Check for Stereo Mix / Loopback
        has_loopback = any(
            'stereo mix' in d['name'].lower() or 
            'what u hear' in d['name'].lower() or
            'wave out mix' in d['name'].lower() or
            'loopback' in d['name'].lower()
            for d in input_devices
        )
        
        if not has_loopback:
            warnings.append("⚠️  No Stereo Mix detected - cannot record system audio (Teams, Zoom)")
            print("   ⚠️  Stereo Mix not enabled - see docs/MEETING_RECORDING_GUIDE.md")
        else:
            print("   ✅ Stereo Mix available for system audio recording")
            
    except ImportError:
        warnings.append("⚠️  Could not check audio devices (sounddevice not installed)")
        print("   ⚠️  sounddevice not installed - will check after setup")
    except Exception as e:
        warnings.append(f"⚠️  Audio check failed: {str(e)}")
        print(f"   ⚠️  Error checking audio: {str(e)}")
    
    # 7. Visual C++ Redistributable Check (Windows only)
    print("\n[7/8] Checking Visual C++ Redistributable...")
    if platform.system() == "Windows":
        try:
            # Check registry for VC++ Redistributable
            import winreg
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
                )
                winreg.CloseKey(key)
                print("   ✅ Visual C++ Redistributable installed")
            except FileNotFoundError:
                warnings.append("⚠️  Visual C++ Redistributable may be missing")
                print("   ⚠️  Not found - may be needed for some dependencies")
                print("   Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        except Exception as e:
            warnings.append(f"⚠️  Could not check VC++ Redistributable: {str(e)}")
            print(f"   ⚠️  Check failed: {str(e)}")
    else:
        print("   ⏭️  Skipped (not Windows)")
    
    # 8. Internet Connection Check (for model downloads)
    print("\n[8/8] Checking internet connection...")
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
        print("   ✅ Internet connected (needed for first-time model download)")
    except:
        warnings.append("⚠️  No internet connection - offline model download required")
        print("   ⚠️  No internet - models must be downloaded manually")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if not issues and not warnings:
        print("\n✅ All checks passed! System is ready for installation.")
        return True
    
    if issues:
        print(f"\n❌ Found {len(issues)} critical issue(s):")
        for issue in issues:
            print(f"   {issue}")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   {warning}")
    
    if issues:
        print("\n❌ Please fix critical issues before installation.")
        return False
    else:
        print("\n⚠️  System will work but some features may be limited.")
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        return response == 'y'


if __name__ == "__main__":
    print("\n🔍 Meeting Transcription - System Check\n")
    
    result = check_prerequisites()
    
    if result:
        print("\n✅ Ready to proceed with setup!")
        print("   Run: .\\setup.ps1")
        sys.exit(0)
    else:
        print("\n❌ System requirements not met.")
        sys.exit(1)
