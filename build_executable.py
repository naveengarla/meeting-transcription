"""
Build Executable with PyInstaller
Automates the process of creating a standalone Windows executable.
"""

import subprocess
import sys
import shutil
from pathlib import Path
import platform


def check_prerequisites():
    """Check if all prerequisites are installed."""
    print("🔍 Checking prerequisites...")
    
    # Check PyInstaller
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__} installed")
    except ImportError:
        print("❌ PyInstaller not found")
        print("\nInstalling PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("✅ PyInstaller installed")
    
    # Check if on Windows
    if platform.system() != 'Windows':
        print(f"⚠️ Warning: This script is designed for Windows. Current OS: {platform.system()}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Build cancelled.")
            sys.exit(0)
    
    # Check if .spec file exists
    spec_file = Path('meeting_transcription.spec')
    if not spec_file.exists():
        print(f"❌ Spec file not found: {spec_file}")
        print("Please ensure meeting_transcription.spec is in the current directory.")
        sys.exit(1)
    else:
        print(f"✅ Spec file found: {spec_file}")
    
    print()


def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("🧹 Cleaning previous build artifacts...")
    
    dirs_to_remove = ['build', 'dist']
    
    for dir_name in dirs_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"   Removing {dir_name}/")
            shutil.rmtree(dir_path)
    
    print("✅ Clean complete\n")


def build_executable():
    """Build the executable using PyInstaller."""
    print("🔨 Building executable...")
    print("   This may take several minutes...\n")
    
    # Run PyInstaller
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",  # Clean PyInstaller cache
        "--noconfirm",  # Overwrite output directory
        "meeting_transcription.spec"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Build successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed with error code: {e.returncode}")
        return False


def verify_build():
    """Verify the build output."""
    print("\n🔍 Verifying build output...")
    
    dist_dir = Path('dist')
    exe_file = dist_dir / 'MeetingTranscription.exe'
    
    if not dist_dir.exists():
        print("❌ dist/ directory not found")
        return False
    
    if not exe_file.exists():
        print(f"❌ Executable not found: {exe_file}")
        return False
    
    # Get file size
    size_mb = exe_file.stat().st_size / (1024 * 1024)
    print(f"✅ Executable found: {exe_file}")
    print(f"   Size: {size_mb:.1f} MB")
    
    # Check for critical files
    expected_files = ['README.md', 'QUICKSTART.md']
    for file_name in expected_files:
        file_path = dist_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} included")
        else:
            print(f"⚠️ {file_name} not found in dist/")
    
    return True


def create_distribution_package():
    """Create a distribution package."""
    print("\n📦 Creating distribution package...")
    
    dist_dir = Path('dist')
    
    # Create README for distribution
    dist_readme = dist_dir / 'README_DISTRIBUTION.txt'
    
    readme_content = """
Meeting Transcription - Windows Executable
==========================================

Thank you for downloading Meeting Transcription!

QUICK START:
1. Double-click MeetingTranscription.exe to run
2. First run will download AI models (75MB - 1.5GB depending on selection)
3. Configure settings via Settings menu
4. Enable Stereo Mix for meeting recording (see guide below)

REQUIREMENTS:
- Windows 10 (Version 1809 or later)
- 4GB RAM minimum (8GB recommended)
- 3GB free disk space
- Audio input device (microphone or Stereo Mix)

FIRST-TIME SETUP:
1. Run check_system.py to verify your system
2. Run audio_setup_helper.py to configure audio devices
3. Run download_models.py to pre-download AI models (optional)

ENABLING STEREO MIX (for meeting transcription):
1. Right-click Speaker icon in system tray
2. Click "Sound settings" → "Recording" tab
3. Right-click empty area → "Show Disabled Devices"
4. Enable "Stereo Mix"
5. Set as default recording device

DOCUMENTATION:
- See QUICKSTART.md for step-by-step guide
- See docs/ folder for technical documentation
- See README.md for full project information

TROUBLESHOOTING:
- Run diagnostic_report.py to generate troubleshooting report
- Check logs/ folder for error details
- See docs/REQUIREMENTS.md for system requirements

SUPPORT:
- GitHub: https://github.com/naveengarla/meeting-transcription
- Issues: https://github.com/naveengarla/meeting-transcription/issues

VERSION: 1.0.0
BUILD DATE: {build_date}
"""
    
    from datetime import datetime
    build_date = datetime.now().strftime("%Y-%m-%d")
    
    with open(dist_readme, 'w') as f:
        f.write(readme_content.format(build_date=build_date))
    
    print(f"✅ Created {dist_readme.name}")
    
    # Create batch file for easy execution
    batch_file = dist_dir / 'Run_Meeting_Transcription.bat'
    
    batch_content = """@echo off
echo Starting Meeting Transcription...
echo.
MeetingTranscription.exe
pause
"""
    
    with open(batch_file, 'w') as f:
        f.write(batch_content)
    
    print(f"✅ Created {batch_file.name}")
    
    print("\n📦 Distribution package ready in dist/ folder")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("BUILD COMPLETE!")
    print("="*60)
    
    print("\n📁 Output location: dist/MeetingTranscription.exe")
    
    print("\n📋 Next Steps:")
    print("1. Test the executable:")
    print("   cd dist")
    print("   .\\MeetingTranscription.exe")
    print()
    print("2. Test on a clean Windows machine without Python installed")
    print()
    print("3. Create installer (optional):")
    print("   - Use NSIS (Nullsoft Scriptable Install System)")
    print("   - Or use Inno Setup")
    print("   - Or distribute as ZIP file")
    print()
    print("4. Distribution checklist:")
    print("   ✓ Test executable on clean Windows 10 machine")
    print("   ✓ Test with different audio devices")
    print("   ✓ Verify model downloads work")
    print("   ✓ Test Settings UI functionality")
    print("   ✓ Check error handling and fallbacks")
    print()
    print("5. Optional: Code signing")
    print("   - Get code signing certificate")
    print("   - Sign with: signtool.exe")
    print("   - Prevents Windows SmartScreen warnings")
    
    print("\n" + "="*60)


def main():
    """Main build process."""
    print("="*60)
    print("Meeting Transcription - Build Executable")
    print("="*60)
    print()
    
    # Step 1: Check prerequisites
    check_prerequisites()
    
    # Step 2: Clean previous builds
    response = input("Clean previous build artifacts? (y/n): ")
    if response.lower() == 'y':
        clean_build_artifacts()
    
    # Step 3: Build executable
    print("Ready to build executable.")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Build cancelled.")
        return
    
    success = build_executable()
    
    if not success:
        print("\n❌ Build failed. Check error messages above.")
        sys.exit(1)
    
    # Step 4: Verify build
    if not verify_build():
        print("\n⚠️ Build verification failed. Check dist/ folder.")
        sys.exit(1)
    
    # Step 5: Create distribution package
    create_distribution_package()
    
    # Step 6: Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
