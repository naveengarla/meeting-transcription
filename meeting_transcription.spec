# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification file for Meeting Transcription
Builds a standalone Windows executable with all dependencies.
"""

import sys
from pathlib import Path

block_cipher = None

# Application metadata
app_name = 'MeetingTranscription'
version = '1.0.0'

# Paths
spec_root = Path(SPECPATH)
project_root = spec_root
src_path = project_root / 'src'

# Data files to include
datas = [
    # Models directory - EXCLUDED to reduce size (models download on first run)
    # Uncomment the line below to bundle models (increases size by ~2GB):
    # ('models', 'models'),
    
    # Documentation
    ('README.md', '.'),
    ('QUICKSTART.md', '.'),
    ('docs', 'docs'),
    
    # Example .env template
    ('.env.example', '.'),
]

# Hidden imports (modules that PyInstaller might miss)
hiddenimports = [
    # Core dependencies
    'faster_whisper',
    'whisper',
    'sounddevice',
    'soundfile',
    'numpy',
    'scipy',
    'librosa',
    
    # PyQt6
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtWidgets',
    'PyQt6.QtGui',
    
    # Azure Speech
    'azure.cognitiveservices.speech',
    
    # Utilities
    'psutil',
    'python-dotenv',
    'dotenv',
    
    # Faster-whisper dependencies
    'ctranslate2',
    'huggingface_hub',
    'tokenizers',
    
    # Audio processing
    '_soundfile_data',
]

# Binary files to include
binaries = []

# Collect audio library binaries
if sys.platform == 'win32':
    # PortAudio DLL for sounddevice
    import sounddevice
    sd_path = Path(sounddevice.__file__).parent
    portaudio_dll = sd_path / '_sounddevice_data' / 'portaudio-binaries' / 'libportaudio64bit.dll'
    if portaudio_dll.exists():
        binaries.append((str(portaudio_dll), '_sounddevice_data/portaudio-binaries'))

# Analysis
a = Analysis(
    ['run.py'],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        'PIL',
        'tkinter',
        'jupyter',
        'notebook',
        'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out duplicate or unnecessary files
a.datas = [x for x in a.datas if not x[0].startswith('share/')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress executable
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Show console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon file path here if you have one: icon='icon.ico'
)

# Optional: Create a COLLECT for one-folder mode (easier debugging)
# Uncomment if you prefer a folder distribution instead of single .exe
"""
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)
"""
