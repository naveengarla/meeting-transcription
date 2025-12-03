"""
Diagnostic Report Generator
Creates a comprehensive troubleshooting report for support purposes.
"""

import sys
import platform
import json
from datetime import datetime
from pathlib import Path


def generate_diagnostic_report():
    """Generate comprehensive diagnostic report."""
    report = {}
    
    # System Information
    print("Collecting system information...")
    report['system'] = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'python_executable': sys.executable,
    }
    
    # Hardware Information
    try:
        import psutil
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        report['hardware'] = {
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_total_gb': round(mem.total / (1024**3), 2),
            'ram_available_gb': round(mem.available / (1024**3), 2),
            'ram_percent': mem.percent,
            'disk_total_gb': round(disk.total / (1024**3), 2),
            'disk_free_gb': round(disk.free / (1024**3), 2),
            'disk_percent': disk.percent,
        }
    except Exception as e:
        report['hardware'] = {'error': str(e)}
    
    # Audio Devices
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        report['audio_devices'] = {
            'total_devices': len(devices),
            'input_devices': [],
            'output_devices': [],
            'loopback_available': False,
        }
        
        for idx, device in enumerate(devices):
            device_info = {
                'index': idx,
                'name': device['name'],
                'channels_in': device['max_input_channels'],
                'channels_out': device['max_output_channels'],
                'sample_rate': device['default_samplerate'],
            }
            
            if device['max_input_channels'] > 0:
                report['audio_devices']['input_devices'].append(device_info)
                
                # Check for stereo mix / loopback
                if any(keyword in device['name'].lower() for keyword in 
                       ['stereo mix', 'what u hear', 'wave out mix', 'loopback']):
                    report['audio_devices']['loopback_available'] = True
            
            if device['max_output_channels'] > 0:
                report['audio_devices']['output_devices'].append(device_info)
        
    except Exception as e:
        report['audio_devices'] = {'error': str(e)}
    
    # Installed Python Packages
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            report['installed_packages'] = {
                'count': len(packages),
                'packages': {pkg['name']: pkg['version'] for pkg in packages}
            }
        else:
            report['installed_packages'] = {'error': 'pip list failed'}
    except Exception as e:
        report['installed_packages'] = {'error': str(e)}
    
    # Whisper Model Cache
    try:
        from pathlib import Path
        
        # Check common cache locations
        cache_locations = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "whisper",
            Path("models"),
        ]
        
        report['model_cache'] = {
            'locations': [],
        }
        
        for cache_dir in cache_locations:
            if cache_dir.exists():
                models = []
                for item in cache_dir.iterdir():
                    if item.is_file() or item.is_dir():
                        size_mb = 0
                        if item.is_file():
                            size_mb = round(item.stat().st_size / (1024**2), 2)
                        elif item.is_dir():
                            # Calculate directory size
                            size_mb = round(sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024**2), 2)
                        
                        models.append({
                            'name': item.name,
                            'size_mb': size_mb,
                            'type': 'file' if item.is_file() else 'directory'
                        })
                
                if models:
                    report['model_cache']['locations'].append({
                        'path': str(cache_dir),
                        'models': models,
                        'total_size_mb': round(sum(m['size_mb'] for m in models), 2)
                    })
    
    except Exception as e:
        report['model_cache'] = {'error': str(e)}
    
    # Performance Logs
    try:
        log_files = list(Path('logs').glob('performance_*.jsonl'))
        
        report['performance_logs'] = {
            'log_files': [str(f) for f in log_files],
            'total_logs': len(log_files),
        }
        
        # Read last 5 entries from most recent log
        if log_files:
            latest_log = sorted(log_files)[-1]
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                recent_entries = [json.loads(line) for line in lines[-5:]]
                report['performance_logs']['recent_transcriptions'] = recent_entries
    
    except Exception as e:
        report['performance_logs'] = {'error': str(e)}
    
    # Configuration
    try:
        from dotenv import dotenv_values
        config = dotenv_values('.env')
        
        # Redact sensitive information
        safe_config = {}
        for key, value in config.items():
            if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                safe_config[key] = '***REDACTED***'
            else:
                safe_config[key] = value
        
        report['configuration'] = safe_config
    
    except Exception as e:
        report['configuration'] = {'error': str(e)}
    
    # Recent Errors (if error log exists)
    try:
        error_log = Path('logs') / 'errors.log'
        if error_log.exists():
            with open(error_log, 'r') as f:
                lines = f.readlines()
                report['recent_errors'] = lines[-20:]  # Last 20 errors
        else:
            report['recent_errors'] = []
    except Exception as e:
        report['recent_errors'] = {'error': str(e)}
    
    return report


def print_report(report):
    """Print diagnostic report in readable format."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC REPORT")
    print("=" * 70)
    
    # System
    print("\n📊 SYSTEM INFORMATION")
    print("-" * 70)
    sys_info = report.get('system', {})
    print(f"Platform:        {sys_info.get('platform', 'N/A')}")
    print(f"System:          {sys_info.get('system', 'N/A')}")
    print(f"Version:         {sys_info.get('version', 'N/A')}")
    print(f"Processor:       {sys_info.get('processor', 'N/A')}")
    print(f"Python:          {sys_info.get('python_version', 'N/A').split()[0]}")
    
    # Hardware
    if 'hardware' in report and 'error' not in report['hardware']:
        hw = report['hardware']
        print("\n💻 HARDWARE")
        print("-" * 70)
        print(f"CPU Cores:       {hw.get('cpu_count_physical', 'N/A')} physical, {hw.get('cpu_count_logical', 'N/A')} logical")
        print(f"CPU Usage:       {hw.get('cpu_percent', 'N/A')}%")
        print(f"RAM:             {hw.get('ram_available_gb', 'N/A')}/{hw.get('ram_total_gb', 'N/A')} GB available ({hw.get('ram_percent', 'N/A')}% used)")
        print(f"Disk:            {hw.get('disk_free_gb', 'N/A')}/{hw.get('disk_total_gb', 'N/A')} GB free ({hw.get('disk_percent', 'N/A')}% used)")
    
    # Audio
    if 'audio_devices' in report and 'error' not in report['audio_devices']:
        audio = report['audio_devices']
        print("\n🎤 AUDIO DEVICES")
        print("-" * 70)
        print(f"Total Devices:   {audio.get('total_devices', 0)}")
        print(f"Input Devices:   {len(audio.get('input_devices', []))}")
        print(f"Output Devices:  {len(audio.get('output_devices', []))}")
        print(f"Stereo Mix:      {'✅ Available' if audio.get('loopback_available') else '❌ Not found'}")
        
        if audio.get('input_devices'):
            print("\nInput Devices:")
            for dev in audio['input_devices'][:5]:  # Show first 5
                print(f"  • {dev['name']}")
    
    # Models
    if 'model_cache' in report and 'error' not in report['model_cache']:
        cache = report['model_cache']
        print("\n🤖 MODEL CACHE")
        print("-" * 70)
        total_size = sum(loc['total_size_mb'] for loc in cache.get('locations', []))
        total_models = sum(len(loc['models']) for loc in cache.get('locations', []))
        print(f"Cached Models:   {total_models}")
        print(f"Total Size:      {total_size:.0f} MB")
        
        for loc in cache.get('locations', []):
            if loc['models']:
                print(f"\n{loc['path']}:")
                for model in loc['models'][:5]:  # Show first 5
                    print(f"  • {model['name']} ({model['size_mb']:.0f} MB)")
    
    # Performance
    if 'performance_logs' in report and 'error' not in report['performance_logs']:
        perf = report['performance_logs']
        print("\n⚡ PERFORMANCE LOGS")
        print("-" * 70)
        print(f"Log Files:       {perf.get('total_logs', 0)}")
        
        if perf.get('recent_transcriptions'):
            print("\nRecent Transcriptions:")
            for entry in perf['recent_transcriptions'][-3:]:  # Last 3
                duration = entry.get('audio_duration_sec', 0)
                speed = entry.get('speed_multiplier', 0)
                print(f"  • {duration:.1f}s audio → {speed:.1f}x real-time ({entry.get('model', 'N/A')} model)")
    
    # Configuration
    if 'configuration' in report and 'error' not in report['configuration']:
        config = report['configuration']
        print("\n⚙️  CONFIGURATION")
        print("-" * 70)
        for key, value in list(config.items())[:10]:  # Show first 10
            print(f"{key:25s} = {value}")
    
    print("\n" + "=" * 70)


def save_report(report, filename=None):
    """Save report to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostic_report_{timestamp}.json"
    
    Path('logs').mkdir(exist_ok=True)
    filepath = Path('logs') / filename
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    return filepath


if __name__ == "__main__":
    print("\n🔍 Generating Diagnostic Report...")
    print("This may take a few seconds...\n")
    
    try:
        report = generate_diagnostic_report()
        print_report(report)
        
        # Save to file
        filepath = save_report(report)
        print(f"\n💾 Full report saved to: {filepath}")
        print(f"\n📧 Share this file when reporting issues for faster support!")
        
    except Exception as e:
        print(f"\n❌ Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
