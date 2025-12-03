"""
Performance Analysis Script
Analyzes transcription performance metrics from logs
"""
import json
from pathlib import Path
from datetime import datetime
import statistics

# Find all performance log files
logs_dir = Path(__file__).parent / "logs"

if not logs_dir.exists():
    print("No logs directory found. Run some transcriptions first!")
    exit(1)

log_files = list(logs_dir.glob("performance_*.jsonl"))

if not log_files:
    print("No performance logs found. Run some transcriptions first!")
    exit(1)

print(f"Found {len(log_files)} performance log file(s)\n")

# Parse all metrics
all_metrics = []
for log_file in log_files:
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                metrics = json.loads(line.strip())
                all_metrics.append(metrics)
            except json.JSONDecodeError:
                continue

if not all_metrics:
    print("No valid metrics found in log files")
    exit(1)

print(f"📊 Performance Analysis Report")
print(f"=" * 70)
print(f"Total transcriptions: {len(all_metrics)}\n")

# Group by model
by_model = {}
for m in all_metrics:
    model = m.get('model', 'unknown')
    if model not in by_model:
        by_model[model] = []
    by_model[model].append(m)

# Analyze each model
for model, metrics_list in by_model.items():
    print(f"\n🎯 Model: {model}")
    print(f"-" * 70)
    print(f"Transcriptions: {len(metrics_list)}")
    
    # Filter out errors
    successful = [m for m in metrics_list if 'error' not in m]
    failed = [m for m in metrics_list if 'error' in m]
    
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for f in failed:
            print(f"   - Error: {f['error']}")
    
    if not successful:
        continue
    
    # Calculate statistics
    speeds = [m['speed_multiplier'] for m in successful if 'speed_multiplier' in m]
    durations = [m['audio_duration_sec'] for m in successful if 'audio_duration_sec' in m]
    trans_times = [m['transcription_time_sec'] for m in successful if 'transcription_time_sec' in m]
    cpu_usage = [m['cpu_percent_avg'] for m in successful if 'cpu_percent_avg' in m]
    memory_usage = [m['memory_used_mb'] for m in successful if 'memory_used_mb' in m]
    
    print(f"\n⚡ Speed Performance:")
    if speeds:
        print(f"   Average speed: {statistics.mean(speeds):.2f}x real-time")
        print(f"   Best speed: {max(speeds):.2f}x real-time")
        print(f"   Worst speed: {min(speeds):.2f}x real-time")
        if len(speeds) > 1:
            print(f"   Std deviation: {statistics.stdev(speeds):.2f}x")
    
    print(f"\n🎵 Audio Processing:")
    if durations:
        print(f"   Total audio: {sum(durations):.1f} seconds ({sum(durations)/60:.1f} minutes)")
        print(f"   Average duration: {statistics.mean(durations):.1f} seconds")
        print(f"   Longest recording: {max(durations):.1f} seconds")
    
    print(f"\n⏱️ Transcription Time:")
    if trans_times:
        print(f"   Total time: {sum(trans_times):.1f} seconds ({sum(trans_times)/60:.1f} minutes)")
        print(f"   Average time: {statistics.mean(trans_times):.1f} seconds")
        print(f"   Time saved: {sum(durations) - sum(trans_times):.1f} seconds")
    
    print(f"\n💻 Resource Usage:")
    if cpu_usage:
        print(f"   Average CPU: {statistics.mean(cpu_usage):.1f}%")
        print(f"   Peak CPU: {max(cpu_usage):.1f}%")
    if memory_usage:
        print(f"   Average memory: {statistics.mean(memory_usage):.1f} MB")
        print(f"   Peak memory: {max(memory_usage):.1f} MB")
    
    # Language breakdown
    languages = {}
    for m in successful:
        lang = m.get('detected_language', 'unknown')
        languages[lang] = languages.get(lang, 0) + 1
    
    print(f"\n🌍 Languages Detected:")
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
        print(f"   {lang}: {count}")

# Recent performance
print(f"\n\n📈 Recent Performance (Last 10 Transcriptions)")
print(f"=" * 70)
recent = sorted(all_metrics, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]

print(f"{'Date/Time':<20} {'Model':<8} {'Duration':<10} {'Time':<8} {'Speed':<8} {'Lang':<6}")
print(f"-" * 70)

for m in recent:
    timestamp = m.get('timestamp', 'unknown')[:19]  # YYYY-MM-DD HH:MM:SS
    model = m.get('model', 'unknown')
    duration = f"{m.get('audio_duration_sec', 0):.1f}s"
    trans_time = f"{m.get('transcription_time_sec', 0):.1f}s"
    speed = f"{m.get('speed_multiplier', 0):.2f}x"
    lang = m.get('detected_language', 'unknown')[:6]
    
    print(f"{timestamp:<20} {model:<8} {duration:<10} {trans_time:<8} {speed:<8} {lang:<6}")

print(f"\n✅ Analysis complete! Log files location: {logs_dir}")
