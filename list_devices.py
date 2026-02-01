import sounddevice as sd

print("--- Audio Devices ---")
# Handle encoding for Windows Console
import sys
sys.stdout.reconfigure(encoding='utf-8')

devs = sd.query_devices()
for i, d in enumerate(devs):
    try:
        print(f"{i}: {d['name']} (In:{d['max_input_channels']} Out:{d['max_output_channels']})")
    except Exception:
        print(f"{i}: [Decode Error]")
print("---------------------")

try:
    wasapi = sd.query_hostapis(index=sd.default.hostapi)
    print(f"Default Host API: {wasapi['name']}")
except:
    pass
