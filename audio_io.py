import sounddevice as sd
import numpy as np
import threading
import queue
import time
import sys

class AudioIO:
    def __init__(self, sample_rate=16000, block_size=512):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        self.output_queue = queue.Queue() # TTS writes here
        self.input_queue = queue.Queue()  # STT reads from here
        
        self.stream = None
        self.running = False
        
        # Internal state for "is_playing"
        self._is_playing_internal = False

    def start(self):
        if self.stream is not None:
            return
            
        # Device Selection Logic
        input_device = None
        output_device = None
        
        print("🔍 Audio Device Search (Target: NVIDIA Broadcast / RTX Voice)...")
        try:
            devices = sd.query_devices()
            # Targets: "NVIDIA Broadcast", "RTX Voice", "RTX-Audio"
            # We match partial string.
            
            for i, d in enumerate(devices):
                name = d['name']
                name_lower = name.lower()
                # Look for Input
                if ("nvidia" in name_lower and "broadcast" in name_lower) or \
                   ("rtx" in name_lower and ("point" in name_lower or "voice" in name_lower)):
                    if d['max_input_channels'] > 0 and input_device is None:
                        input_device = i
                        print(f"  🎤 Found Input: [{i}] {name}")
                    if d['max_output_channels'] > 0 and output_device is None:
                        # User reported that System Default output wasn't recognized by AEC.
                        # Route Output through NVIDIA Broadcast so it captures the reference signal.
                        output_device = i
                        print(f"  🔊 Found Output: [{i}] {name}")

        except Exception as e:
            print(f"⚠️ Device search failed: {e}")

        # Fallback to defaults if not found
        if input_device is None:
            print("  ⚠️ NVIDIA Broadcast Input not found. Using system default.")
        if output_device is None:
            print("  ⚠️ NVIDIA Broadcast Output not found. Using system default.")

        self.running = True
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=(input_device, output_device),
            dtype='float32',
            channels=1,
            callback=self._callback
        )
        self.stream.start()
        print(f"✅ AudioIO Started | Device: In={input_device}, Out={output_device}")

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def enqueue_output(self, samples):
        # Chunk logic
        cursor = 0
        while cursor < len(samples):
            chunk = samples[cursor : cursor + self.block_size]
            if len(chunk) < self.block_size:
                pad = np.zeros(self.block_size - len(chunk), dtype='float32')
                chunk = np.concatenate((chunk, pad))
            self.output_queue.put(chunk)
            cursor += self.block_size

    def cancel_playback(self):
        """
        現在の再生キューをクリアして直ちに音声を止める
        """
        with self.output_queue.mutex:
            self.output_queue.queue.clear()
        self._is_playing_internal = False

    @property
    def is_playing(self):
        # Check if queue has items or if we recently processed output
        return not self.output_queue.empty() or self._is_playing_internal

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
            
        # 1. Output Processing (Speaker)
        try:
            out_chunk = self.output_queue.get_nowait()
            self._is_playing_internal = True
        except queue.Empty:
            out_chunk = np.zeros(frames, dtype='float32')
            self._is_playing_internal = False
        
        outdata[:] = out_chunk.reshape(-1, 1)

        # 2. Input Processing (Mic)
        # Barge-in Enabled: We pass input through.
        # We rely on NVIDIA Broadcast (set in start()) to remove the echo.
        self.input_queue.put(indata.flatten())
