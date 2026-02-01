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

        self.running = True
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype='float32',
            channels=1,
            callback=self._callback
        )
        self.stream.start()
        print("✅ AudioIO Started (NVIDIA Broadcast Mode)")

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

    @property
    def is_playing(self):
        # Check if queue has items or if we recently processed output
        return not self.output_queue.empty() or self._is_playing_internal

    def _callback(self, indata, outdata, frames, time, status):
        # 1. Output Processing (Speaker)
        try:
            out_chunk = self.output_queue.get_nowait()
            self._is_playing_internal = True
        except queue.Empty:
            out_chunk = np.zeros(frames, dtype='float32')
            self._is_playing_internal = False
        
        outdata[:] = out_chunk.reshape(-1, 1)

        # 2. Input Processing (Mic)
        # Just Pass-through. NVIDIA Broadcast handles noise/echo.
        self.input_queue.put(indata.flatten())
