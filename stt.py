import queue
import threading
import sys
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

class WhisperSTT:
    def __init__(self, model_size="large-v3", device="auto", compute_type="float32"):
        print(f"🔄 Whisperモデル読み込み中 ({model_size})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.sample_rate = 16000
        self.block_size = 512
        


    def _transcribe_worker(self, on_text_callback):
        print("🔄 VADモデル読み込み中...")
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          trust_repo=True)
        # We only need the model for manual inference
        
        print(f"\n🎧 待機中... 話しかけてください (Ctrl+C で終了)\n")

        current_speech_buffer = []
        is_speaking = False
        silence_counter = 0

        # VAD requires exactly 512 samples for 16kHz
        VAD_WINDOW = 512 
        buffer_accum = np.zeros(0, dtype='float32')

        while self.is_running:
            try:
                data = self.audio_queue.get(timeout=1.0) 
            except queue.Empty:
                continue

            # Accumulate buffer
            buffer_accum = np.concatenate((buffer_accum, data.flatten()))
            
            # Process in 512-sample chunks
            while len(buffer_accum) >= VAD_WINDOW:
                # Extract 512 samples
                audio_chunk_np = buffer_accum[:VAD_WINDOW]
                buffer_accum = buffer_accum[VAD_WINDOW:]
                
                audio_chunk = torch.from_numpy(audio_chunk_np)
                
                # Dynamic Threshold Logic
                # If AI is playing, set high threshold (only loud inputs)
                # If silent, set low threshold (sensitive)
                if self.audio_io and self.audio_io.is_playing:
                    threshold = 0.8
                else:
                    threshold = 0.4
                    
                # Manual Inference
                # Note: Silero VAD model expects (batch, samples) or just samples?
                # Usually (1, samples) or (samples,).
                speech_prob = vad_model(audio_chunk, 16000).item()
                
                if speech_prob > threshold:
                    silence_counter = 0
                    if not is_speaking:
                        # Start of speech (Barge-in)
                        if self.audio_io and self.audio_io.is_playing:
                             sys.stdout.write("\n🛑 割り込み検知 (Barge-in) -> 再生停止\n")
                             self.audio_io.cancel_playback()
                             
                        sys.stdout.write("🗣️  認識開始...\r")
                        sys.stdout.flush()
                        is_speaking = True
                        current_speech_buffer = current_speech_buffer[-10:] # Keep pre-roll
                    current_speech_buffer.append(audio_chunk_np)
                else:
                    if is_speaking:
                        silence_counter += 1
                        current_speech_buffer.append(audio_chunk_np) # Keep trailing silence for a bit
                        
                        # End of speech detection (e.g. 500ms silence = ~16 chunks of 512sa)
                        if silence_counter > 20: 
                            sys.stdout.write("                   \r")
                            # Process Audio
                            if len(current_speech_buffer) > 0:
                                full_audio = np.concatenate(current_speech_buffer, axis=0).flatten()
                                
                                segments, _ = self.model.transcribe(
                                    full_audio,
                                    beam_size=10,
                                    language="en",
                                    condition_on_previous_text=False, # Reduce hallucinations in streaming
                                    initial_prompt="This is a polite English conversation.",
                                    word_timestamps=True
                                )

                                segments = list(segments)
                                text_result = "".join([s.text for s in segments]).strip()
                                
                                if text_result:
                                    print(f"User: {text_result}")
                                    on_text_callback(text_result)
                                    
                                    for segment in segments:
                                        if segment.words:
                                            for word in segment.words:
                                                print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}")

                                print("---------------------------")
                            
                            is_speaking = False
                            current_speech_buffer = []
                            silence_counter = 0
                            # vad_model.reset_states() # If model is stateful? Silero standard model is usually stateless per forward? 
                            # Actually Silero V5 is stateful but standard hub load might be v4.
                            # Standard usage `model(x, sr)` is often stateless context-wise unless state is passed.


    def start(self, audio_io, on_text_callback):
        self.is_running = True
        self.audio_io = audio_io
        self.audio_queue = audio_io.input_queue 

        
        self.worker_thread = threading.Thread(
            target=self._transcribe_worker, 
            args=(on_text_callback,), 
            daemon=True
        )
        self.worker_thread.start()

