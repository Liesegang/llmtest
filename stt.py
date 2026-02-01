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
        
    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def _transcribe_worker(self, on_text_callback):
        """
        音声データを監視し、VADで区切られた「一文」ごとに推論を回す
        """
        # Silero VADのロード
        print("🔄 VADモデル読み込み中...")
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          trust_repo=True)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        
        # VADイテレータ
        vad_iterator = VADIterator(vad_model)
        
        print(f"\n🎧 待機中... 話しかけてください (Ctrl+C で終了)\n")

        current_speech_buffer = []
        is_speaking = False

        while self.is_running:
            try:
                data = self.audio_queue.get(timeout=1.0) 
            except queue.Empty:
                continue

            audio_chunk = torch.from_numpy(data.flatten())
            speech_dict = vad_iterator(audio_chunk, return_seconds=True)
            current_speech_buffer.append(data)

            if speech_dict:
                if 'start' in speech_dict:
                    if not is_speaking:
                        sys.stdout.write("🗣️  認識開始...\r")
                        sys.stdout.flush()
                        is_speaking = True
                        current_speech_buffer = current_speech_buffer[-10:]

                if 'end' in speech_dict:
                    sys.stdout.write("                   \r")
                    
                    if len(current_speech_buffer) > 0:
                        full_audio = np.concatenate(current_speech_buffer, axis=0).flatten()
                        
                        # 推論実行 (英語)
                        segments, _ = self.model.transcribe(
                            full_audio,
                            beam_size=10,
                            language="en",
                            condition_on_previous_text=False,
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
                    vad_iterator.reset_states()

    def start(self, on_text_callback):
        self.is_running = True
        
        # 録音スレッド開始 (sounddeviceはブロックするので、ここではInputStreamを開いたままworkerを呼ぶ形にするか、
        # あるいはInputStreamを別管理にする必要がある。
        # シンプルにメインスレッドをブロックしないように、ここでもう一つスレッドを作るか、
        # あるいはmain.py側でブロックさせるか。
        # 今回はstt.pyが自律的に動くように、内部でInputStream管理とWorkerスレッド起動を行う。
        
        self.worker_thread = threading.Thread(
            target=self._transcribe_worker_wrapper, 
            args=(on_text_callback,), 
            daemon=True
        )
        self.worker_thread.start()

    def _transcribe_worker_wrapper(self, on_text_callback):
        # InputStreamをこのスレッド(またはCallback)で維持する必要がある。
        # sounddevice.InputStream はContext Managerとして使うのが一般的。
        with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                            callback=self._audio_callback, blocksize=self.block_size):
            self._transcribe_worker(on_text_callback)
