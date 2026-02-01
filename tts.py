from kokoro_onnx import Kokoro
import sounddevice as sd
import threading
import sys
from tts_interface import TTSInterface

class KokoroTTS(TTSInterface):
    def __init__(self, model_path, voices_path, audio_io):
        print("🔄 Kokoro TTSモデル読み込み中...")
        try:
            self.kokoro = Kokoro(model_path, voices_path)
            self.audio_io = audio_io
            self.lock = threading.Lock()
        except Exception as e:
            print(f"⚠️ TTSモデルの読み込みに失敗しました: {e}")
            self.kokoro = None

    def speak(self, text, lang="en-us", voice="af_bella"):
        """
        指定されたテキストをTTSで読み上げる (AudioIOへエンキュー)
        """
        if not self.kokoro or not text:
            return

        print(f"🔊 読み上げ中: {text}")
        try:
            with self.lock:
                # Kokoro returns (samples, sample_rate)
                # target_sample_rate is hardcoded to 24000 in Kokoro usually? 
                # We need to resample if AudioIO expects 16000.
                # But for simplicity, we assume AudioIO handles buffering, 
                # AND we need to resample here if rates mismatch.
                
                samples, sample_rate = self.kokoro.create(
                    text, 
                    voice=voice, 
                    speed=1.0, 
                    lang=lang
                )
                
                if samples is not None and len(samples) > 0:
                    # Resample to AudioIO rate (16000)
                    if sample_rate != self.audio_io.sample_rate:
                         import scipy.signal
                         # Calculate number of samples
                         num_samples = int(len(samples) * self.audio_io.sample_rate / sample_rate)
                         samples = scipy.signal.resample(samples, num_samples)
                    
                    self.audio_io.enqueue_output(samples)
                else:
                    print("⚠️ 音声生成に失敗しました")
        except Exception as e:
            print(f"❌ TTSエラー: {e}")
