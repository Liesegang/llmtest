from kokoro_onnx import Kokoro
import sounddevice as sd
import threading
import sys

class KokoroTTS:
    def __init__(self, model_path, voices_path):
        print("🔄 Kokoro TTSモデル読み込み中...")
        try:
            self.kokoro = Kokoro(model_path, voices_path)
            self.lock = threading.Lock()
        except Exception as e:
            print(f"⚠️ TTSモデルの読み込みに失敗しました: {e}")
            self.kokoro = None

    def speak(self, text, lang="en-us", voice="af_bella"):
        """
        指定されたテキストをTTSで読み上げる
        """
        if not self.kokoro or not text:
            return

        print(f"🔊 読み上げ中: {text}")
        try:
            with self.lock:
                samples, sample_rate = self.kokoro.create(
                    text, 
                    voice=voice, 
                    speed=1.0, 
                    lang=lang
                )
                
                if samples is not None and len(samples) > 0:
                    sd.play(samples, sample_rate)
                    sd.wait()
                else:
                    print("⚠️ 音声生成に失敗しました")
        except Exception as e:
            print(f"❌ TTSエラー: {e}")
