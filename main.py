import threading
import time
from stt import WhisperSTT
from tts import KokoroTTS

try:
    from llm import LocalLLM
except ImportError:
    print("⚠️ LocalLLMモジュール(llama_cpp)が見つかりません。")
    LocalLLM = None

# --- 設定 ---
# STT設定
MODEL_SIZE = "large-v3"
DEVICE = "auto"
COMPUTE_TYPE = "float32"

# TTS設定
KOKORO_MODEL_PATH = "model_assets/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "model_assets/voices-v1.0.bin"

# LLM設定
MODEL_PATH = "model_assets/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"

# 処理中のフラグ
processing_lock = threading.Lock()

def process_response(text, tts, llm):
    """
    LLMに投げてTTSで喋らせる（ブロッキング実行想定）
    """
    if not text:
        return

    with processing_lock: # 同時実行を防ぐ（簡易的）
        print(f"🤔 AI考え中... User: {text}")
        try:
            # ストリーミング生成で、一文ごとにTTSに投げる
            print(f"🤖 AI Answer: ", end="", flush=True)
            for sentence in llm.generate_stream(text):
                print(sentence, end="", flush=True)
                tts.speak(sentence, lang="en-us")
            print("") # 改行
        except Exception as e:
            print(f"❌ LLM生成エラー: {e}")

def main():
    # 0. AudioIO初期化 (NVIDIA Broadcast想定)
    from audio_io import AudioIO
    audio_io = AudioIO(sample_rate=16000)

    # 1. TTS初期化
    tts = KokoroTTS(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH, audio_io)

    # 2. STT初期化
    stt = WhisperSTT(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    # 3. LLM初期化
    try:
        llm = LocalLLM(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ LLM初期化失敗: {e}")
        llm = None
        print("LLM機能なしで起動します")

    if llm is None:
        class DummyLLM:
            def generate_stream(self, prompt):
                yield f"Echo: {prompt}"
        llm = DummyLLM()

    # コールバック定義 (ここで直接処理をキックする)
    def on_stt_text(text):
        if not text.strip():
            return
        # 処理スレッドにオフロードして、STTのメインループを止めないようにする
        # (ただし、会話の順番を守るならここでブロックしても良いが、音声取得が止まると困る)
        threading.Thread(target=process_response, args=(text, tts, llm)).start()

    # 5. AudioIO & STT開始
    audio_io.start()
    stt.start(audio_io, on_text_callback=on_stt_text)

    print("\n🎤 会話待機中... 話しかけると自動で返答します (Ctrl+C で終了)\n")

    # メインスレッドを維持
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 終了します")
        audio_io.stop()
        stt.is_running = False

if __name__ == "__main__":
    main()
