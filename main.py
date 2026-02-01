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

# グローバルバッファ
stt_buffer = []
buffer_lock = threading.Lock()

def on_stt_text(text):
    """
    STTからテキストが返ってきたときのコールバック
    """
    with buffer_lock:
        stt_buffer.append(text)

def input_listener(tts, llm):
    """
    Enterキー入力を監視して、バッファの内容を読み上げる
    """
    print("\n⌨️  Enterキーを押すと、ここまでの会話に対して返答します...\n")
    while True:
        try:
            input() # Enter待機
            
            user_input = ""
            with buffer_lock:
                if stt_buffer:
                    user_input = " ".join(stt_buffer)
                    stt_buffer.clear()
            
            if user_input:
                print(f"🤔 AI考え中... User: {user_input}")
                try:
                    # ストリーミング生成で、一文ごとにTTSに投げる
                    print(f"🤖 AI Answer: ", end="", flush=True)
                    for sentence in llm.generate_stream(user_input):
                        print(sentence, end="", flush=True)
                        tts.speak(sentence, lang="en-us")
                    print("") # 改行
                except Exception as e:
                    print(f"❌ LLM生成エラー: {e}")
            else:
                print("📭 バッファは空です")
                
        except EOFError:
            break
        except KeyboardInterrupt:
            break

def main():
    # 0. AudioIO初期化 (AEC搭載)
    from audio_io import AudioIO
    audio_io = AudioIO(sample_rate=16000)

    # 1. TTS初期化 (AudioIOを注入)
    tts = KokoroTTS(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH, audio_io)

    # 2. STT初期化
    stt = WhisperSTT(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    # 3. LLM初期化
    try:
        llm = LocalLLM(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ LLM初期化失敗: {e}")
        llm = None
        print("LLM機能なしで起動します（Enterでオウム返しになります）")

    # 4. 入力監視スレッド開始
    if llm is None:
        class DummyLLM:
            def generate(self, prompt):
                return f"Echo: {prompt}"
        llm = DummyLLM()

    input_thread = threading.Thread(target=input_listener, args=(tts, llm), daemon=True)
    input_thread.start()

    # 5. AudioIO & STT開始
    audio_io.start()
    stt.start(audio_io, on_text_callback=on_stt_text)

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
