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
                    response = llm.generate(user_input)
                    print(f"🤖 AI Answer: {response}")
                    tts.speak(response, lang="en-us")
                except Exception as e:
                    print(f"❌ LLM生成エラー: {e}")
            else:
                print("📭 バッファは空です")
                
        except EOFError:
            break
        except KeyboardInterrupt:
            break

def main():
    # 1. TTS初期化
    tts = KokoroTTS(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)

    # 2. STT初期化
    stt = WhisperSTT(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    # 3. LLM初期化
    # モデルが存在しない、またはライブラリがない場合は例外が出る可能性があるため注意
    try:
        llm = LocalLLM(LLM_MODEL_PATH)
    except Exception as e:
        print(f"⚠️ LLM初期化失敗: {e}")
        llm = None
        print("LLM機能なしで起動します（Enterでオウム返しになります）")

    # 4. 入力監視スレッド開始
    # llmがNoneの場合は簡易的にオウム返しにするか、エラーにするか。
    # ここでは簡易ダミーLLMクラスを作るか、input_listener内で分岐するかだが、
    # input_listenerを修正して対応する。
    if llm is None:
        # ダミーLLM (オウム返し)
        class DummyLLM:
            def generate(self, prompt):
                return f"Echo: {prompt}"
        llm = DummyLLM()

    input_thread = threading.Thread(target=input_listener, args=(tts, llm), daemon=True)
    input_thread.start()

    # 5. STT開始 (ブロックしない)
    stt.start(on_text_callback=on_stt_text)

    # メインスレッドを維持
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 終了します")
        stt.is_running = False # 停止フラグ (stt.py側でチェックが必要)

if __name__ == "__main__":
    main()
