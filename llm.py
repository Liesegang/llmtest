import os
from llama_cpp import Llama

class LocalLLM:
    def __init__(self, model_path: str, context_size: int = 512, gpu_layers: int = -1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"🧠 LLM Loading... ({model_path})")
        
        # --- 高速化設定 ---
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=gpu_layers, # -1 = GPUフル使用
            
            # 【高速化1】コンテキストサイズを減らす
            # デフォルト4096だとメモリを食うので、雑談程度なら2048で十分高速になります
            n_ctx=context_size,      
            
            # 【高速化2】バッチサイズを増やす
            # 一度に処理するトークン数。大きい方がプロンプト処理が速い（VRAMは食う）
            n_batch=1024,

            # 【高速化3】Flash Attention有効化 (爆速化の要)
            # 対応していれば劇的に速くなります
            flash_attn=True, 

            verbose=True  # ★GPU使用ログを見るためにTrueにする
        )
        
        # --- GPU使用確認ロジック ---
        print("✅ LLM Ready")

    def generate_stream(self, prompt: str, system_prompt: str = None):
        if system_prompt is None:
            system_prompt = "You are the chat assistant like Amazon echo, Siri and Google assistant. Please answer my questions as conversation in English shortly."

        formatted_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        stream = self.llm(
            formatted_prompt,
            max_tokens=1024,
            stop=["<|im_end|>"],
            stream=True,
            temperature=0.7
        )
        
        buffer = ""
        delimiters = ["。", "！", "？", "\n", "!", "?", "."]

        for output in stream:
            token = output['choices'][0]['text']
            buffer += token
            if any(d in token for d in delimiters):
                yield buffer
                buffer = ""
        
        if buffer.strip():
            yield buffer

# --- 動作確認用 ---
if __name__ == "__main__":
    # パスは環境に合わせて変更してください
    MODEL_PATH = "model_assets/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
    
    bot = LocalLLM(MODEL_PATH)
    
    print("\n--- ストリーミング生成テスト ---")
    for sentence in bot.generate_stream("PythonでFizzBuzzを書いて"):
        print(f"受信: {sentence}") # ここでTTSに sentence を投げればOK