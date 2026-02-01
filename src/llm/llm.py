import os
from llama_cpp import Llama

class LocalLLM:
    def __init__(self, model_path: str = None, repo_id: str = None, filename: str = None, context_size: int = 512, gpu_layers: int = -1):
        source = repo_id if repo_id else model_path
        print(f"🧠 LLM Loading... ({source})")
        
        # --- 高速化設定 ---
        common_params = {
            "n_gpu_layers": gpu_layers, # -1 = GPUフル使用
            "n_ctx": context_size,      
            "n_batch": 1024,
            "flash_attn": True, 
            "verbose": True
        }

        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            **common_params
        )
        
        print("✅ LLM Ready")

    def chat_stream(self, messages, tools=None):
        """
        Chat completion with streaming. Handles both text content and tool calls.
        Yields:
          ("content", text_chunk)
          ("tool_calls", tool_calls_list)
        """
        response = self.llm.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            stream=True,
            temperature=0.7
        )
        
        buffer = ""
        delimiters = ["。", "！", "？", "\n", "!", "?", "."]
        
        # Track tool calls
        # We need to accumulate them because they come in chunks
        collected_tool_calls = {} # index -> {id, type, function: {name, arguments}}
        
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            
            # Handle Tool Calls
            if 'tool_calls' in delta:
                for tc in delta['tool_calls']:
                    index = tc.index
                    if index not in collected_tool_calls:
                        collected_tool_calls[index] = {
                            "id": "", "type": "function", "function": {"name": "", "arguments": ""}
                        }
                    
                    if "id" in tc:
                        collected_tool_calls[index]["id"] += tc.id
                    if "type" in tc:
                        collected_tool_calls[index]["type"] = tc.type
                    if "function" in tc:
                        fn = tc.function
                        if "name" in fn:
                            collected_tool_calls[index]["function"]["name"] += fn.name
                        if "arguments" in fn:
                            collected_tool_calls[index]["function"]["arguments"] += fn.arguments
                continue
            
            # Handle Content
            if 'content' in delta and delta['content']:
                token = delta['content']
                buffer += token
                if any(d in token for d in delimiters):
                    yield ("content", buffer)
                    buffer = ""
        
        # Flush buffer
        if buffer.strip():
            yield ("content", buffer)
            
        # Flush tool calls
        if collected_tool_calls:
            # Convert dict to list
            final_tool_calls = [collected_tool_calls[i] for i in sorted(collected_tool_calls.keys())]
            yield ("tool_calls", final_tool_calls)

    def generate_stream(self, prompt: str, system_prompt: str = None):
        # Legacy support
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        for type, data in self.chat_stream(messages):
            if type == "content":
                yield data

# --- 動作確認用 ---
if __name__ == "__main__":
    # パスは環境に合わせて変更してください
    MODEL_PATH = "model_assets/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
    
    bot = LocalLLM(MODEL_PATH)
    
    print("\n--- ストリーミング生成テスト ---")
    for sentence in bot.generate_stream("PythonでFizzBuzzを書いて"):
        print(f"受信: {sentence}") # ここでTTSに sentence を投げればOK