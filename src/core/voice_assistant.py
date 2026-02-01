
import threading
import json
import time
import re
from typing import Optional, List, Dict

from src.core.conversation import ConversationManager
from src.mcp.mcp_client import MCPClient

class VoiceAssistant:
    def __init__(self, llm, tts, mcp_client: MCPClient, conversation_manager: ConversationManager):
        self.llm = llm
        self.tts = tts
        self.mcp_client = mcp_client
        self.conversation = conversation_manager
        self.lock = threading.Lock()
        
        # Cache tools definition for LLM
        self.tools_def = self.mcp_client.list_tools()

    def process_input(self, text: str):
        """
        Main entry point for processing user voice input.
        Thread-safe execution of the conversation loop.
        """
        if not text.strip():
            return

        # Acquire lock to prevent interleaved conversations
        if not self.lock.acquire(blocking=False):
            print("⚠️ Busy processing another request.")
            return

        try:
            print(f"🤔 AI考え中... User: {text}")
            self.conversation.add_user_message(text)
            self._run_conversation_loop()
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.lock.release()

    def _run_conversation_loop(self):
        """
        Executes the LLM -> Tool -> LLM loop.
        """
        while True:
            print(f"🤖 AI Answer: ", end="", flush=True)
            
            content_buffer = ""
            tool_calls_buffer = []
            
            # State for parsing explicit <tool_call> tags (Qwen-style)
            # and <think> tags (Reasoning models)
            stream_buffer = ""
            in_tool_tag = False
            in_think_tag = False
            
            # Get generator from LLM
            history = self.conversation.get_history()
            start_time = time.time()
            first_token_received = False
            
            stream_gen = self.llm.chat_stream(history, tools=self.tools_def if self.tools_def else None)

            for type_, data in stream_gen:
                if type_ == "content":
                    # [DEBUG] Print data only if not just whitespace to reduce noise, or keep raw
                    # print(f"[RAW]: {repr(data)}") 
                    
                    if not first_token_received:
                        ttft = time.time() - start_time
                        print(f" (LLM First Token: {ttft:.2f}s)")
                        first_token_received = True

                    stream_buffer += data
                    
                    # --- Think Tag Filtering ---
                    if "<think>" in stream_buffer:
                        in_think_tag = True
                    
                    if in_think_tag:
                        if "</think>" in stream_buffer:
                            # Remove thought block completely
                            stream_buffer = re.sub(r'<think>.*?</think>', '', stream_buffer, flags=re.DOTALL)
                            in_think_tag = False
                        else:
                            # Still in thought, keep buffering (do NOT output)
                            continue

                    # --- Qwen/XML Tool Call Parsing Logic ---
                    if "<tool_call>" in stream_buffer:
                        in_tool_tag = True
                    
                    if in_tool_tag:
                        if "</tool_call>" in stream_buffer:
                            # 1. Extract and Parse Tool Calls
                            tool_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', stream_buffer, re.DOTALL)
                            for block in tool_blocks:
                                self._parse_and_buffer_tool_call(block, tool_calls_buffer)
                            
                            # 2. Extract Text Content (remove tool calls)
                            text_part = re.sub(r'<tool_call>.*?</tool_call>', '', stream_buffer, flags=re.DOTALL)
                            if text_part.strip():
                                print(text_part, end="", flush=True)
                                self.tts.speak(text_part, lang="en-us")
                                content_buffer += text_part
                            
                            # 3. Reset Buffer
                            stream_buffer = ""
                            in_tool_tag = False
                    else:
                        # Encapsulate partial tag check
                        if not self._is_partial_tag(stream_buffer):
                            if stream_buffer: # Allow whitespace for spacing in console, but TTS handles strip check
                                print(stream_buffer, end="", flush=True)
                                self.tts.speak(stream_buffer, lang="en-us")
                                content_buffer += stream_buffer
                            stream_buffer = "" # Consumed
                
                elif type_ == "tool_calls":
                    tool_calls_buffer.extend(data)

            # Flush remaining buffer
            if stream_buffer and not in_tool_tag and not in_think_tag:
                 if stream_buffer:
                    print(stream_buffer, end="", flush=True)
                    self.tts.speak(stream_buffer, lang="en-us")
                    content_buffer += stream_buffer

            print("") # End of line

            # Record interaction in history
            has_tool_call = len(tool_calls_buffer) > 0
            if content_buffer or has_tool_call:
                self.conversation.add_assistant_message(content=content_buffer, tool_calls=tool_calls_buffer)

            # Exit loop if no tools to call
            if not has_tool_call:
                break

            # Execute Tools
            self._execute_tool_calls(tool_calls_buffer)

    def _parse_and_buffer_tool_call(self, json_block: str, buffer: List[Dict]) -> bool:
        try:
            # Cleanup Qwen double-brace artifact if present
            cleaned_json = json_block.strip()
            if cleaned_json.startswith("{{") and cleaned_json.endswith("}}"):
                cleaned_json = cleaned_json.replace("{{", "{").replace("}}", "}")
            
            tool_data = json.loads(cleaned_json)
            buffer.append({
                "id": f"call_{int(time.time()*1000)}",
                "type": "function",
                "function": {
                    "name": tool_data.get("name"),
                    "arguments": json.dumps(tool_data.get("arguments", {}))
                }
            })
            return True
        except Exception as e:
            print(f"❌ JSON Parse Error: {e} | Block: {repr(json_block)}")
            return False

    def _is_partial_tag(self, text: str) -> bool:
        """Check if text ends with a partial start tag (<, <tool, <think, etc)"""
        if not text:
            return False
        # Check for start of tool_call or think
        return (
            text.endswith("<") or 
            text.endswith("<t") or 
            text.endswith("<to") or 
            text.endswith("<too") or 
            text.endswith("<tool") or
            text.endswith("<th") or
            text.endswith("<thi") or
            text.endswith("<thin") or
            text.endswith("<think")
        )

    def _execute_tool_calls(self, tool_calls: List[Dict]):
        # Late import to avoid circular dependency if any (though usually safe here)
        from src.common import config as cfg
        
        for tc in tool_calls:
            fn_name = tc['function']['name']
            args_str = tc['function']['arguments']
            call_id = tc.get('id', 'unknown')
            
            print(f"\n🛠️ Calling Tool: {fn_name} args={args_str}")
            
            try:
                args = json.loads(args_str)
                result_text = self.mcp_client.call_tool(fn_name, args)
            except Exception as e:
                result_text = f"Error: {e}"
            
            # --- Safety Truncation ---
            text_str = str(result_text)
            if len(text_str) > cfg.MAX_TOOL_OUTPUT_CHARS:
                limit = cfg.MAX_TOOL_OUTPUT_CHARS
                truncated_len = len(text_str)
                text_str = text_str[:limit] + f"\n... (Truncated. Output length: {truncated_len}, Limit: {limit})"
                
            print(f"   -> Result: {text_str[:100]}...")
            
            self.conversation.add_tool_output(call_id, text_str)
