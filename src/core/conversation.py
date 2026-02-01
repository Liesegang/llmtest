
from typing import List, Dict, Any

class ConversationManager:
    """Manages the conversation history for the LLM."""
    
    def __init__(self, system_prompt: str, max_history: int = 20):
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str = None, tool_calls: List[Dict] = None):
        if not content and not tool_calls:
            return
            
        msg = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
            
        self.history.append(msg)
        self._trim_history()

    def add_tool_output(self, tool_call_id: str, content: str):
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })
        self._trim_history()

    def _trim_history(self):
        """Keeps the system prompt + last N messages."""
        if len(self.history) > self.max_history + 1:
            # Always keep index 0 (System Prompt)
            self.history = [self.history[0]] + self.history[-self.max_history:]

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
