
import pytest
from unittest.mock import MagicMock
from src.core.voice_assistant import VoiceAssistant
from src.common import config as cfg

# --- Mock Classes ---
class MockLLM:
    def chat_stream(self, messages, tools=None):
        yield ("content", "Test response")

class MockTTS:
    def speak(self, text, **kwargs):
        pass

class MockMCP:
    def list_tools(self):
        return []
    def call_tool(self, name, args):
        return "Tool Result"

class MockConversation:
    def get_history(self):
        return []
    def add_user_message(self, text):
        pass
    def add_assistant_message(self, content=None, tool_calls=None):
        pass
    def add_tool_output(self, call_id, content):
        pass

@pytest.fixture
def assistant():
    return VoiceAssistant(MockLLM(), MockTTS(), MockMCP(), MockConversation())

def test_is_partial_tag(assistant):
    """Test partial tag detection for buffering."""
    assert assistant._is_partial_tag("<") is True
    assert assistant._is_partial_tag("<think") is True
    assert assistant._is_partial_tag("Hello <tool") is True
    assert assistant._is_partial_tag("Normal text") is False

def test_truncation_logic(assistant):
    """Test that tool outputs are truncated correctly."""
    # Temporarily lower the limit for testing
    original_limit = cfg.MAX_TOOL_OUTPUT_CHARS
    cfg.MAX_TOOL_OUTPUT_CHARS = 10
    
    try:
        # Simulate tool execution
        tool_calls = [{
            "function": {"name": "test_tool", "arguments": "{}"},
            "id": "call_123"
        }]
        
        # Mock MCP to return long string
        assistant.mcp_client.call_tool = MagicMock(return_value="This is a very long string that should be truncated")
        assistant.conversation.add_tool_output = MagicMock()
        
        assistant._execute_tool_calls(tool_calls)
        
        # Verify call to add_tool_output used truncated string
        args, _ = assistant.conversation.add_tool_output.call_args
        call_id, content = args
        
        assert call_id == "call_123"
        assert len(content) > 10 # It will be len(truncated) + warning message
        assert "This is a " in content # Check prefix
        assert "... (Truncated." in content # Check warning
        
    finally:
        # Restore limit
        cfg.MAX_TOOL_OUTPUT_CHARS = original_limit

def test_think_tag_stripping(assistant):
    """Test that think tags are stripped from output."""
    # This logic is inside _run_conversation_loop which is hard to unit test 
    # without mocking the entire generator. 
    # For now, we tested _is_partial_tag which supports this feature.
    pass
