import sys
import threading
import time

# Relative imports for package execution (python -m src.main)
from .common import config as cfg
from .stt.stt import WhisperSTT
from .tts.tts_sbv2 import SBV2TTS
from .common.audio_io import AudioIO
from .mcp.mcp_client import MCPClient
from .core.conversation import ConversationManager
from .core.voice_assistant import VoiceAssistant

try:
    from .llm.llm import LocalLLM
except ImportError:
    print("⚠️ LocalLLM module not found")
    sys.exit(1)

def main():
    print("🚀 Initializing Voice Assistant System...")

    # 1. Initialize Audio System
    audio_io = AudioIO(sample_rate=cfg.AUDIO_SAMPLE_RATE)
    audio_io.start()

    # 2. Initialize Core Components
    tts = SBV2TTS(audio_io, api_url=cfg.TTS_API_URL)
    stt = WhisperSTT(cfg.STT_MODEL_SIZE, device=cfg.STT_DEVICE, compute_type=cfg.STT_COMPUTE_TYPE)
    
    # 3. Initialize LLM
    try:
        # Use repo_id/filename if available
        llm_repo = getattr(cfg, "LLM_REPO_ID", None)
        llm_filename = getattr(cfg, "LLM_FILENAME", None)
        
        llm = LocalLLM(
            repo_id=llm_repo,
            filename=llm_filename,
            context_size=cfg.LLM_CONTEXT_SIZE
        )
    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")
        return

    # 4. Initialize MCP
    mcp_client = MCPClient(config_path=cfg.MCP_CONFIG_PATH)

    # 5. Initialize Assistant Logic
    conversation = ConversationManager(system_prompt=cfg.SYSTEM_PROMPT)
    assistant = VoiceAssistant(llm, tts, mcp_client, conversation)

    # 6. Start STT Callback
    def on_stt_text(text):
        # Dispatch to assistant in a separate thread to not block STT
        threading.Thread(target=assistant.process_input, args=(text,)).start()

    stt.start(audio_io, on_text_callback=on_stt_text)

    print("\n🎤 Ready! Speak into the microphone. (Ctrl+C to exit)\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        audio_io.stop()
        stt.is_running = False
        mcp_client.close()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
