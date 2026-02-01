import asyncio
import threading
import json
import os
import sys
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self, config_path):
        self.config_path = config_path
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.sessions = {} # tool_name -> session
        self.tools = []    # LLM-ready tool definitions
        self._ready = threading.Event()
        self._shutdown = asyncio.Event()

        self.thread.start()
        print("🔌 MCP Client starting...")
        if not self._ready.wait(timeout=30):
            print("❌ MCP Connection Timed Out")
        else:
            print(f"✅ MCP Connected. Loaded {len(self.tools)} tools.")

    def _run_loop(self):
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main_lifecycle())

    async def _main_lifecycle(self):
        async with AsyncExitStack() as stack:
            try:
                config = json.load(open(self.config_path, encoding='utf-8'))
            except Exception as e:
                print(f"❌ Failed to load config: {e}")
                self._ready.set() # Unblock init
                return

            print("🔌 Connecting to servers...")
            for name, cfg in config.get("mcpServers", {}).items():
                try:
                    # Resolve command if it's 'python' to the current executable to avoid path issues
                    cmd = cfg["command"]
                    if cmd == "python":
                        cmd = sys.executable
                    # If it is not python, we trust it is in PATH or absolute.
                    # shutil.which could check it, but subprocess does that too.
                    
                    # Merge env
                    env = os.environ.copy()
                    if "env" in cfg:
                        env.update(cfg["env"])

                    print(f"   🚀 Launching: {cmd} {cfg['args']}")
                    params = StdioServerParameters(command=cmd, args=cfg["args"], env=env)
                    
                    # stdio_client doesn't easily expose stderr unless we use custom process creation
                    # But if it closes immediately, it usually raises an exception or context exit.
                    read, write = await stack.enter_async_context(stdio_client(params))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()

                    result = await session.list_tools()
                    for tool in result.tools:
                        self.sessions[tool.name] = session
                        self.tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name, 
                                "description": tool.description,
                                "parameters": tool.inputSchema
                            }
                        })
                    print(f"   ✅ [{name}] {len(result.tools)} tools")
                except Exception as e:
                    print(f"   ❌ Failed to connect to [{name}]: {e}")
                    import traceback
                    traceback.print_exc()

            self._ready.set()
            await self._shutdown.wait()

    def list_tools(self):
        return self.tools

    def call_tool(self, name, arguments):
        if name not in self.sessions:
            return f"Error: Tool '{name}' not found."
        
        session = self.sessions[name]
        future = asyncio.run_coroutine_threadsafe(session.call_tool(name, arguments), self.loop)
        try:
            result = future.result(timeout=60) # 60s timeout for tool execution
            if result.content:
                # Concatenate text content
                return "".join([c.text for c in result.content if c.type == 'text'])
            return "OK" # No content
        except Exception as e:
            return f"Error executing tool {name}: {e}"

    def close(self):
        self.loop.call_soon_threadsafe(self._shutdown.set)
