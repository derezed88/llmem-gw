# Quick Start

## Minimal setup (shell.py only)

```bash
source venv/bin/activate
python llmem-gw.py
```

In a second terminal:
```bash
python shell.py
```

Type `!help` to see all commands.

## With llama proxy (external apps)

Enable the llama proxy plugin first:
```bash
python llmemctl.py enable plugin_proxy_llama
```

Then start the server — it now listens on both ports:
- **8765** — shell.py (MCP protocol)
- **11434** — OpenAI/Ollama API for external apps

Point any OpenAI or Ollama-compatible app at `http://localhost:11434`.

## Essential commands

```
!model                  list available LLMs (current marked with *)
!model gemini25         switch to a different model
!llm_tools list         show per-model tool access
!tool_preview_length 0  show full tool results (no truncation)
!reset                  clear conversation history
!help                   full command reference
```

## Per-turn model switch

Prefix any message with `@ModelName` to use a different model for one turn:

```
@Win11Local extract https://www.example.com and summarize it
```

## Managing plugins and models

```bash
python llmemctl.py           # interactive menu
python llmemctl.py list      # plugin status overview
python llmemctl.py models    # model list
```

See [ADMINISTRATION.md](ADMINISTRATION.md) for full details.
See [ARCHITECTURE.md](ARCHITECTURE.md) for system internals.
