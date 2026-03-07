# Plugin: plugin_proxy_llama

OpenAI/Ollama-compatible API proxy. Enables external chat apps (open-webui, Enchanted, LM Studio, iOS/Android apps) to connect to the agent.

## What it provides

- Configurable port (default `11434`) with auto-detecting protocol handler
- Ollama endpoints: `GET /api/version`, `GET /api/tags`, `POST /api/generate`, `POST /api/chat`
- OpenAI endpoints: `GET /v1/models`, `POST /v1/chat/completions`
- Enchanted hybrid: `/v1/api/*` paths → Ollama response format
- Streaming and non-streaming responses

## Client detection

| Signal | Protocol used |
|---|---|
| User-Agent contains "ollama" OR path `/api/*` | Ollama (NDJSON) |
| User-Agent contains "open-webui" | OpenAI (SSE) |
| Path starts with `/v1/api/` | Enchanted hybrid |
| Default | OpenAI (SSE) |

## Important behaviors

- **Model parameter is ignored** — the server uses the session's active model regardless of what the client sends
- **Client history is ignored** — server maintains its own session history
- **Client system prompt is ignored** — server uses its own `.system_prompt` files
- **All `!commands` work** — send `!help`, `!model`, `!llm_tools list`, etc. as chat messages
- Session ID format: `llama-<client-ip>`

## Dependencies

None beyond base requirements.

## Enable

```bash
python llmemctl.py enable plugin_proxy_llama
```

## Configuration

In `plugins-enabled.json` — set `llama_port` to any available port:
```json
"plugin_proxy_llama": {
  "enabled": true,
  "llama_port": 11434,
  "llama_host": "0.0.0.0"
}
```

`11434` is the default (matching the Ollama convention), but any port works. Change it if another service is already using that port.

## Test

```bash
# Ollama format  (replace 11434 with your configured llama_port)
curl http://localhost:11434/api/chat -d '{"model":"gemini25","messages":[{"role":"user","content":"hello"}],"stream":false}'

# OpenAI format
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemini25","messages":[{"role":"user","content":"hello"}],"stream":false}'
```

## Client compatibility

| App | Format | Notes |
|---|---|---|
| open-webui | OpenAI | Set base URL to `http://host:<llama_port>/v1` |
| Enchanted (iOS/macOS) | Ollama | Set host to `http://host:<llama_port>` |
| Ollama CLI/Desktop | Ollama | Set `OLLAMA_HOST=http://host:<llama_port>` |
| LM Studio | OpenAI | Set base URL to `http://host:<llama_port>/v1` |
| iOS OpenAI apps | OpenAI | Set base URL to `http://host:<llama_port>/v1` |
