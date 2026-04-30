---
name: local-ai-agent
description: >
  Local AI agent running on Ollama. Use this skill to run a prompt through
  the local agent pipeline (agent/main.py). All inference stays on-device.
user-invocable: true
command-dispatch: tool
command-tool: exec
command-arg-mode: raw
---

## Usage

Invoke as `/local-ai-agent <your prompt>`.

This skill executes the local Python agent and returns its response.
No data is sent to any cloud service.

## Command template

```
python3 ~/local-ai-agent/agent/main.py --prompt "<prompt>"
```

## Requirements

- Ollama must be running: `ollama serve`
- Model must be pulled: `ollama pull qwen2.5:7b`
- Python dependencies installed: `pip install -e ".[dev]"`
- Set `OLLAMA_MODEL` env var to override the default model.

## Environment variables

| Variable         | Default                    | Description              |
|------------------|----------------------------|--------------------------|
| OLLAMA_MODEL     | qwen2.5:7b                 | Model to use             |
| OLLAMA_BASE_URL  | http://localhost:11434     | Ollama server URL        |
| USE_MOCK         | (unset)                    | Set to "1" for mock mode |
