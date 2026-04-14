# Backend Profiles

NeuroLoom Runner speaks one live protocol to the frontend and can source tokens from different local Qwen backends.

## Supported Profiles

- `LM Studio`
  Default local OpenAI server on `http://127.0.0.1:1234`
- `Ollama`
  OpenAI-compatible bridge on `http://127.0.0.1:11434`
- `vLLM`
  OpenAI server on `http://127.0.0.1:8000`
- `custom`
  Any OpenAI-compatible chat endpoint

The runner will try to infer the provider from the URL. If that is ambiguous, set `NEUROLOOM_BACKEND_PROVIDER`.

NeuroLoom always presents the canonical profile as `Qwen/Qwen3.5-0.8B`, but in adapter mode the runner can remap that request to a provider-specific model ID through `NEUROLOOM_BACKEND_MODEL`.

## Environment Variables

- `NEUROLOOM_BACKEND_URL`
  Base URL or full `/v1/chat/completions` endpoint.
- `NEUROLOOM_BACKEND_MODEL`
  Model name sent to the backend.
- `NEUROLOOM_BACKEND_API_KEY`
  Bearer token for protected backends.
- `NEUROLOOM_BACKEND_STREAM`
  Defaults to `true`. Set to `false` to force buffered mode.
- `NEUROLOOM_BACKEND_PROVIDER`
  Optional override: `lmstudio`, `ollama`, `vllm`, or `custom`.

## LM Studio

Example env:

```bash
cp tools/runner/examples/lmstudio.env.example .env.runner
export $(grep -v '^#' .env.runner | xargs)
pnpm dev:runner
```

LM Studio should expose its OpenAI-compatible local server before NeuroLoom starts.

## Ollama

Example env:

```bash
cp tools/runner/examples/ollama.env.example .env.runner
export $(grep -v '^#' .env.runner | xargs)
pnpm dev:runner
```

Make sure the Qwen model is available to Ollama first.

## vLLM

Example env:

```bash
cp tools/runner/examples/vllm.env.example .env.runner
export $(grep -v '^#' .env.runner | xargs)
pnpm dev:runner
```

vLLM should already be serving an OpenAI-compatible endpoint for `Qwen/Qwen3.5-0.8B`.

## Health Surface

`GET /health` now reports:

- `backendProvider`
- `backendLabel`
- `backendDetectedFrom`
- `backendUrl`
- `backendEndpoint`
- `backendSetupHint`

The frontend reads those fields and shows the active provider directly in the session panel.

`GET /backend/probe` performs an active reachability check against the configured backend, resolves the corresponding models endpoint, and reports the first few model IDs. The frontend uses it for the `Probe Backend` action.
