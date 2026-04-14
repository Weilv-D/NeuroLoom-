# NeuroLoom

NeuroLoom is a live-first visual stage for `Qwen/Qwen3.5-0.8B`.

It turns a single text conversation into a dense starfield of residual flow, grouped attention, DeltaNet memory, and decode pressure. The same live session can then be exported as `.loomtrace` and replayed frame by frame.

## What It Is

- One model only: `Qwen/Qwen3.5-0.8B`
- One core experience: `live chat -> starfield -> replay export`
- One visual language: dense star clusters, flowing arcs, logits waterfalls, and structural focus instead of abstract module boxes

NeuroLoom does not try to be a generic model debugger. It is a purpose-built stage for one Qwen profile.

## Product Shape

- `Live mode`
  A local runner starts a Qwen session, streams token-step events over WebSocket, and drives the stage in real time.
- `Replay mode`
  The same session can be exported as `.loomtrace`, reloaded locally, scrubbed on a timeline, and inspected without the runner.
- `Starfield renderer`
  Structural nodes are rendered as luminous star clusters. Sampled units appear as fine-grained stars around each block. Flow arcs pulse between them as tokens move through the model.

## Monorepo Layout

- `apps/studio`
  React + Vite + React Three Fiber frontend for the live/replay stage.
- `packages/core`
  `.loomtrace` schema, archive I/O, validator, replay engine, and renderer contract.
- `packages/official-traces`
  Qwen-only session recorder, sample trace builder, and live event definitions.
- `tools/exporters`
  Generates the official fallback replay: `qwen3.5-0.8b-sample.loomtrace`.
- `tools/runner`
  Local NeuroLoom Runner with chat initiation, WebSocket live stream, and trace export.

## Quick Start

Requirements:

- `Node.js 22+`
- `pnpm 10+`

Install and generate the official replay:

```bash
pnpm install
pnpm generate:traces
```

Run the frontend only:

```bash
pnpm dev
```

Run the local live runner in another terminal:

```bash
pnpm dev:runner
```

Then open `http://localhost:5173`.

If the runner is not available, the app falls back to the official replay bundle.

## Local Runner

The runner is the standard live transport for NeuroLoom.

Endpoints:

- `POST /v1/chat/completions`
  Starts a new session from a prompt.
- `GET /sessions`
  Lists recent runner sessions and whether replay export is ready.
- `POST /sessions/:sessionId/cancel`
  Stops a live session and seals a partial replay.
- `GET /backend/probe`
  Verifies whether the configured backend is reachable and lists the models it reports.
- `WS /live/:sessionId`
  Streams `session_started`, `token_step`, and `session_completed`.
- `GET /sessions/:sessionId/trace`
  Exports the finished session as `.loomtrace`.
- `GET /health`
  Reports runner status and mode.

By default the runner uses a deterministic synthetic text source so the stage works immediately. If you provide an OpenAI-compatible backend, the runner can adapt a real text completion endpoint while still emitting NeuroLoom-specific live events.

When `NEUROLOOM_BACKEND_URL` is set, the runner now prefers `stream: true` and bridges SSE token deltas into NeuroLoom `token_step` events in real time. Set `NEUROLOOM_BACKEND_STREAM=false` to force buffered adapter mode.

In adapter mode, NeuroLoom keeps its canonical model identity in the UI while remapping live inference requests to `NEUROLOOM_BACKEND_MODEL`. This is required for local backends that expose Qwen under provider-specific IDs such as `qwen3.5:0.8b`.

Environment variables:

- `NEUROLOOM_RUNNER_PORT`
- `NEUROLOOM_BACKEND_URL`
- `NEUROLOOM_BACKEND_API_KEY`
- `NEUROLOOM_BACKEND_MODEL`
- `NEUROLOOM_BACKEND_STREAM`
- `NEUROLOOM_BACKEND_PROVIDER`
- `NEUROLOOM_SESSION_RETENTION`

## Backend Profiles

NeuroLoom now recognizes common local Qwen backends and surfaces the detected provider in the UI and `/health`.

- `LM Studio`
  Use [tools/runner/examples/lmstudio.env.example](./tools/runner/examples/lmstudio.env.example)
- `Ollama`
  Use [tools/runner/examples/ollama.env.example](./tools/runner/examples/ollama.env.example)
- `vLLM`
  Use [tools/runner/examples/vllm.env.example](./tools/runner/examples/vllm.env.example)
- `custom`
  Set `NEUROLOOM_BACKEND_URL` to any OpenAI-compatible chat endpoint. Use `NEUROLOOM_BACKEND_PROVIDER=custom` if the URL is ambiguous.

See [docs/backends.md](./docs/backends.md) for setup notes and launch examples.

## `.loomtrace`

NeuroLoom still uses `.loomtrace` as its replay bundle format.

In this project the supported profile is intentionally narrow:

- `family: transformer`
- `model profile: Qwen3.5-0.8B`
- live session = replay session
- each generated token = one replay frame

Each frame carries:

- token text
- layer norms
- residual bands
- grouped attention scores
- attention row summary
- sampled unit stars
- top logits
- camera anchor

See [docs/loomtrace-spec.md](./docs/loomtrace-spec.md) for the profile details.

## Commands

```bash
pnpm generate:traces
pnpm validate:samples
pnpm dev
pnpm dev:runner
pnpm build
pnpm test
pnpm test:visual
```

## Scope Boundary

- Single-model only
- Language-only only
- Local runner first
- Replay export preserved
- No multi-model family switcher
- No Story Mode / Studio Mode split
- No browser ONNX rebuild path
- No arbitrary runtime capture API
