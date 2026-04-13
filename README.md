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
- `WS /live/:sessionId`
  Streams `session_started`, `token_step`, and `session_completed`.
- `GET /sessions/:sessionId/trace`
  Exports the finished session as `.loomtrace`.
- `GET /health`
  Reports runner status and mode.

By default the runner uses a deterministic synthetic text source so the stage works immediately. If you provide an OpenAI-compatible backend, the runner can adapt a real text completion endpoint while still emitting NeuroLoom-specific live events.

Environment variables:

- `NEUROLOOM_RUNNER_PORT`
- `NEUROLOOM_BACKEND_URL`
- `NEUROLOOM_BACKEND_API_KEY`
- `NEUROLOOM_BACKEND_MODEL`

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
