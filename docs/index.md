# NeuroLoom Docs

NeuroLoom is a live-first visual stage for `Qwen/Qwen3.5-0.8B`.

## Read First

- [README](../README.md)
  Product definition, repo layout, local setup, runner, and scope boundary.
- [`.loomtrace` profile](./loomtrace-spec.md)
  The replay bundle format used for exported Qwen sessions and the built-in sample replay.

## Product Surface

- `Live stage`
  Prompt input, live token streaming, starfield motion, structural focus, and local runner status.
- `Replay scrubber`
  Timeline control, PNG export, `.loomtrace` export/import, and chapter jumps.
- `Focus panels`
  Current token, logits, structural block digest, and selection-specific details for tokens, nodes, and sampled clusters.

## Local Commands

```bash
pnpm install
pnpm generate:traces
pnpm dev
pnpm dev:runner
pnpm build
pnpm test
pnpm validate:samples
```

## Repo Map

- `apps/studio`
  Single-page Qwen starfield frontend.
- `packages/core`
  Shared archive, replay, schema, validation, and CLI.
- `packages/official-traces`
  Qwen sample trace builder and live session recorder.
- `tools/exporters`
  Emits the official fallback `.loomtrace`.
- `tools/runner`
  Local API and WebSocket server for live sessions.

## Boundary

- One model: `Qwen/Qwen3.5-0.8B`
- One family profile: `transformer` with `delta` structural nodes
- Live-first, replay-capable
- No generic model library
- No browser inference path in the app shell
