# NeuroLoom Docs

NeuroLoom is a replay-first neural network explainer for `MLP`, `CNN`, and decoder-only `Transformer` traces.

## Read This First
- [README](../README.md)
  Project definition, monorepo layout, setup commands, and current scope.
- [`.loomtrace` Specification](./loomtrace-spec.md)
  The controlled replay bundle format used by the studio and exporters.

## Product Surface
- `Story Mode`
  Guided chapter-by-chapter walkthrough of an official trace.
- `Studio Mode`
  Frame-by-frame inspection with timeline control, structure selection, render payload lens, and PNG export.

## Official Content
- `tiny-mlp-mixer`
  Patch-to-mixer fan-out, token/channel mixing, and classifier collapse.
- `tiny-convnext`
  Depthwise filtering, inverted bottlenecks, and classifier lift.
- `tiny-llama`
  RoPE, grouped-query attention, residual flow, and decode stabilization.

## Local Commands
```bash
pnpm install
pnpm generate:traces
pnpm dev
pnpm test
pnpm build
pnpm validate:samples
```

## Repo Map
- `apps/studio`
  React + Vite + React Three Fiber app.
- `packages/core`
  Schema, validator, archive I/O, replay engine, renderer contract, CLI.
- `tools/exporters`
  Official trace generators.

## Scope Boundary
NeuroLoom v1 is intentionally narrow.

- Replay-first only.
- Desktop-first.
- Three supported families only.
- No live streaming.
- No arbitrary runtime capture.
- No generic DAG renderer fallback.
