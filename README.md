# NeuroLoom

<p align="center">
  <img src="./docs/workflow.svg" alt="NeuroLoom Workflow" width="100%">
</p>

**NeuroLoom** is a cinematic, open-source 2.5D neural network execution replay interpreter. It reads controlled `.loomtrace` bundles and reconstructs modern miniature `MLP`, `CNN`, and decoder-only `Transformer` runs as replayable visual scenes.

*Read this in other languages: [English](README.md), [简体中文](README_zh.md)*

## Key Features

- **Micro SOTA Trace Generation**: Built-in definitions and exporters for modern, minimalistic architectures:
  - **Tiny MLP-Mixer**: Token and channel mixing without convolutions.
  - **Tiny ConvNeXt**: Modernized visual processing using depthwise convolutions.
  - **Tiny Llama**: Feature-rich Transformer with RoPE (Rotary Positional Embeddings), GQA (Grouped-Query Attention), and SwiGLU.
- **Cinematic Rendering**: 2.5D physical animations, frosted glass materials, and glowing refractions via React Three Fiber.
- **Trace-Driven Replay**: Reconstructs deterministic execution graphs and tensors perfectly from `.loomtrace` binary files.
- **Dual Presentation Modes**: *Story Mode* for narrative presentations and *Studio Mode* for frame-by-frame metric analysis.
- **Deep Interaction**: Isolate and freeze neural nodes to inspect raw tensors, activations, and multi-head attention weights.

## Getting Started

Make sure you have `Node.js 22+` and `pnpm 10+` installed. 

```bash
# 1. Install dependencies
pnpm install

# 2. Build the workspace packages
pnpm build

# 3. Generate official SOTA traces (MLP-Mixer, ConvNeXt, Llama)
pnpm generate:traces

# 4. Start the interactive 3D studio locally (defaults to http://localhost:5173)
pnpm dev
```

## Structure

- `apps/studio`: The WebGL/React 19 replay studio frontend.
- `packages/core`: The `.loomtrace` schema, validator, archive I/O, and replay engine.
- `packages/official-traces`: Official graph layouts, per-frame payloads, and narrative content for the three supported families.
- `tools/exporters`: Bundle generators that emit the official `.loomtrace` samples used by the studio.
- `tools/model-training`: Optional PyTorch to ONNX experiments and intermediate extraction utilities.
