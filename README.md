# NeuroLoom

<p align="center">
  <img src="./docs/workflow.svg" alt="NeuroLoom Workflow" width="100%">
</p>

**NeuroLoom** is a cinematic, open-source 2.5D neural network execution replay interpreter. It visualizes the exact forward and backward passes of `MLP`, `CNN`, and `Transformer` (including Vision Transformers) models using highly polished WebGL/WebGPU graphics.

*Read this in other languages: [English](README.md), [简体中文](README_zh.md)*

## Key Features

- **Cinematic Rendering**: 2.5D physical animations, frosted glass materials, and glowing refractions via React Three Fiber.
- **In-Browser Inference**: Authentic forward passes using ONNX Runtime Web + WebGPU.
- **Micro SOTA Trace Generation**: Built-in PyTorch scripts for training and exporting minimalistic state-of-the-art architectures (e.g., Tiny GPT, Tiny ViT) with automatic GPU (CUDA) acceleration for fast graph and tensor capture.
- **Trace-Driven**: Reconstructs execution from deterministic `.loomtrace` files.
- **Dual Modes**: *Story Mode* for narrative presentations and *Studio Mode* for frame-by-frame analysis.
- **Interaction**: Isolate and freeze nodes to inspect raw tensors, activations, and attention weights.

## Getting Started

Make sure you have `Node.js 22+` and `pnpm 10+` installed. If you intend to generate new traces from PyTorch models, ensure you also have a Python 3 environment with `torch` and `onnxruntime` installed.

```bash
# 1. Install dependencies
pnpm install

# 2. Generate official sample traces (e.g., MLP, CNN, Tiny GPT, Tiny ViT)
pnpm generate:traces

# 3. Start the local studio
pnpm dev
```

## Structure

- `apps/studio`: The WebGL/React 19 visualizer.
- `packages/core`: The `.loomtrace` schema engine.
- `tools/exporters`: Trace payload generators.
- `tools/model-training`: PyTorch to ONNX model trainers, supporting GPU-accelerated intermediate value extraction.
