# NeuroLoom

NeuroLoom is an open-source neural network execution replay interpreter designed for `MLP`, `CNN`, and standard `GPT-style Transformer` architectures.

It doesn't directly connect to arbitrary model runtimes, nor does it attempt to be a generic graph visualization platform. Instead, NeuroLoom reads controlled `.loomtrace` replay files and reconstructs a training or inference process into a highly cinematic 2.5D animated scene, ensuring that glowing effects, data flows, camera movements, and numeric inspections align perfectly to the same frame.

## Current Status

The repository has completed the core v1 loop of NeuroLoom, showcasing advanced 2.5D physical rendering and strict testing methodologies.

**Key Achievements:**
- Cinematic **@react-spring/three** physical animations for perfectly damped transitions of nodes, edges, and data flows.
- Exquisite **meshPhysicalMaterial** integration rendering frosted-glass neural nodes with internal glowing refractions.
- Deep post-processing pipeline featuring **Cinematic Bloom, Chromatic Aberration**, and immersive ambient space particles.
- `Replay-first` workflow supporting three official model families: `mlp`, `cnn`, `transformer`.
- Three official content traces: `spiral-2d-mlp`, `fashion-mnist-cnn`, `tiny-gpt-style-transformer`.
- Dual modes: `Story Mode` and `Studio Mode`.
- Strict End-to-End visual regression testing via **Playwright** using unambiguous `data-testid` selectors.
- Comprehensive formatting and linting pipeline (ESLint, Prettier).
- Replay Engine, robust payload decoding, and trace validators.

**Planned but unfinished:**
- In-browser official trace reconstruction leveraging `ONNX Runtime Web + WebGPU`.
- Isolated "freeze" workflow for specific visual regions.
- Public documentation site and standardized open-source distribution wrappers.

NeuroLoom is a fully runnable, buildable, and demonstrable v1 MVP.

## What NeuroLoom Does

NeuroLoom focuses not on "what a neural network looks like", but "what exactly happened during a specific execution of a neural network".

It breaks down a controlled run into a set of deterministic replay frames. Every frame simultaneously captures:
- **Structural Semantics:** Nodes, edges, and hierarchies.
- **Temporal Semantics:** Step number, sub-stages, and active phase (`forward`, `backward`, etc.).
- **Visual Semantics:** What lights up, where data flows, and where the camera focuses.
- **Numerical Semantics:** Activations, attentions, feature maps, gradients, and metrics.

NeuroLoom doesn't just "draw" a network; it "performs" an execution, ensuring every visual effect leads back to verifiable raw data.

## Product Modes

### Story Mode
Designed for guided tours and presentations. It organizes an official narrative path chapter by chapter, combined with preset camera positions, annotations, and keyframe focus.

### Studio Mode
Designed for frame-by-frame detailed analysis. It provides:
- Timeline scrubbing and playback.
- Structural tree selections.
- Right-side inspectors and family-specific explorers.
- Local `.loomtrace` imports and PNG frame exports.

Selecting a node updates the inspector and actively drives the 2.5D scene (highlighting nodes, boosting adjacent edge flows, dimming out-of-focus elements, and focusing specific Transformer attention ribbons).

## Official Contents

### `spiral-2d-mlp`
Demonstrates how input features fan out to hidden layers, how loss converges into a scalar, and how backward pulses traverse returning to early layers. Studio mode allows toggling decision boundary snapshots and viewing regional response intensities.

### `fashion-mnist-cnn`
Focused on staged convolutional networks. Highlights feature map stacking, convolutional responses, pooling compressions, and classification heads. Studio mode supports switching stages and channels to inspect stage maps and classification confident scores.

### `tiny-gpt-style-transformer`
Built around standard `decoder-only Transformer` principles. Highlights token rails, attention ribbons, residual bands, and decode candidates. Studio mode enables head/token toggling, inspecting precise attention matrices and focus rows.

## `.loomtrace`

The `.loomtrace` format is the sole input protocol for NeuroLoom v1. It operates as a zipped archive describing a controlled execution run.

Core components:
- `manifest.json`
- `graph.json`
- `timeline.ndjson`
- `payload/`
- `narrative.json`

The protocol solves the "controlled interpreted replay" problem rather than handling "arbitrary runtime dumps". It enforces that replays are absolutely deterministic, supporting `mlp`, `cnn`, and `transformer` logic across sequences like `forward`, `loss`, `backward`, `update`, and `decode`.
- renderer 依赖稳定的语义节点和边，而不是运行时临时推断

更详细的协议说明见 [docs/loomtrace-spec.md](docs/loomtrace-spec.md)。

## 仓库结构

NeuroLoom 采用 monorepo 组织，分成三层。

### `apps/studio`

Web 端工作台。使用 `React 19`、`Vite`、`React Three Fiber`、`Three.js` 和 `@react-three/postprocessing` 构建，负责 Story Mode、Studio Mode、主场景渲染、时间轴和 inspector。

### `packages/core`

项目的协议与回放内核。这里定义 `.loomtrace` schema、语义校验、archive 读写、replay engine、renderer contract 和 CLI。

### `tools/exporters`

官方内容生成器。它负责构建三套官方 trace，并把样例输出到 `apps/studio/public/traces/`。

## 技术路线

NeuroLoom 当前的主渲染基线是 `WebGL + EffectComposer`。这条路线是为了保证 2.5D 场景、发光、景深、色带和后处理链的稳定性，而不是为了追求实验性 API。

`WebGPU` 在当前仓库里还没有进入主渲染路径。它只保留为后续官方 demo 的浏览器内 trace 重建能力，也就是一个渐进增强方向，而不是当前产品的前提条件。

## 本地运行

要求：

- `Node.js 22+`
- `pnpm 10+`

安装依赖：

```bash
pnpm install
```

生成官方样例：

```bash
pnpm generate:traces
```

启动本地开发环境：

```bash
pnpm dev
```

构建整个仓库：

```bash
pnpm build
```

运行测试：

```bash
pnpm test
```

校验官方样例：

```bash
pnpm validate:samples
```

## 交互说明

Studio 当前支持以下快捷操作：

- `Space`：播放 / 暂停
- `← / →`：逐帧步进
- `S`：导出当前帧 PNG

用户既可以从内置官方样例开始，也可以直接导入本地 `.loomtrace` 文件。

## 范围边界

NeuroLoom v1 的边界是明确收缩过的。

它不是：

- 通用模型运行时监控平台
- Live streaming 可视化系统
- 任意模型 hook SDK
- 通用 DAG 回退查看器
- 多人协作或远程任务管理系统

它当前专注于一件事：把三类标准模型族的一次受控运行，以美观、可交互、可审计的方式重放出来。

## 文档

- [docs/index.md](docs/index.md)
- [docs/loomtrace-spec.md](docs/loomtrace-spec.md)

项目定义保持一致：

> NeuroLoom is a neural network replay explainer.
