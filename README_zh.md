# NeuroLoom

<p align="center">
  <img src="./docs/workflow.svg" alt="NeuroLoom Workflow" width="100%">
</p>

**NeuroLoom** 是一个面向 `MLP`、`CNN` 和 `Transformer` (包括 Vision Transformers) 架构的开源大模型运行回放解释器。它通过高度沉浸式的 WebGL/WebGPU 图形技术，在 2.5D 环境下精准还原神经网络的前向与反向传播。

*其他语言版本：[English](README.md), [简体中文](README_zh.md)*

## 核心特性

- **电影级渲染**：利用 `@react-spring/three` 驱动物理阻尼动画，基于 `meshPhysicalMaterial` 渲染半透明毛玻璃与内部透光。
- **纯浏览器推理**：借助 ONNX Runtime Web + WebGPU，实现浏览器内的真实前向传播计算（非伪造合成数据）。
- **微型 SOTA 可视化生成**：内置 PyTorch 脚本，能够训练和导出极小版领先架构 (如 Tiny GPT、Tiny ViT)。脚本自动利用 GPU (CUDA) 加速来快速捕获网络计算图及张量中间构件。
- **Trace 驱动**：根据精准确定性的 `.loomtrace` 格式文件复现节点交互、张量流动和动画。
- **双模体验**：用于演示导读的 *Story Mode*（故事模式）与逐帧深度剖析的 *Studio Mode*（工作室模式）。
- **冻结与解剖**：支持一键选中并冻结节点，深度查看原始输入输出、梯度指标和注意力矩阵。

## 快速开始

请确保本地已安装 `Node.js 22+` 与 `pnpm 10+`。如果需要通过 PyTorch 自行生成 Trace 文件，请准备支持 `torch` 与 `onnxruntime` 的 Python 3 环境。

```bash
# 1. 安装项目 NPM 依赖
pnpm install

# 2. 生成官方内置演示模型 Trace 文件 (附带 MLP, CNN, Tiny GPT, Tiny ViT)
pnpm generate:traces

# 3. 启动本地 3D 渲染环境
pnpm dev
```

## 架构组成

- `apps/studio`: 基于 WebGL/React 19 开发的 2.5D Studio 前端应用。
- `packages/core`: `.loomtrace` 核心协议读写与解析引擎。
- `tools/exporters`: Payload 生成器与格式校验缓冲层。
- `tools/model-training`: PyTorch 微型网络训练及 ONNX 转换层（支持 GPU 加速捕捉计算状态）。
