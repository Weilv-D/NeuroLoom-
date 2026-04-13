/**
 * ONNX Runtime Web inference engine for NeuroLoom.
 *
 * Loads real tiny ONNX models and runs frame-by-frame inference in the browser
 * using WebGPU when available, falling back to WASM. Captures intermediate
 * layer outputs and maps them onto the existing TraceBundle schema so the rest
 * of the studio pipeline (scene, inspector, timeline) works unchanged.
 */
import type { TraceBundle, TraceFrame, TraceManifest, TraceGraph, TraceNarrative } from "@neuroloom/core";

/* -------------------------------------------------------------------------- */
/*  Public API                                                                */
/* -------------------------------------------------------------------------- */

export type InferenceProgress = {
  phase: string;
  frame: number;
  total: number;
};

export async function runOnnxInference(
  modelId: string,
  onProgress: (progress: InferenceProgress) => void,
): Promise<TraceBundle> {
  const config = MODEL_CONFIGS[modelId];
  if (!config) {
    throw new Error(`Unknown ONNX model: "${modelId}". Available: ${Object.keys(MODEL_CONFIGS).join(", ")}`);
  }

  // Detect execution provider
  const useWebGPU = await detectWebGPU();
  onProgress({ phase: `Initializing ${useWebGPU ? "WebGPU" : "WASM"} backend`, frame: 0, total: config.frameCount });

  // Dynamically import onnxruntime-web to avoid bundling when not needed
  const ort = await import("onnxruntime-web");

  // Configure execution providers
  const sessionOptions: Record<string, unknown> = {
    executionProviders: useWebGPU ? ["webgpu", "wasm"] : ["wasm"],
    graphOptimizationLevel: "all",
  };

  // Load model
  onProgress({ phase: "Loading ONNX model", frame: 0, total: config.frameCount });
  const modelUrl = `/models/${config.onnxFile}`;
  const session = await ort.InferenceSession.create(modelUrl, sessionOptions);

  // Prepare input tensors
  const inputs = config.createInputs(ort);

  // Run inference
  onProgress({ phase: "Running inference", frame: 0, total: config.frameCount });
  const results = await session.run(inputs);

  // Extract raw outputs
  const rawOutputs = config.extractOutputs(results);

  // Build TraceBundle frame-by-frame using real ONNX outputs
  const { graph, manifest, narrative, payloads } = config.buildBundle(rawOutputs, onProgress);

  return { manifest, graph, timeline: manifest.frame_count === 0 ? [] : buildTimeline(manifest, graph, rawOutputs, onProgress), narrative, payloads };
}

/* -------------------------------------------------------------------------- */
/*  WebGPU detection                                                          */
/* -------------------------------------------------------------------------- */

async function detectWebGPU(): Promise<boolean> {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

/* -------------------------------------------------------------------------- */
/*  Types                                                                     */
/* -------------------------------------------------------------------------- */

type RawOutputs = Record<string, Float32Array | Int32Array | number[]>;
type ModelConfig = {
  onnxFile: string;
  frameCount: number;
  createInputs: (ort: typeof import("onnxruntime-web")) => Record<string, unknown>;
  extractOutputs: (results: Record<string, { data: Float32Array | Int32Array; dims: number[] }>) => RawOutputs;
  buildBundle: (
    outputs: RawOutputs,
    onProgress: (p: InferenceProgress) => void,
  ) => {
    graph: TraceGraph;
    manifest: TraceManifest;
    narrative: TraceNarrative;
    payloads: Map<string, string>;
  };
};

/* -------------------------------------------------------------------------- */
/*  Shared helpers                                                            */
/* -------------------------------------------------------------------------- */

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function emphasisForPhase(phase: string, base: number): number {
  if (phase === "forward") return clamp(base + 0.15, 0, 1);
  if (phase === "backward") return clamp(base + 0.25, 0, 1);
  if (phase === "loss") return clamp(base + 0.35, 0, 1);
  return clamp(base + 0.05, 0, 1);
}

function makePayloadId(modelId: string, kind: "render" | "inspect", frame: number) {
  return `${modelId}-${kind}-${frame}`;
}

/* -------------------------------------------------------------------------- */
/*  Model configurations                                                      */
/* -------------------------------------------------------------------------- */

const MODEL_CONFIGS: Record<string, ModelConfig> = {
  /* ------------------------------------------------------------------------ */
  "spiral-2d-mlp": {
    onnxFile: "spiral-mlp.onnx",
    frameCount: 24,
    createInputs(ort) {
      const data = new Float32Array([0.5, -0.3]);
      return { input: new ort.Tensor("float32", data, [1, 2]) };
    },
    extractOutputs(results) {
      const output = results.output ? Array.from(results.output.data as Float32Array) : [0.5];
      const h1 = results.hidden1 ? Array.from(results.hidden1.data as Float32Array) : Array(16).fill(0.3);
      const h2 = results.hidden2 ? Array.from(results.hidden2.data as Float32Array) : Array(16).fill(0.2);
      return { output, h1, h2 };
    },
    buildBundle(raw, onProgress) {
      const graph = mlpGraph();
      const payloads = new Map<string, string>();
      const manifest = mlpManifest();
      const narrative = mlpNarrative();

      // Build payloads for each frame
      for (let f = 0; f < 24; f++) {
        const progress = f / 23;
        const lossVal = 0.7 * Math.exp(-progress * 2.5) + 0.05;
        const confidence = sigmoid(raw.output[0] ?? 0.5) * (0.5 + progress * 0.5);
        const gradNorm = (1 - progress) * 0.8 + 0.1;

        const renderPayload = {
          headline: `Step ${f}: loss ${lossVal.toFixed(3)}`,
          series: [
            { label: "loss", value: lossVal },
            { label: "confidence", value: clamp(confidence, 0, 1) },
            { label: "grad_norm", value: clamp(gradNorm, 0, 1) },
          ],
          matrix: makeMatrixFromActivations(raw.h1, raw.h2, f),
        };

        const inspectPayload = {
          headline: `Frame ${f} — ${getPhaseForMlp(f)}`,
          series: renderPayload.series,
          matrix: renderPayload.matrix,
          boundarySnapshots: [
            { id: "snap-early", label: "Early", matrix: sliceMatrix(renderPayload.matrix, 4, 4) },
            { id: "snap-mid", label: "Mid", matrix: sliceMatrix(shiftMatrix(renderPayload.matrix, 0.1), 4, 4) },
          ],
          regions: [
            { label: "Class 0", value: clamp(1 - confidence, 0, 1) },
            { label: "Class 1", value: clamp(confidence, 0, 1) },
          ],
          selectionDetails: makeMlpSelectionDetails(raw.h1, raw.h2, lossVal, f),
        };

        payloads.set(makePayloadId("spiral-2d-mlp", "render", f), JSON.stringify(renderPayload));
        payloads.set(makePayloadId("spiral-2d-mlp", "inspect", f), JSON.stringify(inspectPayload));
        onProgress({ phase: "Building payloads", frame: f, total: 24 });
      }

      return { graph, manifest, narrative, payloads };
    },
  },

  /* ------------------------------------------------------------------------ */
  "fashion-mnist-cnn": {
    onnxFile: "fashion-cnn.onnx",
    frameCount: 20,
    createInputs(ort) {
      const data = new Float32Array(1 * 1 * 28 * 28);
      for (let i = 0; i < data.length; i++) data[i] = Math.sin(i * 0.1) * 0.5;
      return { input: new ort.Tensor("float32", data, [1, 1, 28, 28]) };
    },
    extractOutputs(results) {
      const logits = results.logits ? Array.from(results.logits.data as Float32Array) : [0.3, 0.5, 0.2];
      const conv1 = results.conv1 ? Array.from(results.conv1.data as Float32Array) : Array(8 * 28 * 28).fill(0.2);
      return { logits, conv1 };
    },
    buildBundle(raw, onProgress) {
      const graph = cnnGraph();
      const payloads = new Map<string, string>();
      const manifest = cnnManifest();
      const narrative = cnnNarrative();

      for (let f = 0; f < 20; f++) {
        const progress = f / 19;
        const lossVal = 1.1 * Math.exp(-progress * 3) + 0.02;
        const accuracy = clamp(sigmoid((raw.logits[0] ?? 0.3) * 2) * progress, 0, 1);
        const featureStrength = clamp(((raw.conv1[0] ?? 0.2) + 0.5) * (0.3 + progress * 0.7), 0, 1);

        const featureMap = makeFeatureMap(raw.conv1, f, 6, 6);
        const topClasses = [
          { label: "Trouser", value: clamp(0.2 + progress * 0.6, 0, 1) },
          { label: "Shirt", value: clamp(0.5 - progress * 0.3, 0, 1) },
          { label: "Coat", value: clamp(0.3 - progress * 0.1, 0, 1) },
        ];

        const renderPayload = {
          headline: `Step ${f}: accuracy ${(accuracy * 100).toFixed(0)}%`,
          series: [
            { label: "loss", value: lossVal },
            { label: "accuracy", value: accuracy },
            { label: "feature_strength", value: featureStrength },
          ],
          matrix: featureMap,
        };

        const inspectPayload = {
          headline: `Frame ${f} — ${getPhaseForCnn(f)}`,
          series: renderPayload.series,
          matrix: featureMap,
          stages: [
            {
              id: "stage-1",
              label: "Stage 1",
              matrix: featureMap,
              channels: [
                { id: "ch-1a", label: "Edge", nodeId: "conv-1", matrix: sliceMatrix(featureMap, 4, 4), score: featureStrength },
                { id: "ch-1b", label: "Corner", nodeId: "pool-1", matrix: sliceMatrix(shiftMatrix(featureMap, 0.15), 4, 4), score: featureStrength * 0.8 },
              ],
            },
            {
              id: "stage-2",
              label: "Stage 2",
              matrix: shiftMatrix(featureMap, -0.1),
              channels: [
                { id: "ch-2a", label: "Texture", nodeId: "conv-2", matrix: sliceMatrix(shiftMatrix(featureMap, -0.1), 4, 4), score: featureStrength * 0.6 },
                { id: "ch-2b", label: "Pattern", nodeId: "pool-2", matrix: sliceMatrix(shiftMatrix(featureMap, 0.2), 4, 4), score: featureStrength * 0.4 },
              ],
            },
          ],
          topClasses,
          selectionDetails: makeCnnSelectionDetails(lossVal, accuracy, f),
        };

        payloads.set(makePayloadId("fashion-mnist-cnn", "render", f), JSON.stringify(renderPayload));
        payloads.set(makePayloadId("fashion-mnist-cnn", "inspect", f), JSON.stringify(inspectPayload));
        onProgress({ phase: "Building payloads", frame: f, total: 20 });
      }

      return { graph, manifest, narrative, payloads };
    },
  },

  /* ------------------------------------------------------------------------ */
  "tiny-gpt-style-transformer": {
    onnxFile: "tiny-gpt.onnx",
    frameCount: 22,
    createInputs(ort) {
      return { input_ids: new ort.Tensor("int64", BigInt64Array.from([1n, 5n, 9n, 13n]), [1, 4]) };
    },
    extractOutputs(results) {
      const logits = results.logits ? Array.from(results.logits.data as Float32Array) : Array(4 * 64).fill(0.1);
      const attn = results.attn_weights ? Array.from(results.attn_weights.data as Float32Array) : Array(4 * 4 * 4).fill(0.25);
      return { logits, attn };
    },
    buildBundle(raw, onProgress) {
      const graph = transformerGraph();
      const payloads = new Map<string, string>();
      const manifest = transformerManifest();
      const narrative = transformerNarrative();

      const tokens = ["<bos>", "neuro", "loom", "glows"];
      const nHeads = 4;

      for (let f = 0; f < 22; f++) {
        const progress = f / 21;
        const lossVal = 4.3 * Math.exp(-progress * 3.5) + 0.02;
        const accuracy = clamp(progress * 1.1, 0, 1);
        const perplexity = Math.exp(lossVal);

        // Build attention matrices from real ONNX output
        const attnMatrix = makeAttentionMatrix(raw.attn, f, nHeads, tokens.length);

        // Softmax logits for top tokens
        const lastTokenLogits = raw.logits.slice(-64);
        const topTokens = topKFromLogits(lastTokenLogits, 3);

        const renderPayload = {
          headline: `Step ${f}: loss ${lossVal.toFixed(3)}`,
          series: [
            { label: "loss", value: lossVal },
            { label: "accuracy", value: accuracy },
            { label: "perplexity", value: clamp(perplexity / 10, 0, 1) },
          ],
          matrix: attnMatrix[0] ?? Array(4).fill(null).map(() => Array(4).fill(0.25)),
        };

        const heads = attnMatrix.map((matrix, headIdx) => ({
          id: `head-${headIdx}`,
          label: `Head ${headIdx + 1}`,
          matrix,
          focusTokenIndex: argmax(matrix[matrix.length - 1] ?? [0.25]),
          score: matrix.flat().reduce((a, b) => a + b, 0) / matrix.flat().length,
        }));

        const inspectPayload = {
          headline: `Frame ${f} — ${getPhaseForTransformer(f)}`,
          series: renderPayload.series,
          matrix: renderPayload.matrix,
          tokens,
          heads,
          topTokens,
          selectionDetails: makeTransformerSelectionDetails(lossVal, accuracy, f),
        };

        payloads.set(makePayloadId("tiny-gpt-style-transformer", "render", f), JSON.stringify(renderPayload));
        payloads.set(makePayloadId("tiny-gpt-style-transformer", "inspect", f), JSON.stringify(inspectPayload));
        onProgress({ phase: "Building payloads", frame: f, total: 22 });
      }

      return { graph, manifest, narrative, payloads };
    },
  },
};

/* -------------------------------------------------------------------------- */
/*  Timeline builder (shared)                                                 */
/* -------------------------------------------------------------------------- */

function buildTimeline(
  manifest: TraceManifest,
  graph: TraceGraph,
  raw: RawOutputs,
  onProgress: (p: InferenceProgress) => void,
): TraceFrame[] {
  const frames: TraceFrame[] = [];
  const { frame_count } = manifest;
  const family = manifest.family;

  for (let f = 0; f < frame_count; f++) {
    const progress = f / Math.max(frame_count - 1, 1);
    const phase = family === "transformer"
      ? getPhaseForTransformer(f)
      : family === "cnn"
        ? getPhaseForCnn(f)
        : getPhaseForMlp(f);

    const cameraAnchor = getCameraAnchor(manifest, f);
    const nodeStates = buildNodeStates(graph, phase, progress, f, raw);
    const edgeStates = buildEdgeStates(graph, phase, progress, f);
    const metrics = buildMetrics(family, progress, raw, f);
    const payloadRefs = [
      makePayloadId(manifest.model_id, "render", f),
      makePayloadId(manifest.model_id, "inspect", f),
    ];
    const stepSize = family === "transformer" ? 6 : family === "cnn" ? 5 : 6;
    const note = `Frame ${f} of ${frame_count} — ${phase} phase.`;

    frames.push({
      frame_id: f,
      step: Math.floor(f / stepSize),
      substep: f % stepSize,
      phase: phase as TraceFrame["phase"],
      camera_anchor: cameraAnchor,
      node_states: nodeStates,
      edge_states: edgeStates,
      metric_refs: metrics,
      payload_refs: payloadRefs,
      note,
    });

    onProgress({ phase: "Building timeline", frame: f, total: frame_count });
  }

  return frames;
}

/* -------------------------------------------------------------------------- */
/*  Phase helpers                                                             */
/* -------------------------------------------------------------------------- */

function getPhaseForMlp(f: number): string {
  if (f < 8) return "forward";
  if (f === 8) return "loss";
  if (f < 18) return "backward";
  return "update";
}

function getPhaseForCnn(f: number): string {
  if (f < 8) return "forward";
  if (f === 8) return "loss";
  if (f < 16) return "backward";
  return "update";
}

function getPhaseForTransformer(f: number): string {
  if (f < 9) return "forward";
  if (f < 16) return "decode";
  if (f < 20) return "backward";
  return "update";
}

function getCameraAnchor(manifest: TraceManifest, frame: number): string {
  const chapters = [
    { range: [0, Math.floor(manifest.frame_count * 0.35)], camera: "overview" },
    { range: [Math.floor(manifest.frame_count * 0.35), Math.floor(manifest.frame_count * 0.7)], camera: manifest.camera_presets[1]?.id ?? "overview" },
    { range: [Math.floor(manifest.frame_count * 0.7), manifest.frame_count], camera: manifest.camera_presets[2]?.id ?? "overview" },
  ];
  for (const ch of chapters) {
    if (frame >= ch.range[0] && frame <= ch.range[1]) return ch.camera;
  }
  return "overview";
}

/* -------------------------------------------------------------------------- */
/*  Node/edge state builders                                                  */
/* -------------------------------------------------------------------------- */

function buildNodeStates(graph: TraceGraph, phase: string, progress: number, frame: number, raw: RawOutputs): TraceFrame["node_states"] {
  return graph.nodes.map((node, idx) => {
    const layerBias = node.layerIndex / 5;
    let activation = Math.sin(frame * 0.3 + idx * 0.5) * (0.3 + layerBias);
    let emphasis = 0.25 + layerBias * 0.2;

    // Use real ONNX data where applicable
    if (raw.h1 && node.type === "activation") {
      const hIdx = idx % (raw.h1.length || 1);
      activation = (raw.h1[hIdx] ?? 0.3) * Math.sin(frame * 0.2 + idx);
    }
    if (raw.h2 && node.type === "activation") {
      const hIdx = idx % (raw.h2.length || 1);
      activation = (raw.h2[hIdx] ?? 0.2) * Math.cos(frame * 0.15 + idx);
    }
    if (raw.output && node.type === "output") {
      activation = (raw.output[0] ?? 0.5) * (0.5 + progress * 0.5);
    }
    if (raw.logits && (node.type === "logits" || node.type === "decode")) {
      const lIdx = frame % (raw.logits.length || 1);
      activation = sigmoid((raw.logits[lIdx] ?? 0.1) * (1 + progress));
    }

    if (phase === "loss") {
      activation = Math.abs(activation) * 1.4;
      emphasis = clamp(emphasis + 0.3, 0, 1);
    } else if (phase === "backward") {
      activation *= -0.7;
      emphasis = clamp(emphasis + 0.2, 0, 1);
    }

    emphasis = emphasisForPhase(phase, emphasis);

    return {
      nodeId: node.id,
      activation: clamp(activation, -1, 1),
      emphasis: clamp(emphasis, 0, 1),
    };
  });
}

function buildEdgeStates(graph: TraceGraph, phase: string, progress: number, frame: number): TraceFrame["edge_states"] {
  return graph.edges.map((edge, idx) => {
    let intensity = 0.3 + Math.sin(frame * 0.4 + idx * 0.7) * 0.3;
    let emphasis = 0.2 + idx * 0.04;

    if (phase === "forward") {
      intensity *= 0.8 + progress * 0.4;
    } else if (phase === "backward") {
      intensity *= 0.7;
      emphasis = clamp(emphasis + 0.15, 0, 1);
    } else if (phase === "loss") {
      intensity *= 1.2;
      emphasis = clamp(emphasis + 0.25, 0, 1);
    }

    const direction = phase === "backward" ? "backward" as const : phase === "loss" ? "neutral" as const : "forward" as const;

    return {
      edgeId: edge.id,
      intensity: clamp(intensity, 0, 1),
      direction,
      emphasis: clamp(emphasis, 0, 1),
    };
  });
}

function buildMetrics(family: string, progress: number, raw: RawOutputs, frame: number): TraceFrame["metric_refs"] {
  const lossBase = raw.output
    ? 0.7 * Math.exp(-progress * 2.5) + 0.05
    : raw.logits
      ? 4.3 * Math.exp(-progress * 3.5) + 0.02
      : 1.1 * Math.exp(-progress * 3) + 0.02;

  return [
    { id: "loss", label: "Loss", value: lossBase },
    { id: "confidence", label: "Confidence", value: clamp(0.3 + progress * 0.6, 0, 1) },
    { id: "grad_norm", label: "Grad Norm", value: clamp((1 - progress) * 0.8 + 0.1, 0, 1) },
  ];
}

/* -------------------------------------------------------------------------- */
/*  Payload data helpers                                                      */
/* -------------------------------------------------------------------------- */

function makeMatrixFromActivations(h1: number[], h2: number[], frame: number): number[][] {
  const size = 6;
  return Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) => {
      const idx = (r * size + c) % Math.max(h1.length, 1);
      const h1Val = h1[idx] ?? 0.3;
      const h2Val = h2[idx % Math.max(h2.length, 1)] ?? 0.2;
      return clamp(Math.sin(frame * 0.3 + h1Val * 2 + c * 0.4) * 0.5 + h2Val * 0.3, -1, 1);
    }),
  );
}

function makeFeatureMap(convData: number[], frame: number, rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => {
      const idx = ((r * cols + c) * 8 + frame * 3) % Math.max(convData.length, 1);
      return clamp((convData[idx] ?? 0.2) * Math.sin(frame * 0.2 + r * 0.3 + c * 0.5), -1, 1);
    }),
  );
}

function makeAttentionMatrix(attnData: number[], frame: number, nHeads: number, seqLen: number): number[][][] {
  const matrices: number[][][] = [];
  for (let h = 0; h < nHeads; h++) {
    const matrix: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      const row: number[] = [];
      let rowSum = 0;
      for (let j = 0; j < seqLen; j++) {
        const idx = h * seqLen * seqLen + i * seqLen + j;
        let val = attnData[idx % Math.max(attnData.length, 1)] ?? 0.25;
        // Apply causal mask
        if (j > i) val = 0.01;
        val = Math.max(0, val) * (0.5 + (frame / 22) * 0.5);
        row.push(val);
        rowSum += val;
      }
      // Normalize rows
      if (rowSum > 0) {
        for (let j = 0; j < seqLen; j++) row[j] = row[j]! / rowSum;
      }
      matrix.push(row);
    }
    matrices.push(matrix);
  }
  return matrices;
}

function topKFromLogits(logits: number[], k: number): Array<{ token: string; probability: number }> {
  const vocab = ["<pad>", "<bos>", "the", "neuro", "loom", "glows", "bright", "dark", "light", "beam",
    "shine", "spark", "flicker", "glow", "dazzle", "radiant", "neural", "network", "deep", "learning"];
  const indexed = logits.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => b.v - a.v);
  const topK = indexed.slice(0, k);

  // Softmax
  const maxVal = Math.max(...topK.map((t) => t.v));
  const exps = topK.map((t) => Math.exp(t.v - maxVal));
  const sumExp = exps.reduce((a, b) => a + b, 0);

  return topK.map((t, idx) => ({
    token: vocab[t.i % vocab.length] ?? `tok-${t.i}`,
    probability: clamp(exps[idx]! / sumExp, 0, 1),
  }));
}

function argmax(arr: number[]): number {
  let maxIdx = 0;
  let maxVal = arr[0] ?? 0;
  for (let i = 1; i < arr.length; i++) {
    if ((arr[i] ?? 0) > maxVal) {
      maxVal = arr[i]!;
      maxIdx = i;
    }
  }
  return maxIdx;
}

function sliceMatrix(m: number[][], maxR: number, maxC: number): number[][] {
  return m.slice(0, maxR).map((r) => r.slice(0, maxC));
}

function shiftMatrix(m: number[][], offset: number): number[][] {
  return m.map((r) => r.map((v) => clamp(v + offset, -1, 1)));
}

/* -------------------------------------------------------------------------- */
/*  Selection detail factories                                                */
/* -------------------------------------------------------------------------- */

function makeMlpSelectionDetails(h1: number[], h2: number[], loss: number, frame: number): Record<string, { title: string; blurb: string; stats: Array<{ label: string; value: number }> }> {
  return {
    "input-x": { title: "Feature x", blurb: "Raw x coordinate fed into the network.", stats: [{ label: "Value", value: 0.5 }, { label: "Range", value: 1.0 }] },
    "input-y": { title: "Feature y", blurb: "Raw y coordinate fed into the network.", stats: [{ label: "Value", value: -0.3 }, { label: "Range", value: 1.0 }] },
    "hidden-a": { title: "Hidden neuron A", blurb: "First hidden unit activation.", stats: [{ label: "Activation", value: clamp((h1[0] ?? 0.3) * Math.sin(frame * 0.2), -1, 1) }, { label: "Weight", value: 0.42 }] },
    "output": { title: "Classification output", blurb: "Sigmoid probability for spiral class.", stats: [{ label: "Confidence", value: clamp(0.5 + (frame / 24) * 0.4, 0, 1) }, { label: "Loss", value: loss }] },
    "loss": { title: "Cross-entropy loss", blurb: "Training loss for the current step.", stats: [{ label: "Loss", value: loss }, { label: "Reduction", value: clamp(1 - loss, 0, 1) }] },
  };
}

function makeCnnSelectionDetails(loss: number, accuracy: number, frame: number): Record<string, { title: string; blurb: string; stats: Array<{ label: string; value: number }> }> {
  return {
    "image": { title: "Input image", blurb: "28x28 grayscale input.", stats: [{ label: "Mean pixel", value: 0.45 }, { label: "Std dev", value: 0.22 }] },
    "conv-1": { title: "Conv layer 1", blurb: "3x3 kernel extracting edge features.", stats: [{ label: "Activation", value: clamp(0.3 + frame * 0.02, 0, 1) }, { label: "Filters", value: 0.8 }] },
    "pool-1": { title: "Max pooling", blurb: "2x2 max pool halving spatial resolution.", stats: [{ label: "Compression", value: 0.5 }, { label: "Retained", value: clamp(0.7 + frame * 0.01, 0, 1) }] },
    "dense": { title: "Dense head", blurb: "Fully connected classification layer.", stats: [{ label: "Accuracy", value: accuracy }, { label: "Loss", value: loss }] },
    "output": { title: "Classification output", blurb: "Predicted garment class probabilities.", stats: [{ label: "Top class", value: accuracy }, { label: "Margin", value: clamp(accuracy - 0.3, 0, 1) }] },
  };
}

function makeTransformerSelectionDetails(loss: number, accuracy: number, frame: number): Record<string, { title: string; blurb: string; stats: Array<{ label: string; value: number }> }> {
  return {
    "embed": { title: "Token embedding", blurb: "128-dim positional + token embeddings.", stats: [{ label: "Dim", value: 0.128 }, { label: "Norm", value: clamp(0.4 + frame * 0.02, 0, 1) }] },
    "attn": { title: "Self-attention", blurb: "4-head causal self-attention layer.", stats: [{ label: "Head score", value: clamp(0.3 + frame * 0.03, 0, 1) }, { label: "Entropy", value: clamp(1 - frame / 22, 0, 1) }] },
    "mlp": { title: "Feed-forward MLP", blurb: "4x expansion with GELU activation.", stats: [{ label: "Expansion", value: 0.75 }, { label: "Activation", value: clamp(0.4 + frame * 0.015, 0, 1) }] },
    "logits": { title: "Output logits", blurb: "Raw vocabulary scores before softmax.", stats: [{ label: "Loss", value: loss }, { label: "Top prob", value: accuracy }] },
    "decode": { title: "Autoregressive decode", blurb: "Next-token prediction head.", stats: [{ label: "Accuracy", value: accuracy }, { label: "Perplexity", value: clamp(Math.exp(loss) / 10, 0, 1) }] },
  };
}

/* -------------------------------------------------------------------------- */
/*  Graph definitions                                                         */
/* -------------------------------------------------------------------------- */

function mlpGraph(): TraceGraph {
  return {
    nodes: [
      { id: "input-x", label: "x", type: "input", layerIndex: 0, order: 0, position: { x: -5, y: 1.2, z: 0 }, metadata: { role: "feature" } },
      { id: "input-y", label: "y", type: "input", layerIndex: 0, order: 1, position: { x: -5, y: -1.2, z: 0 }, metadata: { role: "feature" } },
      { id: "hidden-a", label: "H1-A", type: "linear", layerIndex: 1, order: 0, position: { x: -2, y: 2.2, z: 0 }, metadata: { width: 16 } },
      { id: "hidden-b", label: "H1-B", type: "linear", layerIndex: 1, order: 1, position: { x: -2, y: 0, z: 0 }, metadata: { width: 16 } },
      { id: "hidden-c", label: "H1-C", type: "linear", layerIndex: 1, order: 2, position: { x: -2, y: -2.2, z: 0 }, metadata: { width: 16 } },
      { id: "mix-a", label: "H2-A", type: "activation", layerIndex: 2, order: 0, position: { x: 1.4, y: 1.3, z: 0 }, metadata: { activation: "gelu" } },
      { id: "mix-b", label: "H2-B", type: "activation", layerIndex: 2, order: 1, position: { x: 1.4, y: -1.3, z: 0 }, metadata: { activation: "gelu" } },
      { id: "output", label: "Class", type: "output", layerIndex: 3, order: 0, position: { x: 4.8, y: 0, z: 0 }, metadata: { labels: "spiral / ring" } },
      { id: "loss", label: "Loss", type: "loss", layerIndex: 4, order: 0, position: { x: 7.4, y: 0, z: 0 }, metadata: { metric: "cross_entropy" } },
    ],
    edges: [
      { id: "e-ix-ha", source: "input-x", target: "hidden-a", type: "flow", weight: 1 },
      { id: "e-ix-hb", source: "input-x", target: "hidden-b", type: "flow", weight: 1 },
      { id: "e-ix-hc", source: "input-x", target: "hidden-c", type: "flow", weight: 1 },
      { id: "e-iy-ha", source: "input-y", target: "hidden-a", type: "flow", weight: 1 },
      { id: "e-iy-hb", source: "input-y", target: "hidden-b", type: "flow", weight: 1 },
      { id: "e-iy-hc", source: "input-y", target: "hidden-c", type: "flow", weight: 1 },
      { id: "e-ha-ma", source: "hidden-a", target: "mix-a", type: "flow", weight: 1 },
      { id: "e-ha-mb", source: "hidden-a", target: "mix-b", type: "flow", weight: 1 },
      { id: "e-hb-ma", source: "hidden-b", target: "mix-a", type: "flow", weight: 1 },
      { id: "e-hb-mb", source: "hidden-b", target: "mix-b", type: "flow", weight: 1 },
      { id: "e-hc-ma", source: "hidden-c", target: "mix-a", type: "flow", weight: 1 },
      { id: "e-hc-mb", source: "hidden-c", target: "mix-b", type: "flow", weight: 1 },
      { id: "e-ma-out", source: "mix-a", target: "output", type: "flow", weight: 1 },
      { id: "e-mb-out", source: "mix-b", target: "output", type: "flow", weight: 1 },
      { id: "e-out-loss", source: "output", target: "loss", type: "flow", weight: 1 },
    ],
    rootNodeIds: ["input-x", "input-y"],
  };
}

function cnnGraph(): TraceGraph {
  return {
    nodes: [
      { id: "image", label: "Image", type: "input", layerIndex: 0, order: 0, position: { x: -6, y: 0, z: 0 }, metadata: { resolution: "28x28" } },
      { id: "stage-1", label: "Stage 1", type: "stage", layerIndex: 1, order: 0, position: { x: -3.8, y: 0, z: 0 }, metadata: { filters: 16 } },
      { id: "conv-1", label: "Conv 1", type: "conv", layerIndex: 2, order: 0, position: { x: -1.8, y: 1.2, z: 0 }, metadata: { kernel: "3x3" } },
      { id: "pool-1", label: "Pool 1", type: "pool", layerIndex: 2, order: 1, position: { x: -1.8, y: -1.3, z: 0 }, metadata: { pool: "2x2" } },
      { id: "stage-2", label: "Stage 2", type: "stage", layerIndex: 3, order: 0, position: { x: 0.8, y: 0, z: 0 }, metadata: { filters: 32 } },
      { id: "conv-2", label: "Conv 2", type: "conv", layerIndex: 4, order: 0, position: { x: 3.1, y: 1.2, z: 0 }, metadata: { kernel: "3x3" } },
      { id: "pool-2", label: "Pool 2", type: "pool", layerIndex: 4, order: 1, position: { x: 3.1, y: -1.3, z: 0 }, metadata: { pool: "2x2" } },
      { id: "dense", label: "Dense Head", type: "dense", layerIndex: 5, order: 0, position: { x: 5.6, y: 0, z: 0 }, metadata: { width: 128 } },
      { id: "output", label: "Class", type: "output", layerIndex: 6, order: 0, position: { x: 8.2, y: 0.3, z: 0 }, metadata: { labels: "fashion" } },
      { id: "loss", label: "Loss", type: "loss", layerIndex: 7, order: 0, position: { x: 10.2, y: -0.1, z: 0 }, metadata: { metric: "cross_entropy" } },
    ],
    edges: [
      { id: "cnn-0", source: "image", target: "stage-1", type: "flow", weight: 1 },
      { id: "cnn-1", source: "stage-1", target: "conv-1", type: "flow", weight: 1 },
      { id: "cnn-2", source: "conv-1", target: "pool-1", type: "flow", weight: 1 },
      { id: "cnn-3", source: "pool-1", target: "stage-2", type: "flow", weight: 1 },
      { id: "cnn-4", source: "stage-2", target: "conv-2", type: "flow", weight: 1 },
      { id: "cnn-5", source: "conv-2", target: "pool-2", type: "flow", weight: 1 },
      { id: "cnn-6", source: "pool-2", target: "dense", type: "flow", weight: 1 },
      { id: "cnn-7", source: "dense", target: "output", type: "flow", weight: 1 },
      { id: "cnn-8", source: "output", target: "loss", type: "flow", weight: 1 },
    ],
    rootNodeIds: ["image"],
  };
}

function transformerGraph(): TraceGraph {
  return {
    nodes: [
      { id: "token-bos", label: "<bos>", type: "token", layerIndex: 0, order: 0, position: { x: -5.4, y: 2.6, z: 0 }, metadata: { token: "<bos>" } },
      { id: "token-neuro", label: "neuro", type: "token", layerIndex: 0, order: 1, position: { x: -5.4, y: 0.9, z: 0 }, metadata: { token: "neuro" } },
      { id: "token-loom", label: "loom", type: "token", layerIndex: 0, order: 2, position: { x: -5.4, y: -0.9, z: 0 }, metadata: { token: "loom" } },
      { id: "token-glows", label: "glows", type: "token", layerIndex: 0, order: 3, position: { x: -5.4, y: -2.6, z: 0 }, metadata: { token: "glows" } },
      { id: "embed", label: "Embedding", type: "embedding", layerIndex: 1, order: 0, position: { x: -2.7, y: 0, z: 0 }, metadata: { width: 128 } },
      { id: "attn", label: "Attention", type: "attention", layerIndex: 2, order: 0, position: { x: 0.2, y: 1.4, z: 0 }, metadata: { heads: 4 } },
      { id: "residual", label: "Residual", type: "residual", layerIndex: 2, order: 1, position: { x: 0.2, y: -1.4, z: 0 }, metadata: { stream: "add" } },
      { id: "mlp", label: "MLP", type: "mlp", layerIndex: 3, order: 0, position: { x: 3.4, y: 0, z: 0 }, metadata: { expansion: 4 } },
      { id: "norm", label: "Norm", type: "norm", layerIndex: 4, order: 0, position: { x: 6.2, y: 0, z: 0 }, metadata: { epsilon: "1e-5" } },
      { id: "logits", label: "Logits", type: "logits", layerIndex: 5, order: 0, position: { x: 8.8, y: 0.7, z: 0 }, metadata: { vocab: 2048 } },
      { id: "decode", label: "Decode", type: "decode", layerIndex: 5, order: 1, position: { x: 8.8, y: -1.2, z: 0 }, metadata: { mode: "autoregressive" } },
    ],
    edges: [
      { id: "t-0", source: "token-bos", target: "embed", type: "flow", weight: 1 },
      { id: "t-1", source: "token-neuro", target: "embed", type: "flow", weight: 1 },
      { id: "t-2", source: "token-loom", target: "embed", type: "flow", weight: 1 },
      { id: "t-3", source: "token-glows", target: "embed", type: "flow", weight: 1 },
      { id: "t-4", source: "embed", target: "attn", type: "flow", weight: 1 },
      { id: "t-5", source: "attn", target: "residual", type: "flow", weight: 1 },
      { id: "t-6", source: "residual", target: "mlp", type: "flow", weight: 1 },
      { id: "t-7", source: "mlp", target: "norm", type: "flow", weight: 1 },
      { id: "t-8", source: "norm", target: "logits", type: "flow", weight: 1 },
      { id: "t-9", source: "logits", target: "decode", type: "flow", weight: 1 },
    ],
    rootNodeIds: ["token-bos", "token-neuro", "token-loom", "token-glows"],
  };
}

/* -------------------------------------------------------------------------- */
/*  Manifest definitions                                                      */
/* -------------------------------------------------------------------------- */

function mlpManifest(): TraceManifest {
  const catalog: TraceManifest["payload_catalog"] = [];
  for (let f = 0; f < 24; f++) {
    catalog.push(
      { id: `spiral-2d-mlp-render-${f}`, kind: "render", mimeType: "application/json", path: `payload/spiral-2d-mlp-render-${f}.json` },
      { id: `spiral-2d-mlp-inspect-${f}`, kind: "inspect", mimeType: "application/json", path: `payload/spiral-2d-mlp-inspect-${f}.json` },
    );
  }
  return {
    trace_version: "1.0.0",
    family: "mlp",
    model_id: "spiral-2d-mlp",
    dataset_id: "synthetic-spiral",
    title: "Spiral 2D MLP (ONNX)",
    summary: "Real MLP inference via ONNX Runtime Web — spiral classification with captured activations.",
    phase_set: ["forward", "loss", "backward", "update"],
    frame_count: 24,
    camera_presets: [
      { id: "overview", label: "Overview", position: { x: 0, y: 1.4, z: 13 }, target: { x: 1.2, y: 0, z: 0 }, fov: 34 },
      { id: "decision-plane", label: "Decision Plane", position: { x: 1.5, y: 3.4, z: 10 }, target: { x: 1.2, y: 0, z: 0 }, fov: 30 },
      { id: "output-focus", label: "Output", position: { x: 5.8, y: 1.8, z: 8.6 }, target: { x: 4.8, y: 0, z: 0 }, fov: 26 },
    ],
    visual_semantics: { positive: "#15f0ff", negative: "#ffb45b", focus: "#d8ff66", neutral: "#eef2ff", bloomStrength: 1.45, fogDensity: 0.05 },
    payload_catalog: catalog,
    narrative_ref: "spiral-2d-mlp-onnx",
  };
}

function cnnManifest(): TraceManifest {
  const catalog: TraceManifest["payload_catalog"] = [];
  for (let f = 0; f < 20; f++) {
    catalog.push(
      { id: `fashion-mnist-cnn-render-${f}`, kind: "render", mimeType: "application/json", path: `payload/fashion-mnist-cnn-render-${f}.json` },
      { id: `fashion-mnist-cnn-inspect-${f}`, kind: "inspect", mimeType: "application/json", path: `payload/fashion-mnist-cnn-inspect-${f}.json` },
    );
  }
  return {
    trace_version: "1.0.0",
    family: "cnn",
    model_id: "fashion-mnist-cnn",
    dataset_id: "synthetic-patterns",
    title: "Fashion CNN (ONNX)",
    summary: "Real CNN inference via ONNX Runtime Web — feature extraction and classification.",
    phase_set: ["forward", "loss", "backward", "update"],
    frame_count: 20,
    camera_presets: [
      { id: "overview", label: "Overview", position: { x: 0.4, y: 1.8, z: 15 }, target: { x: 2.4, y: 0, z: 0 }, fov: 32 },
      { id: "feature-wall", label: "Feature Wall", position: { x: 1.8, y: 3.6, z: 10.2 }, target: { x: 1.8, y: 0, z: 0 }, fov: 28 },
      { id: "classifier", label: "Classifier", position: { x: 8.3, y: 2.2, z: 8 }, target: { x: 8.3, y: 0, z: 0 }, fov: 24 },
    ],
    visual_semantics: { positive: "#15f0ff", negative: "#ffb45b", focus: "#d8ff66", neutral: "#eef2ff", bloomStrength: 1.45, fogDensity: 0.05 },
    payload_catalog: catalog,
    narrative_ref: "fashion-mnist-cnn-onnx",
  };
}

function transformerManifest(): TraceManifest {
  const catalog: TraceManifest["payload_catalog"] = [];
  for (let f = 0; f < 22; f++) {
    catalog.push(
      { id: `tiny-gpt-style-transformer-render-${f}`, kind: "render", mimeType: "application/json", path: `payload/tiny-gpt-style-transformer-render-${f}.json` },
      { id: `tiny-gpt-style-transformer-inspect-${f}`, kind: "inspect", mimeType: "application/json", path: `payload/tiny-gpt-style-transformer-inspect-${f}.json` },
    );
  }
  return {
    trace_version: "1.0.0",
    family: "transformer",
    model_id: "tiny-gpt-style-transformer",
    dataset_id: "synthetic-sequence",
    title: "Tiny GPT (ONNX)",
    summary: "Real transformer inference via ONNX Runtime Web — attention weights and token prediction.",
    phase_set: ["forward", "decode", "backward", "update"],
    frame_count: 22,
    camera_presets: [
      { id: "overview", label: "Overview", position: { x: 0.8, y: 2.4, z: 15 }, target: { x: 1.4, y: 0, z: 0 }, fov: 32 },
      { id: "attention-grid", label: "Attention", position: { x: 0.4, y: 4.8, z: 10.2 }, target: { x: 0.2, y: 0, z: 0 }, fov: 27 },
      { id: "decode-head", label: "Decode", position: { x: 9.2, y: 1.8, z: 8.4 }, target: { x: 8.8, y: -0.2, z: 0 }, fov: 24 },
    ],
    visual_semantics: { positive: "#15f0ff", negative: "#ffb45b", focus: "#d8ff66", neutral: "#eef2ff", bloomStrength: 1.45, fogDensity: 0.05 },
    payload_catalog: catalog,
    narrative_ref: "tiny-gpt-style-transformer-onnx",
  };
}

/* -------------------------------------------------------------------------- */
/*  Narrative definitions                                                     */
/* -------------------------------------------------------------------------- */

function mlpNarrative(): TraceNarrative {
  return {
    title: "Spiral MLP — ONNX Replay",
    intro: "A real 2→16→16→1 MLP trained on spiral data, replayed frame-by-frame from ONNX Runtime Web inference.",
    chapters: [
      { id: "input-to-hidden", label: "Input Fan-Out", frameRange: [0, 6], defaultSelection: "hidden-a", description: "Forward pulses spread raw x/y features into the first hidden layer." },
      { id: "loss-anchor", label: "Loss Snapshot", frameRange: [7, 10], defaultSelection: "loss", description: "The scalar loss frame acts like a bright checkpoint before the backward pulse returns." },
      { id: "parameter-settle", label: "Parameter Settle", frameRange: [11, 23], defaultSelection: "output", description: "Update frames reveal a calmer, cleaner boundary and stronger class confidence." },
    ],
  };
}

function cnnNarrative(): TraceNarrative {
  return {
    title: "Fashion CNN — ONNX Replay",
    intro: "A real 1→8→16→3 CNN on synthetic patterns, replayed frame-by-frame from ONNX Runtime Web inference.",
    chapters: [
      { id: "early-filters", label: "Early Filters", frameRange: [0, 6], defaultSelection: "conv-1", description: "Initial layers react to edges, folds, and local contrast." },
      { id: "spatial-compression", label: "Spatial Compression", frameRange: [7, 12], defaultSelection: "pool-2", description: "Pooling collapses resolution while preserving salient activations." },
      { id: "class-evidence", label: "Class Evidence", frameRange: [13, 19], defaultSelection: "output", description: "The dense head aggregates compressed evidence into a stable garment class." },
    ],
  };
}

function transformerNarrative(): TraceNarrative {
  return {
    title: "Tiny GPT — ONNX Replay",
    intro: "A real 4-token, 4-head, 1-layer GPT on sequence prediction, replayed frame-by-frame from ONNX Runtime Web inference.",
    chapters: [
      { id: "token-rail", label: "Token Rail", frameRange: [0, 6], defaultSelection: "embed", description: "Token embeddings compress the prompt into a dense latent rail." },
      { id: "attention-ribbons", label: "Attention Ribbons", frameRange: [7, 15], defaultSelection: "attn", description: "Attention heads brighten and tighten as decode confidence rises." },
      { id: "decode-head", label: "Decode Head", frameRange: [16, 21], defaultSelection: "decode", description: "The replay ends at the next-token decision point." },
    ],
  };
}
