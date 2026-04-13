import type { TraceBundle, TraceFamily, TraceFrame } from "@neuroloom/core";

export const officialTraceIds = ["spiral-2d-mlp", "fashion-mnist-cnn", "tiny-gpt-style-transformer"] as const;

export type OfficialTraceId = (typeof officialTraceIds)[number];

type PayloadCatalogEntry = TraceBundle["manifest"]["payload_catalog"][number];
type GraphNode = TraceBundle["graph"]["nodes"][number];
type GraphEdge = TraceBundle["graph"]["edges"][number];
type CameraPreset = TraceBundle["manifest"]["camera_presets"][number];
type NarrativeChapter = TraceBundle["narrative"]["chapters"][number];

const visualSemantics = {
  positive: "#15f0ff",
  negative: "#ffb45b",
  focus: "#d8ff66",
  neutral: "#eef2ff",
  bloomStrength: 1.45,
  fogDensity: 0.05,
} satisfies TraceBundle["manifest"]["visual_semantics"];

export function createOfficialTraceBundles(): TraceBundle[] {
  return [createMlpTrace(), createCnnTrace(), createTransformerTrace()];
}

export function createOfficialTraceBundle(id: OfficialTraceId): TraceBundle {
  switch (id) {
    case "spiral-2d-mlp":
      return createMlpTrace();
    case "fashion-mnist-cnn":
      return createCnnTrace();
    case "tiny-gpt-style-transformer":
      return createTransformerTrace();
  }
}

export function isOfficialTraceId(id: string): id is OfficialTraceId {
  return officialTraceIds.includes(id as OfficialTraceId);
}

function createMlpTrace(): TraceBundle {
  const nodes: GraphNode[] = [
    node("input-x", "x", "input", 0, 0, -5, 1.2, 0, { role: "feature" }),
    node("input-y", "y", "input", 0, 1, -5, -1.2, 0, { role: "feature" }),
    node("hidden-a", "H1-A", "linear", 1, 0, -2, 2.2, 0, { width: 16 }),
    node("hidden-b", "H1-B", "linear", 1, 1, -2, 0, 0, { width: 16 }),
    node("hidden-c", "H1-C", "linear", 1, 2, -2, -2.2, 0, { width: 16 }),
    node("mix-a", "H2-A", "activation", 2, 0, 1.4, 1.3, 0, { activation: "gelu" }),
    node("mix-b", "H2-B", "activation", 2, 1, 1.4, -1.3, 0, { activation: "gelu" }),
    node("output", "Class", "output", 3, 0, 4.8, 0, 0, { labels: "spiral / ring" }),
    node("loss", "Loss", "loss", 4, 0, 7.4, 0, 0, { metric: "cross_entropy" }),
  ];

  const edges: GraphEdge[] = [
    edge("e-ix-ha", "input-x", "hidden-a"),
    edge("e-ix-hb", "input-x", "hidden-b"),
    edge("e-ix-hc", "input-x", "hidden-c"),
    edge("e-iy-ha", "input-y", "hidden-a"),
    edge("e-iy-hb", "input-y", "hidden-b"),
    edge("e-iy-hc", "input-y", "hidden-c"),
    edge("e-ha-ma", "hidden-a", "mix-a"),
    edge("e-ha-mb", "hidden-a", "mix-b"),
    edge("e-hb-ma", "hidden-b", "mix-a"),
    edge("e-hb-mb", "hidden-b", "mix-b"),
    edge("e-hc-ma", "hidden-c", "mix-a"),
    edge("e-hc-mb", "hidden-c", "mix-b"),
    edge("e-ma-out", "mix-a", "output"),
    edge("e-mb-out", "mix-b", "output"),
    edge("e-out-loss", "output", "loss"),
  ];

  const timeline: TraceFrame[] = [];
  const payloads = new Map<string, string>();
  const payloadCatalog: PayloadCatalogEntry[] = [];
  const frameCount = 24;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const phase = frame < 8 ? "forward" : frame === 8 ? "loss" : frame < 18 ? "backward" : "update";
    const t = frame / (frameCount - 1);
    const lossValue = 1.24 - t * 0.68;
    const confidence = 0.38 + t * 0.47;
    const gradientNorm = 0.74 - t * 0.41;
    const renderId = `mlp-render-${frame}`;
    const inspectId = `mlp-inspect-${frame}`;

    payloadCatalog.push(payloadEntry(renderId, "render"), payloadEntry(inspectId, "inspect"));
    payloads.set(
      renderId,
      JSON.stringify({
        series: [
          { label: "loss", value: round(lossValue) },
          { label: "confidence", value: round(confidence) },
          { label: "gradient", value: round(gradientNorm) },
        ],
        matrix: createMatrix(10, 10, (x, y) => Math.tanh((x - 0.5) * 2.4 + Math.sin(y * 5 + t * 4))),
        headline: "Decision boundary drift",
      }),
    );
    payloads.set(
      inspectId,
      JSON.stringify({
        headline: "Spiral classifier",
        series: [
          { label: "train loss", value: round(lossValue) },
          { label: "margin", value: round(0.18 + t * 0.29) },
          { label: "lr", value: round(0.01 - t * 0.006) },
        ],
        matrix: createMatrix(8, 8, (x, y) => Math.sin((x + y) * 4 + t * 5)),
        boundarySnapshots: [
          {
            id: "input-plane",
            label: "Input Plane",
            matrix: createMatrix(8, 8, (x, y) => Math.tanh((x - 0.5) * 2.1 + Math.sin(y * 4.4 + t * 4.2))),
          },
          {
            id: "hidden-mix",
            label: "Hidden Mix",
            matrix: createMatrix(8, 8, (x, y) => Math.sin((x * 3.2 + y * 2.4 + t * 5.4) * Math.PI)),
          },
          {
            id: "margin-map",
            label: "Margin Map",
            matrix: createMatrix(8, 8, (x, y) => Math.cos((x - y) * 3.8 + t * 3.6)),
          },
        ],
        regions: [
          { label: "outer spiral", value: round(0.42 + t * 0.23) },
          { label: "center split", value: round(0.31 + Math.sin(t * 4.4) * 0.08 + 0.08) },
          { label: "lower arc", value: round(0.26 + Math.cos(t * 3.2) * 0.07 + 0.07) },
        ],
        selectionDetails: {
          "hidden-a": detail("Hidden unit A", "Responds to the outer spiral arc.", [
            metric("activation", 0.54 + t * 0.2),
            metric("weight norm", 1.62 - t * 0.18),
          ]),
          "hidden-b": detail("Hidden unit B", "Sharpens the center split.", [
            metric("activation", 0.33 + Math.sin(t * 3) * 0.2),
            metric("sparsity", 0.42),
          ]),
          "hidden-c": detail("Hidden unit C", "Separates lower curve fragments.", [
            metric("activation", 0.47 + Math.cos(t * 4) * 0.12),
            metric("weight norm", 1.28),
          ]),
          output: detail("Classifier output", "Aggregates the mixed features into a class logit.", [
            metric("logit", -0.12 + t * 1.44),
            metric("confidence", confidence),
          ]),
        },
      }),
    );

    timeline.push({
      frame_id: frame,
      step: Math.floor(frame / 6),
      substep: frame % 6,
      phase,
      camera_anchor: frame < 6 ? "overview" : frame < 14 ? "decision-plane" : "output-focus",
      node_states: [
        nodeState("input-x", Math.sin(t * 5.4), 0.6, inspectId),
        nodeState("input-y", Math.cos(t * 4.7), 0.55, inspectId),
        nodeState("hidden-a", 0.42 + Math.sin(t * 6) * 0.35, emphasisForPhase(phase, 0.78), inspectId),
        nodeState("hidden-b", 0.15 + Math.cos(t * 7) * 0.48, emphasisForPhase(phase, 0.7), inspectId),
        nodeState("hidden-c", -0.23 + Math.sin(t * 3.5) * 0.38, emphasisForPhase(phase, 0.72), inspectId),
        nodeState("mix-a", 0.51 + Math.sin(t * 4.1) * 0.27, emphasisForPhase(phase, 0.86), inspectId),
        nodeState("mix-b", 0.18 + Math.cos(t * 5.4) * 0.26, emphasisForPhase(phase, 0.82), inspectId),
        nodeState("output", -0.22 + t * 1.35, emphasisForPhase(phase, 0.94), inspectId),
        nodeState("loss", lossValue, phase === "loss" ? 1 : phase === "backward" ? 0.86 : 0.42, inspectId),
      ],
      edge_states: edges.map((currentEdge, index) => ({
        edgeId: currentEdge.id,
        intensity: round(0.25 + Math.abs(Math.sin(t * 4 + index * 0.4)) * 0.8),
        direction: phase === "backward" ? "backward" : "forward",
        emphasis: round(0.35 + (index % 3) / 4 + (phase === "update" ? 0.1 : 0)),
      })),
      metric_refs: [
        { id: "loss", label: "Loss", value: round(lossValue) },
        { id: "confidence", label: "Confidence", value: round(confidence) },
        { id: "gradient_norm", label: "Gradient", value: round(gradientNorm) },
      ],
      payload_refs: [renderId, inspectId],
      note:
        frame < 8
          ? "Signals fan out through two hidden stages before collapsing into a single class logit."
          : frame === 8
            ? "The loss frame freezes the prediction into a scalar teaching signal."
            : frame < 18
              ? "Backward frames push the strongest pulse from the output head into earlier layers."
              : "Update frames show the network settling into a cleaner decision boundary.",
    });
  }

  return {
    manifest: {
      trace_version: "1.0.0",
      family: "mlp",
      model_id: "spiral-2d-mlp",
      dataset_id: "spiral-2d",
      title: "Spiral MLP",
      summary: "A compact multilayer perceptron learning a spiral decision boundary.",
      phase_set: ["forward", "loss", "backward", "update"],
      frame_count: frameCount,
      camera_presets: [
        camera("overview", "Overview", { x: 0, y: 1.4, z: 13 }, { x: 1.2, y: 0, z: 0 }, 34),
        camera("decision-plane", "Decision Plane", { x: 1.5, y: 3.4, z: 10 }, { x: 1.2, y: 0, z: 0 }, 30),
        camera("output-focus", "Output", { x: 5.8, y: 1.8, z: 8.6 }, { x: 4.8, y: 0, z: 0 }, 26),
      ],
      visual_semantics: visualSemantics,
      payload_catalog: payloadCatalog,
      narrative_ref: "narrative.json",
    },
    graph: {
      nodes,
      edges,
      rootNodeIds: ["input-x", "input-y"],
    },
    timeline,
    narrative: {
      intro: "The replay starts wide, then narrows onto the decision plane and finally into the output head.",
      chapters: [
        chapter(
          "input-to-hidden",
          "Input Fan-Out",
          [0, 6],
          "hidden-a",
          "Forward pulses spread raw x/y features into the first hidden layer.",
        ),
        chapter(
          "loss-anchor",
          "Loss Snapshot",
          [7, 10],
          "loss",
          "The scalar loss frame acts like a bright checkpoint before the backward pulse returns.",
        ),
        chapter(
          "parameter-settle",
          "Parameter Settle",
          [11, 23],
          "output",
          "Update frames reveal a calmer, cleaner boundary and stronger class confidence.",
        ),
      ],
    },
    payloads,
  };
}

function createCnnTrace(): TraceBundle {
  const nodes: GraphNode[] = [
    node("image", "Image", "input", 0, 0, -6, 0, 0, { resolution: "28x28" }),
    node("stage-1", "Stage 1", "stage", 1, 0, -3.8, 0, 0, { filters: 16 }),
    node("conv-1", "Conv 1", "conv", 2, 0, -1.8, 1.2, 0, { kernel: "3x3" }),
    node("pool-1", "Pool 1", "pool", 2, 1, -1.8, -1.3, 0, { pool: "2x2" }),
    node("stage-2", "Stage 2", "stage", 3, 0, 0.8, 0, 0, { filters: 32 }),
    node("conv-2", "Conv 2", "conv", 4, 0, 3.1, 1.2, 0, { kernel: "3x3" }),
    node("pool-2", "Pool 2", "pool", 4, 1, 3.1, -1.3, 0, { pool: "2x2" }),
    node("dense", "Dense Head", "dense", 5, 0, 5.6, 0, 0, { width: 128 }),
    node("output", "Class", "output", 6, 0, 8.2, 0.3, 0, { labels: "fashion" }),
    node("loss", "Loss", "loss", 7, 0, 10.2, -0.1, 0, { metric: "cross_entropy" }),
  ];

  const edges: GraphEdge[] = [
    edge("cnn-0", "image", "stage-1"),
    edge("cnn-1", "stage-1", "conv-1"),
    edge("cnn-2", "conv-1", "pool-1"),
    edge("cnn-3", "pool-1", "stage-2"),
    edge("cnn-4", "stage-2", "conv-2"),
    edge("cnn-5", "conv-2", "pool-2"),
    edge("cnn-6", "pool-2", "dense"),
    edge("cnn-7", "dense", "output"),
    edge("cnn-8", "output", "loss"),
  ];

  const timeline: TraceFrame[] = [];
  const payloads = new Map<string, string>();
  const payloadCatalog: PayloadCatalogEntry[] = [];
  const frameCount = 20;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const phase = frame < 8 ? "forward" : frame === 8 ? "loss" : frame < 16 ? "backward" : "update";
    const t = frame / (frameCount - 1);
    const accuracy = 0.52 + t * 0.35;
    const lossValue = 0.98 - t * 0.42;
    const activationScale = 0.35 + Math.sin(t * 4.2) * 0.12;
    const renderId = `cnn-render-${frame}`;
    const inspectId = `cnn-inspect-${frame}`;

    payloadCatalog.push(payloadEntry(renderId, "render"), payloadEntry(inspectId, "inspect"));
    payloads.set(
      renderId,
      JSON.stringify({
        headline: "Feature map mosaic",
        series: [
          { label: "accuracy", value: round(accuracy) },
          { label: "loss", value: round(lossValue) },
          { label: "compression", value: round(0.18 + t * 0.54) },
        ],
        matrix: createMatrix(8, 8, (x, y) => Math.sin((x * 2.7 + y * 3.4 + t * 6) * Math.PI)),
        selectionDetails: {
          "conv-1": detail("Conv stage 1", "Early filters react to edge energy and cloth folds.", [
            metric("kernel energy", 0.72 - t * 0.1),
            metric("map entropy", 0.63),
          ]),
          "conv-2": detail("Conv stage 2", "Later filters compress textures into category signatures.", [
            metric("kernel energy", 0.88 - t * 0.18),
            metric("map entropy", 0.44),
          ]),
          dense: detail("Dense head", "Final pooled evidence is compressed into class scores.", [
            metric("activation", 0.56 + t * 0.18),
            metric("sparsity", 0.48),
          ]),
        },
      }),
    );
    payloads.set(
      inspectId,
      JSON.stringify({
        headline: "Kernel microscope",
        series: [
          { label: "receptive field", value: round(0.24 + t * 0.52) },
          { label: "activation spread", value: round(0.78 - t * 0.29) },
          { label: "class margin", value: round(0.14 + t * 0.41) },
        ],
        matrix: createMatrix(6, 6, (x, y) => Math.cos((x - y) * 4 + t * 5)),
        stages: [
          {
            id: "stage-1",
            label: "Stage 1",
            matrix: createMatrix(6, 6, (x, y) => Math.sin((x * 2.4 + y * 3.1 + t * 4.8) * Math.PI)),
            channels: [
              {
                id: "conv-1-edge",
                label: "Edge",
                nodeId: "conv-1",
                matrix: createMatrix(5, 5, (x, y) => Math.cos((x - y) * 4 + t * 4.2)),
                score: round(0.52 + t * 0.12),
              },
              {
                id: "conv-1-fold",
                label: "Fold",
                nodeId: "conv-1",
                matrix: createMatrix(5, 5, (x, y) => Math.sin((x + y) * 4.4 + t * 5.6)),
                score: round(0.44 + Math.sin(t * 3.4) * 0.08),
              },
              {
                id: "pool-1-sparse",
                label: "Sparse",
                nodeId: "pool-1",
                matrix: createMatrix(5, 5, (x, y) => Math.tanh((x - 0.5) * 2.2 + Math.cos(y * 4.5 + t * 3.4))),
                score: round(0.36 + t * 0.09),
              },
            ],
          },
          {
            id: "stage-2",
            label: "Stage 2",
            matrix: createMatrix(6, 6, (x, y) => Math.cos((x * 2.2 - y * 2.9 + t * 5.4) * Math.PI)),
            channels: [
              {
                id: "conv-2-texture",
                label: "Texture",
                nodeId: "conv-2",
                matrix: createMatrix(5, 5, (x, y) => Math.sin((x * 3.2 + y * 2.1 + t * 5.8) * Math.PI)),
                score: round(0.58 + t * 0.16),
              },
              {
                id: "conv-2-silhouette",
                label: "Silhouette",
                nodeId: "conv-2",
                matrix: createMatrix(5, 5, (x, y) => Math.cos((x - y) * 5.1 + t * 4.1)),
                score: round(0.48 + Math.cos(t * 3.1) * 0.07 + 0.06),
              },
              {
                id: "pool-2-compress",
                label: "Compress",
                nodeId: "pool-2",
                matrix: createMatrix(5, 5, (x, y) => Math.tanh((x + y - 1) * 2.5 + t * 2.4)),
                score: round(0.41 + t * 0.11),
              },
            ],
          },
        ],
        topClasses: [
          { label: "ankle boot", value: round(0.48 + t * 0.26) },
          { label: "bag", value: round(0.22 - t * 0.05 + 0.04) },
          { label: "shirt", value: round(0.16 - t * 0.04 + 0.03) },
        ],
        selectionDetails: {
          image: detail("Input image", "The replay begins with the grayscale garment image entering the first stage.", [
            metric("contrast", 0.66),
            metric("mean pixel", 0.41),
          ]),
          "pool-1": detail("Pool 1", "Spatial details collapse while salient edges survive.", [
            metric("compression", 0.5 + t * 0.1),
            metric("sparsity", 0.38),
          ]),
          output: detail("Classifier output", "Class scores stabilize as texture evidence sharpens.", [
            metric("top-1", accuracy),
            metric("logit gap", 0.21 + t * 0.34),
          ]),
        },
      }),
    );

    timeline.push({
      frame_id: frame,
      step: Math.floor(frame / 5),
      substep: frame % 5,
      phase,
      camera_anchor: frame < 7 ? "overview" : frame < 13 ? "feature-wall" : "classifier",
      node_states: [
        nodeState("image", 0.42, 0.5, inspectId),
        nodeState("stage-1", activationScale, 0.62, inspectId),
        nodeState("conv-1", 0.52 + Math.sin(t * 4.3) * 0.2, emphasisForPhase(phase, 0.8), inspectId),
        nodeState("pool-1", 0.37 + Math.cos(t * 5.1) * 0.18, emphasisForPhase(phase, 0.72), inspectId),
        nodeState("stage-2", 0.44 + Math.sin(t * 3.4) * 0.15, emphasisForPhase(phase, 0.78), inspectId),
        nodeState("conv-2", 0.63 + Math.sin(t * 5.8) * 0.17, emphasisForPhase(phase, 0.9), inspectId),
        nodeState("pool-2", 0.46 + Math.cos(t * 4.2) * 0.13, emphasisForPhase(phase, 0.74), inspectId),
        nodeState("dense", 0.55 + t * 0.18, emphasisForPhase(phase, 0.88), inspectId),
        nodeState("output", 0.22 + t * 0.65, emphasisForPhase(phase, 0.96), inspectId),
        nodeState("loss", lossValue, phase === "loss" ? 1 : phase === "backward" ? 0.84 : 0.36, inspectId),
      ],
      edge_states: edges.map((currentEdge, index) => ({
        edgeId: currentEdge.id,
        intensity: round(0.3 + Math.abs(Math.cos(t * 4.7 + index)) * 0.75),
        direction: phase === "backward" ? "backward" : "forward",
        emphasis: round(0.42 + index * 0.04 + (phase === "update" ? 0.05 : 0)),
      })),
      metric_refs: [
        { id: "accuracy", label: "Accuracy", value: round(accuracy) },
        { id: "loss", label: "Loss", value: round(lossValue) },
        { id: "compression", label: "Compression", value: round(0.18 + t * 0.54) },
      ],
      payload_refs: [renderId, inspectId],
      note:
        frame < 8
          ? "Forward frames progressively squeeze wide spatial maps into denser semantic evidence."
          : frame === 8
            ? "The loss frame captures the mismatch between predicted garment class and label."
            : frame < 16
              ? "Backward frames illuminate which stages should amplify or suppress local patterns."
              : "Update frames show a calmer pipeline with stronger class separation.",
    });
  }

  return {
    manifest: {
      trace_version: "1.0.0",
      family: "cnn",
      model_id: "fashion-mnist-cnn",
      dataset_id: "fashion-mnist",
      title: "Fashion-MNIST CNN",
      summary: "A stage-based convolutional classifier that compresses clothing textures into class evidence.",
      phase_set: ["forward", "loss", "backward", "update"],
      frame_count: frameCount,
      camera_presets: [
        camera("overview", "Overview", { x: 0.4, y: 1.8, z: 15 }, { x: 2.4, y: 0, z: 0 }, 32),
        camera("feature-wall", "Feature Wall", { x: 1.8, y: 3.6, z: 10.2 }, { x: 1.8, y: 0, z: 0 }, 28),
        camera("classifier", "Classifier", { x: 8.3, y: 2.2, z: 8 }, { x: 8.3, y: 0, z: 0 }, 24),
      ],
      visual_semantics: visualSemantics,
      payload_catalog: payloadCatalog,
      narrative_ref: "narrative.json",
    },
    graph: {
      nodes,
      edges,
      rootNodeIds: ["image"],
    },
    timeline,
    narrative: {
      intro: "The camera glides across the convolutional stages, then tightens around the classifier head.",
      chapters: [
        chapter("early-filters", "Early Filters", [0, 6], "conv-1", "Initial layers react to edges, folds, and local contrast."),
        chapter(
          "spatial-compression",
          "Spatial Compression",
          [7, 12],
          "pool-2",
          "Pooling collapses resolution while preserving salient activations.",
        ),
        chapter(
          "class-evidence",
          "Class Evidence",
          [13, 19],
          "output",
          "The dense head aggregates compressed evidence into a stable garment class.",
        ),
      ],
    },
    payloads,
  };
}

function createTransformerTrace(): TraceBundle {
  const nodes: GraphNode[] = [
    node("token-bos", "<bos>", "token", 0, 0, -5.4, 2.6, 0, { token: "<bos>" }),
    node("token-neuro", "neuro", "token", 0, 1, -5.4, 0.9, 0, { token: "neuro" }),
    node("token-loom", "loom", "token", 0, 2, -5.4, -0.9, 0, { token: "loom" }),
    node("token-glows", "glows", "token", 0, 3, -5.4, -2.6, 0, { token: "glows" }),
    node("embed", "Embedding", "embedding", 1, 0, -2.7, 0, 0, { width: 128 }),
    node("attn", "Attention", "attention", 2, 0, 0.2, 1.4, 0, { heads: 4 }),
    node("residual", "Residual", "residual", 2, 1, 0.2, -1.4, 0, { stream: "add" }),
    node("mlp", "MLP", "mlp", 3, 0, 3.4, 0, 0, { expansion: 4 }),
    node("norm", "Norm", "norm", 4, 0, 6.2, 0, 0, { epsilon: "1e-5" }),
    node("logits", "Logits", "logits", 5, 0, 8.8, 0.7, 0, { vocab: 2048 }),
    node("decode", "Decode", "decode", 5, 1, 8.8, -1.2, 0, { mode: "autoregressive" }),
  ];

  const edges: GraphEdge[] = [
    edge("t-0", "token-bos", "embed"),
    edge("t-1", "token-neuro", "embed"),
    edge("t-2", "token-loom", "embed"),
    edge("t-3", "token-glows", "embed"),
    edge("t-4", "embed", "attn"),
    edge("t-5", "attn", "residual"),
    edge("t-6", "residual", "mlp"),
    edge("t-7", "mlp", "norm"),
    edge("t-8", "norm", "logits"),
    edge("t-9", "logits", "decode"),
  ];

  const timeline: TraceFrame[] = [];
  const payloads = new Map<string, string>();
  const payloadCatalog: PayloadCatalogEntry[] = [];
  const frameCount = 22;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const phase = frame < 9 ? "forward" : frame < 16 ? "decode" : frame < 20 ? "backward" : "update";
    const t = frame / (frameCount - 1);
    const entropy = 1.62 - t * 0.54;
    const confidence = 0.28 + t * 0.48;
    const renderId = `transformer-render-${frame}`;
    const inspectId = `transformer-inspect-${frame}`;
    const tokens = ["<bos>", "neuro", "loom", "glows"];
    const headMatrices = createAttentionHeads(tokens.length, t);
    const attention = averageMatrices(headMatrices);

    payloadCatalog.push(payloadEntry(renderId, "render"), payloadEntry(inspectId, "inspect"));
    payloads.set(
      renderId,
      JSON.stringify({
        headline: "Attention ribbons",
        series: [
          { label: "entropy", value: round(entropy) },
          { label: "confidence", value: round(confidence) },
          { label: "decode progress", value: round(t) },
        ],
        matrix: attention,
        tokens,
        heads: headMatrices.map((matrix, index) => ({
          id: `head-${index}`,
          label: `Head ${index + 1}`,
          matrix,
          focusTokenIndex: (index + Math.round(t * 2)) % tokens.length,
        })),
      }),
    );
    payloads.set(
      inspectId,
      JSON.stringify({
        headline: "Residual stream",
        series: [
          { label: "head sharpness", value: round(0.32 + t * 0.51) },
          { label: "residual norm", value: round(0.84 - t * 0.14) },
          { label: "top-1 prob", value: round(confidence) },
        ],
        matrix: attention,
        tokens,
        heads: headMatrices.map((matrix, index) => ({
          id: `head-${index}`,
          label: `Head ${index + 1}`,
          matrix,
          focusTokenIndex: (index + Math.round(t * 2)) % tokens.length,
          score: round(0.34 + index * 0.11 + t * 0.18),
        })),
        topTokens: [
          { token: "glows", probability: round(0.34 + t * 0.22) },
          { token: "bright", probability: round(0.22 + t * 0.12) },
          { token: "again", probability: round(0.14 - t * 0.03) },
        ],
        selectionDetails: {
          attn: detail("Multi-head attention", "Queries concentrate from a wide glow into a tighter token path.", [
            metric("head 0 max", attention[0]![1]!),
            metric("head 3 max", attention[3]![2]!),
          ]),
          residual: detail("Residual stream", "The residual band keeps information visible between the attention and MLP sublayers.", [
            metric("stream norm", 0.88 - t * 0.16),
            metric("focus", 0.36 + t * 0.3),
          ]),
          logits: detail("Logits head", "Vocabulary scores sharpen as the next token becomes predictable.", [
            metric("entropy", entropy),
            metric("top-1 prob", confidence),
          ]),
          decode: detail("Decode step", "The selected next token stabilizes at the tail end of the replay.", [
            metric("step", frame),
            metric("temperature", 0.9),
          ]),
        },
      }),
    );

    timeline.push({
      frame_id: frame,
      step: Math.floor(frame / 4),
      substep: frame % 4,
      phase,
      camera_anchor: frame < 8 ? "overview" : frame < 16 ? "attention-grid" : "decode-head",
      node_states: [
        nodeState("token-bos", 0.42, 0.4, inspectId),
        nodeState("token-neuro", 0.48 + t * 0.1, 0.58, inspectId),
        nodeState("token-loom", 0.44 + Math.sin(t * 4.1) * 0.11, 0.6, inspectId),
        nodeState("token-glows", 0.51 + Math.cos(t * 3.8) * 0.1, 0.62, inspectId),
        nodeState("embed", 0.48 + Math.sin(t * 3.2) * 0.15, emphasisForPhase(phase, 0.7), inspectId),
        nodeState("attn", 0.64 + Math.sin(t * 5.1) * 0.18, emphasisForPhase(phase, 0.95), inspectId),
        nodeState("residual", 0.53 + Math.cos(t * 4.6) * 0.15, emphasisForPhase(phase, 0.84), inspectId),
        nodeState("mlp", 0.47 + Math.sin(t * 4.4) * 0.2, emphasisForPhase(phase, 0.82), inspectId),
        nodeState("norm", 0.58 + Math.cos(t * 5.8) * 0.12, emphasisForPhase(phase, 0.8), inspectId),
        nodeState("logits", -0.18 + t * 1.36, emphasisForPhase(phase, 0.92), inspectId),
        nodeState("decode", confidence, phase === "decode" ? 1 : phase === "update" ? 0.52 : 0.7, inspectId),
      ],
      edge_states: edges.map((currentEdge, index) => ({
        edgeId: currentEdge.id,
        intensity: round(0.22 + Math.abs(Math.sin(t * 6 + index * 0.35)) * 0.92),
        direction: phase === "backward" ? "backward" : "forward",
        emphasis: round(0.38 + (index % 4) * 0.08),
      })),
      metric_refs: [
        { id: "entropy", label: "Entropy", value: round(entropy) },
        { id: "confidence", label: "Confidence", value: round(confidence) },
        { id: "decode_progress", label: "Decode", value: round(t) },
      ],
      payload_refs: [renderId, inspectId],
      note:
        frame < 9
          ? "Forward frames widen attention across the short prompt before collapsing into a residual stream."
          : frame < 16
            ? "Decode frames focus the brightest ribbons onto the next token hypothesis."
            : frame < 20
              ? "Backward frames trace the strongest learning pressure back into the attention block."
              : "Update frames settle the logits head and sharpen the final token choice.",
    });
  }

  return {
    manifest: {
      trace_version: "1.0.0",
      family: "transformer",
      model_id: "tiny-gpt-style-transformer",
      dataset_id: "tiny-prompt",
      title: "Tiny GPT-style Transformer",
      summary: "A replay of token embedding, attention, residual flow, and decode stabilization.",
      phase_set: ["forward", "decode", "backward", "update"],
      frame_count: frameCount,
      camera_presets: [
        camera("overview", "Overview", { x: 0.8, y: 2.4, z: 15 }, { x: 1.4, y: 0, z: 0 }, 32),
        camera("attention-grid", "Attention", { x: 0.4, y: 4.8, z: 10.2 }, { x: 0.2, y: 0, z: 0 }, 27),
        camera("decode-head", "Decode", { x: 9.2, y: 1.8, z: 8.4 }, { x: 8.8, y: -0.2, z: 0 }, 24),
      ],
      visual_semantics: visualSemantics,
      payload_catalog: payloadCatalog,
      narrative_ref: "narrative.json",
    },
    graph: {
      nodes,
      edges,
      rootNodeIds: ["token-bos", "token-neuro", "token-loom", "token-glows"],
    },
    timeline,
    narrative: {
      intro: "A short prompt enters the embedding rail, fans out through attention, and converges into the next-token head.",
      chapters: [
        chapter("token-rail", "Token Rail", [0, 6], "embed", "Token embeddings compress the prompt into a dense latent rail."),
        chapter(
          "attention-ribbons",
          "Attention Ribbons",
          [7, 15],
          "attn",
          "Attention heads brighten and tighten as decode confidence rises.",
        ),
        chapter("decode-head", "Decode Head", [16, 21], "decode", "The replay ends at the next-token decision point."),
      ],
    },
    payloads,
  };
}

function node(
  id: string,
  label: string,
  type: string,
  layerIndex: number,
  order: number,
  x: number,
  y: number,
  z: number,
  metadata: Record<string, string | number | boolean>,
): GraphNode {
  return {
    id,
    label,
    type,
    layerIndex,
    order,
    position: { x, y, z },
    metadata,
  };
}

function edge(id: string, source: string, target: string): GraphEdge {
  return { id, source, target, type: "flow", weight: 1 };
}

function camera(
  id: string,
  label: string,
  position: { x: number; y: number; z: number },
  target: { x: number; y: number; z: number },
  fov: number,
): CameraPreset {
  return { id, label, position, target, fov };
}

function chapter(id: string, label: string, frameRange: [number, number], defaultSelection: string, description: string): NarrativeChapter {
  return { id, label, frameRange, defaultSelection, description };
}

function payloadEntry(id: string, kind: "render" | "inspect"): PayloadCatalogEntry {
  return { id, kind, mimeType: "application/json", path: `payload/${id}.json` };
}

function metric(label: string, value: number) {
  return { label, value: round(value) };
}

function detail(title: string, blurb: string, stats: Array<{ label: string; value: number }>) {
  return { title, blurb, stats };
}

function nodeState(nodeId: string, activation: number, emphasis: number, payloadRef: string) {
  return { nodeId, activation: round(activation), emphasis: clamp(emphasis, 0, 1), payloadRef };
}

function createMatrix(sizeX: number, sizeY: number, fn: (x: number, y: number) => number) {
  return Array.from({ length: sizeY }, (_, row) =>
    Array.from({ length: sizeX }, (_, column) => round(fn(column / Math.max(sizeX - 1, 1), row / Math.max(sizeY - 1, 1)))),
  );
}

function createAttentionHeads(size: number, t: number) {
  return Array.from({ length: 4 }, (_, headIndex) =>
    Array.from({ length: size }, (_, row) =>
      Array.from({ length: size }, (_, column) => {
        const locality = Math.max(0, 0.72 - Math.abs(row - column) * (0.1 + headIndex * 0.03));
        const drift = Math.max(0, Math.sin(t * (4.2 + headIndex * 0.4) + row * 0.8 + column * (0.25 + headIndex * 0.1))) * 0.26;
        const focusBias = row === (headIndex + Math.round(t * 3)) % size ? 0.18 : 0;
        return round(clamp(0.08 + locality * (0.52 + t * 0.24) + drift + focusBias, 0, 1));
      }),
    ),
  );
}

function averageMatrices(matrices: number[][][]) {
  if (matrices.length === 0) return [];
  const rows = matrices[0]!.length;
  const columns = matrices[0]![0]!.length;

  return Array.from({ length: rows }, (_, row) =>
    Array.from({ length: columns }, (_, column) =>
      round(matrices.reduce((sum, matrix) => sum + (matrix[row]?.[column] ?? 0), 0) / matrices.length),
    ),
  );
}

function emphasisForPhase(phase: TraceFrame["phase"], base: number) {
  if (phase === "backward") return base * 0.88;
  if (phase === "update") return base * 0.64;
  if (phase === "loss") return base * 0.95;
  if (phase === "decode") return base * 1.05;
  return base;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function round(value: number) {
  return Math.round(value * 1000) / 1000;
}
