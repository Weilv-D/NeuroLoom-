import type { TraceBundle, TraceFrame } from "@neuroloom/core";

export const officialTraceIds = ["tiny-mlp-mixer", "tiny-convnext", "tiny-llama"] as const;

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
  return [createMlpMixerTrace(), createConvNextTrace(), createLlamaTrace()];
}

export function createOfficialTraceBundle(id: OfficialTraceId): TraceBundle {
  switch (id) {
    case "tiny-mlp-mixer":
      return createMlpMixerTrace();
    case "tiny-convnext":
      return createConvNextTrace();
    case "tiny-llama":
      return createLlamaTrace();
  }
}

export function isOfficialTraceId(id: string): id is OfficialTraceId {
  return officialTraceIds.includes(id as OfficialTraceId);
}

function createMlpMixerTrace(): TraceBundle {
  const modelId = "tiny-mlp-mixer";
  const frameCount = 16;
  const nodes: GraphNode[] = [
    node("patch-a", "Patch 01", "input", 0, 0, -6.4, 3.2, 0, { lane: "token" }),
    node("patch-b", "Patch 02", "input", 0, 1, -6.4, 1.1, 0, { lane: "token" }),
    node("patch-c", "Patch 03", "input", 0, 2, -6.4, -1.1, 0, { lane: "channel" }),
    node("patch-d", "Patch 04", "input", 0, 3, -6.4, -3.2, 0, { lane: "channel" }),
    node("token-mix-1", "Token Mix A", "linear", 1, 0, -2.2, 2.5, -0.4, { family: "token" }),
    node("channel-mix-1", "Channel Mix A", "activation", 1, 1, -2.2, -2.5, 0.4, { family: "channel" }),
    node("token-mix-2", "Token Mix B", "linear", 2, 0, 1.8, 2.1, -0.4, { family: "token" }),
    node("channel-mix-2", "Channel Mix B", "activation", 2, 1, 1.8, -2.1, 0.4, { family: "channel" }),
    node("head", "Classifier", "output", 3, 0, 5.4, 0, 0, { classes: 10 }),
    node("loss", "Loss", "loss", 4, 0, 8.2, 0, 0, { metric: "cross_entropy" }),
  ];
  const edges: GraphEdge[] = [
    edge("mm-1", "patch-a", "token-mix-1"),
    edge("mm-2", "patch-b", "token-mix-1"),
    edge("mm-3", "patch-c", "channel-mix-1"),
    edge("mm-4", "patch-d", "channel-mix-1"),
    edge("mm-5", "patch-a", "channel-mix-1"),
    edge("mm-6", "patch-d", "token-mix-1"),
    edge("mm-7", "token-mix-1", "token-mix-2"),
    edge("mm-8", "channel-mix-1", "channel-mix-2"),
    edge("mm-9", "token-mix-2", "head"),
    edge("mm-10", "channel-mix-2", "head"),
    edge("mm-11", "head", "loss"),
  ];

  const timeline: TraceFrame[] = [];
  const payloads = new Map<string, string>();
  const payloadCatalog: PayloadCatalogEntry[] = [];

  for (let frame = 0; frame < frameCount; frame++) {
    const phase = frame < frameCount / 2 ? "forward" : "backward";
    const local = phase === "forward" ? frame / (frameCount / 2 - 1) : (frame - frameCount / 2) / (frameCount / 2 - 1);
    const progress = frame / (frameCount - 1);
    const tokenMix = round(clamp(0.44 + Math.sin(progress * Math.PI * 1.35 + 0.18) * 0.23 + (phase === "forward" ? 0.12 : -0.04), 0, 1));
    const channelMix = round(clamp(0.38 + Math.cos(progress * Math.PI * 1.1 + 0.55) * 0.21 + (phase === "forward" ? 0.08 : 0.02), 0, 1));
    const confidence = round(clamp(0.26 + progress * 0.58 + Math.sin(frame * 0.42) * 0.08, 0, 1));
    const lossValue = round(clamp(0.88 - progress * 0.56 + (phase === "backward" ? 0.06 : 0), 0.08, 1));
    const renderId = payloadId(modelId, "render", frame);
    const inspectId = payloadId(modelId, "inspect", frame);
    const mixerMatrix = createWaveMatrix(14, 14, progress, 1.15, phase === "backward" ? -0.12 : 0.12);
    const boundaryMatrix = createWaveMatrix(10, 10, progress * 0.8 + 0.12, 0.95, -0.02);

    const renderPayload = {
      headline: phase === "forward" ? "Forward token/channel mixing" : "Backward pressure through the mixer",
      series: [
        { label: "token_mix", value: tokenMix },
        { label: "channel_mix", value: channelMix },
        { label: "confidence", value: confidence },
      ],
      matrix: mixerMatrix,
    };

    const inspectPayload = {
      headline: `Frame ${frame + 1} · ${phase === "forward" ? "Mixer build-up" : "Mixer unwind"}`,
      series: [
        { label: "loss", value: lossValue },
        { label: "token_mix", value: tokenMix },
        { label: "channel_mix", value: channelMix },
      ],
      matrix: boundaryMatrix,
      boundarySnapshots: [
        { id: "decision-early", label: "Early", matrix: sliceMatrix(boundaryMatrix, 5, 5) },
        { id: "decision-mid", label: "Mid", matrix: sliceMatrix(shiftMatrix(boundaryMatrix, 0.12), 5, 5) },
        { id: "decision-late", label: "Late", matrix: sliceMatrix(shiftMatrix(boundaryMatrix, -0.1), 5, 5) },
      ],
      regions: [
        { label: "token lane", value: tokenMix },
        { label: "channel lane", value: channelMix },
        { label: "class margin", value: confidence },
      ],
      selectionDetails: {
        "patch-a": detail("Patch 01", "One of the patch tokens entering the mixer stack.", [
          metric("Activation", 0.44 + tokenMix * 0.3),
          metric("Patch energy", 0.52 + local * 0.28),
        ]),
        "patch-b": detail("Patch 02", "Neighboring patch contribution into token mixing.", [
          metric("Activation", 0.4 + tokenMix * 0.24),
          metric("Cross-token share", tokenMix),
        ]),
        "patch-c": detail("Patch 03", "Lower patch participating in channel mixing.", [
          metric("Activation", 0.34 + channelMix * 0.28),
          metric("Channel share", channelMix),
        ]),
        "patch-d": detail("Patch 04", "Patch response feeding the lower channel lane.", [
          metric("Activation", 0.31 + channelMix * 0.3),
          metric("Residual pulse", 0.22 + (1 - local) * 0.34),
        ]),
        "token-mix-1": detail("Token Mix A", "Token mixing projects information across spatial positions.", [
          metric("Token spread", tokenMix),
          metric("Mixer gain", 0.48 + local * 0.24),
        ]),
        "channel-mix-1": detail("Channel Mix A", "Channel mixing updates each token independently along depth.", [
          metric("Channel spread", channelMix),
          metric("Gating", 0.36 + local * 0.22),
        ]),
        "token-mix-2": detail("Token Mix B", "The second token mixer sharpens the global pattern.", [
          metric("Refinement", 0.42 + tokenMix * 0.4),
          metric("Stability", confidence),
        ]),
        "channel-mix-2": detail("Channel Mix B", "Late channel mixing consolidates per-token evidence.", [
          metric("Compression", 0.28 + channelMix * 0.46),
          metric("Carry-over", 0.22 + (1 - local) * 0.34),
        ]),
        head: detail("Classifier", "The readout head aggregates both mixer lanes into a class score.", [
          metric("Confidence", confidence),
          metric("Margin", clamp(confidence - 0.18, 0, 1)),
        ]),
        loss: detail("Loss", "The scalar loss reflects how cleanly the mixed representation separates classes.", [
          metric("Loss", lossValue),
          metric("Recovery", clamp(1 - lossValue, 0, 1)),
        ]),
      },
    };

    payloads.set(renderId, JSON.stringify(renderPayload));
    payloads.set(inspectId, JSON.stringify(inspectPayload));
    payloadCatalog.push(payloadEntry(renderId, "render"), payloadEntry(inspectId, "inspect"));

    timeline.push({
      frame_id: frame,
      step: frame,
      substep: 0,
      phase,
      camera_anchor: phase === "forward" ? "mixer" : "head",
      node_states: [
        nodeState("patch-a", 0.24 + tokenMix * 0.66, emphasisForPhase(phase, 0.74), inspectId),
        nodeState("patch-b", 0.18 + tokenMix * 0.62, emphasisForPhase(phase, 0.71), inspectId),
        nodeState("patch-c", 0.12 + channelMix * 0.7, emphasisForPhase(phase, 0.66), inspectId),
        nodeState("patch-d", 0.08 + channelMix * 0.72, emphasisForPhase(phase, 0.69), inspectId),
        nodeState("token-mix-1", phase === "forward" ? tokenMix : -0.18 - tokenMix * 0.62, emphasisForPhase(phase, 0.84), inspectId),
        nodeState("channel-mix-1", phase === "forward" ? channelMix : -0.14 - channelMix * 0.58, emphasisForPhase(phase, 0.82), inspectId),
        nodeState("token-mix-2", phase === "forward" ? tokenMix * 0.92 : -0.22 - tokenMix * 0.66, emphasisForPhase(phase, 0.88), inspectId),
        nodeState("channel-mix-2", phase === "forward" ? channelMix * 0.9 : -0.18 - channelMix * 0.64, emphasisForPhase(phase, 0.86), inspectId),
        nodeState("head", confidence * (phase === "forward" ? 1 : -0.72), emphasisForPhase(phase, 0.92), inspectId),
        nodeState("loss", -lossValue * 0.88, emphasisForPhase(phase, 0.96), inspectId),
      ],
      edge_states: [
        edgeState("mm-1", phase, 0.46 + tokenMix * 0.32, 0.74),
        edgeState("mm-2", phase, 0.44 + tokenMix * 0.28, 0.72),
        edgeState("mm-3", phase, 0.42 + channelMix * 0.34, 0.7),
        edgeState("mm-4", phase, 0.38 + channelMix * 0.36, 0.7),
        edgeState("mm-5", phase, 0.34 + channelMix * 0.22, 0.6),
        edgeState("mm-6", phase, 0.36 + tokenMix * 0.24, 0.62),
        edgeState("mm-7", phase, 0.54 + tokenMix * 0.36, 0.88),
        edgeState("mm-8", phase, 0.52 + channelMix * 0.34, 0.86),
        edgeState("mm-9", phase, 0.48 + confidence * 0.34, 0.82),
        edgeState("mm-10", phase, 0.46 + confidence * 0.3, 0.8),
        edgeState("mm-11", phase, 0.42 + lossValue * 0.42, 0.94),
      ],
      metric_refs: [
        { id: "loss", label: "Loss", value: lossValue },
        { id: "confidence", label: "Confidence", value: confidence },
        { id: "token-mix", label: "Token Mix", value: tokenMix },
      ],
      payload_refs: [renderId, inspectId],
      note:
        phase === "forward"
          ? "Token and channel lanes separate, then fold back into the readout head."
          : "Backward pressure returns through both mixer lanes before fading into the patch inputs.",
    });
  }

  return {
    manifest: {
      trace_version: "1.0.0",
      family: "mlp",
      model_id: modelId,
      dataset_id: "synthetic-patch-grid",
      title: "Tiny MLP-Mixer",
      summary: "Token and channel mixing without convolutions.",
      phase_set: ["forward", "backward"],
      frame_count: frameCount,
      camera_presets: [
        camera("mixer", "Mixer Stage", { x: 0.8, y: 4.2, z: 14.2 }, { x: 0.8, y: 0, z: 0 }, 32),
        camera("head", "Classifier Head", { x: 4.8, y: 2.8, z: 12.2 }, { x: 5.4, y: 0, z: 0 }, 28),
      ],
      visual_semantics: visualSemantics,
      payload_catalog: payloadCatalog,
      narrative_ref: "narrative.json",
    },
    graph: {
      nodes,
      edges,
      rootNodeIds: ["patch-a", "patch-b", "patch-c", "patch-d"],
    },
    timeline,
    payloads,
    narrative: {
      intro: "The replay shows how MLP-Mixer alternates token mixing and channel mixing before collapsing into a single class score.",
      chapters: [
        chapter("forward-mix", "Forward", [0, 7], "token-mix-1", "Watch the token lane sweep across patches while the channel lane sharpens each patch independently."),
        chapter("backward-return", "Backward", [8, 15], "head", "The loss pulse leaves the classifier and pushes back through both mixer lanes."),
      ],
    },
  };
}

function createConvNextTrace(): TraceBundle {
  const modelId = "tiny-convnext";
  const frameCount = 14;
  const nodes: GraphNode[] = [
    node("embed", "Patchify", "input", 0, 0, -8, 0, 0, { stride: 4 }),
    node("dwconv1", "DW Conv 7x7", "conv", 1, 0, -4.2, 3, -2.2, { kernel: 7 }),
    node("norm1", "Layer Norm", "norm", 1, 1, -3.1, 0.9, -1.2, { dims: 64 }),
    node("pwconv1", "PW Conv Expand", "dense", 2, 0, -0.4, -0.1, 0, { expansion: 4 }),
    node("act1", "GELU", "activation", 2, 1, 2.1, -1.6, 1.1, { fn: "gelu" }),
    node("pwconv2", "PW Conv Project", "dense", 3, 0, 4.7, 1.7, 1.8, { kernel: 1 }),
    node("head", "Classifier", "output", 4, 0, 8.1, 0, 0, { classes: 10 }),
  ];
  const edges: GraphEdge[] = [
    edge("cx-1", "embed", "dwconv1"),
    edge("cx-2", "dwconv1", "norm1"),
    edge("cx-3", "norm1", "pwconv1"),
    edge("cx-4", "pwconv1", "act1"),
    edge("cx-5", "act1", "pwconv2"),
    edge("cx-6", "pwconv2", "head"),
  ];

  const timeline: TraceFrame[] = [];
  const payloads = new Map<string, string>();
  const payloadCatalog: PayloadCatalogEntry[] = [];

  for (let frame = 0; frame < frameCount; frame++) {
    const phase = frame < 8 ? "forward" : "backward";
    const progress = frame / (frameCount - 1);
    const featureStrength = round(clamp(0.34 + Math.sin(progress * Math.PI * 1.25 + 0.12) * 0.25 + 0.26, 0, 1));
    const bottleneck = round(clamp(0.3 + Math.cos(progress * Math.PI * 1.4 + 0.4) * 0.2 + 0.28, 0, 1));
    const accuracy = round(clamp(0.18 + progress * 0.62 + Math.sin(frame * 0.33) * 0.05, 0, 1));
    const lossValue = round(clamp(0.96 - progress * 0.66 + (phase === "backward" ? 0.08 : 0), 0.09, 1));
    const renderId = payloadId(modelId, "render", frame);
    const inspectId = payloadId(modelId, "inspect", frame);
    const featureMatrix = createWaveMatrix(12, 12, progress + 0.08, 1.35, 0.08);
    const bottleneckMatrix = shiftMatrix(createWaveMatrix(10, 10, progress * 0.9 + 0.14, 1.05, -0.04), -0.08);

    const renderPayload = {
      headline: phase === "forward" ? "Depthwise filtering and bottleneck expansion" : "Backward gradients compress through the inverted bottleneck",
      series: [
        { label: "feature_strength", value: featureStrength },
        { label: "bottleneck", value: bottleneck },
        { label: "accuracy", value: accuracy },
      ],
      matrix: featureMatrix,
    };

    const inspectPayload = {
      headline: `Frame ${frame + 1} · ConvNeXt stage replay`,
      series: [
        { label: "loss", value: lossValue },
        { label: "accuracy", value: accuracy },
        { label: "feature_strength", value: featureStrength },
      ],
      matrix: bottleneckMatrix,
      stages: [
        {
          id: "stage-stem",
          label: "Stem",
          matrix: featureMatrix,
          channels: [
            { id: "stem-edges", label: "Edge", nodeId: "dwconv1", matrix: sliceMatrix(featureMatrix, 5, 5), score: featureStrength },
            { id: "stem-norm", label: "Norm", nodeId: "norm1", matrix: sliceMatrix(shiftMatrix(featureMatrix, 0.1), 5, 5), score: featureStrength * 0.82 },
          ],
        },
        {
          id: "stage-bottleneck",
          label: "Bottleneck",
          matrix: bottleneckMatrix,
          channels: [
            { id: "expand", label: "Expand", nodeId: "pwconv1", matrix: sliceMatrix(bottleneckMatrix, 5, 5), score: bottleneck },
            { id: "activate", label: "Activate", nodeId: "act1", matrix: sliceMatrix(shiftMatrix(bottleneckMatrix, 0.14), 5, 5), score: bottleneck * 0.88 },
            { id: "project", label: "Project", nodeId: "pwconv2", matrix: sliceMatrix(shiftMatrix(bottleneckMatrix, -0.12), 5, 5), score: bottleneck * 0.8 },
          ],
        },
      ],
      topClasses: [
        { label: "coat", value: accuracy },
        { label: "shirt", value: clamp(0.54 - accuracy * 0.22, 0, 1) },
        { label: "bag", value: clamp(0.31 - progress * 0.11, 0, 1) },
      ],
      selectionDetails: {
        embed: detail("Patchify Stem", "The image is first converted into a patch grid before depthwise filtering starts.", [
          metric("Patch stride", 0.4),
          metric("Patch energy", 0.42 + featureStrength * 0.4),
        ]),
        dwconv1: detail("Depthwise 7x7", "A large kernel sweeps spatial patterns one channel at a time.", [
          metric("Feature strength", featureStrength),
          metric("Kernel radius", 0.7),
        ]),
        norm1: detail("Layer Norm", "Normalization stabilizes the depthwise response before expansion.", [
          metric("Stability", 0.42 + progress * 0.28),
          metric("Variance kept", 0.48 + featureStrength * 0.18),
        ]),
        pwconv1: detail("Pointwise Expand", "The bottleneck expands channel width before the non-linearity.", [
          metric("Expansion", bottleneck),
          metric("Width factor", 0.8),
        ]),
        act1: detail("GELU", "The activation selects which expanded channels survive projection.", [
          metric("Activation", 0.32 + bottleneck * 0.46),
          metric("Selectivity", featureStrength),
        ]),
        pwconv2: detail("Pointwise Project", "The projected channels compress back into a compact residual-friendly state.", [
          metric("Projection", 0.34 + bottleneck * 0.42),
          metric("Retention", 0.28 + accuracy * 0.38),
        ]),
        head: detail("Classifier", "The pooled representation resolves into a stable garment label.", [
          metric("Accuracy", accuracy),
          metric("Loss", lossValue),
        ]),
      },
    };

    payloads.set(renderId, JSON.stringify(renderPayload));
    payloads.set(inspectId, JSON.stringify(inspectPayload));
    payloadCatalog.push(payloadEntry(renderId, "render"), payloadEntry(inspectId, "inspect"));

    timeline.push({
      frame_id: frame,
      step: frame,
      substep: 0,
      phase,
      camera_anchor: phase === "forward" ? "overview" : "head",
      node_states: [
        nodeState("embed", 0.22 + featureStrength * 0.56, emphasisForPhase(phase, 0.68), inspectId),
        nodeState("dwconv1", phase === "forward" ? featureStrength : -0.14 - featureStrength * 0.58, emphasisForPhase(phase, 0.88), inspectId),
        nodeState("norm1", phase === "forward" ? featureStrength * 0.82 : -0.1 - featureStrength * 0.5, emphasisForPhase(phase, 0.76), inspectId),
        nodeState("pwconv1", phase === "forward" ? bottleneck : -0.16 - bottleneck * 0.56, emphasisForPhase(phase, 0.9), inspectId),
        nodeState("act1", phase === "forward" ? bottleneck * 0.88 : -0.2 - bottleneck * 0.5, emphasisForPhase(phase, 0.86), inspectId),
        nodeState("pwconv2", phase === "forward" ? bottleneck * 0.8 : -0.22 - bottleneck * 0.54, emphasisForPhase(phase, 0.84), inspectId),
        nodeState("head", accuracy * (phase === "forward" ? 1 : -0.66), emphasisForPhase(phase, 0.94), inspectId),
      ],
      edge_states: [
        edgeState("cx-1", phase, 0.46 + featureStrength * 0.26, 0.72),
        edgeState("cx-2", phase, 0.42 + featureStrength * 0.2, 0.68),
        edgeState("cx-3", phase, 0.48 + bottleneck * 0.28, 0.8),
        edgeState("cx-4", phase, 0.5 + bottleneck * 0.3, 0.84),
        edgeState("cx-5", phase, 0.52 + bottleneck * 0.34, 0.86),
        edgeState("cx-6", phase, 0.48 + accuracy * 0.3, 0.9),
      ],
      metric_refs: [
        { id: "loss", label: "Loss", value: lossValue },
        { id: "accuracy", label: "Accuracy", value: accuracy },
        { id: "feature", label: "Feature", value: featureStrength },
      ],
      payload_refs: [renderId, inspectId],
      note:
        phase === "forward"
          ? "Depthwise filters first isolate local structure, then the bottleneck expands and compresses it."
          : "The backward phase tightens the classifier head and squeezes gradients through the bottleneck.",
    });
  }

  return {
    manifest: {
      trace_version: "1.0.0",
      family: "cnn",
      model_id: modelId,
      dataset_id: "synthetic-fashion-grid",
      title: "Tiny ConvNeXt",
      summary: "Depthwise convolutions and inverted bottlenecks.",
      phase_set: ["forward", "backward"],
      frame_count: frameCount,
      camera_presets: [
        camera("overview", "Overview", { x: 0.4, y: 5.4, z: 12.8 }, { x: 0.4, y: 0, z: 0 }, 30),
        camera("head", "Head", { x: 6.2, y: 3.2, z: 10.8 }, { x: 8.1, y: 0, z: 0 }, 26),
      ],
      visual_semantics: visualSemantics,
      payload_catalog: payloadCatalog,
      narrative_ref: "narrative.json",
    },
    graph: {
      nodes,
      edges,
      rootNodeIds: ["embed"],
    },
    timeline,
    payloads,
    narrative: {
      intro: "The ConvNeXt replay emphasizes how a depthwise convolutional stem feeds an inverted bottleneck before the final classifier head stabilizes.",
      chapters: [
        chapter("forward-stem", "Forward", [0, 7], "dwconv1", "Follow the spatial filtering path from the patch stem into the inverted bottleneck."),
        chapter("backward-head", "Backward", [8, 13], "head", "Backward pressure returns from the classifier and compresses through the bottleneck."),
      ],
    },
  };
}

function createLlamaTrace(): TraceBundle {
  const modelId = "tiny-llama";
  const frameCount = 16;
  const tokens = ["<bos>", "neuro", "loom", "glows"];
  const nodes: GraphNode[] = [
    node("token-bos", "<bos>", "token", 0, 0, -6.4, 3.2, 0, { token: "<bos>" }),
    node("token-neuro", "neuro", "token", 0, 1, -6.4, 1.1, 0, { token: "neuro" }),
    node("token-loom", "loom", "token", 0, 2, -6.4, -1.1, 0, { token: "loom" }),
    node("token-glows", "glows", "token", 0, 3, -6.4, -3.2, 0, { token: "glows" }),
    node("rope", "RoPE", "embedding", 1, 0, -2.6, 0, 0, { encoding: "rotary" }),
    node("gqa", "Grouped-Q Attn", "attention", 2, 0, 1.1, 1.9, 0.8, { groups: 2 }),
    node("residual", "Residual", "residual", 2, 1, 1.1, -1.9, -0.6, { stream: "add" }),
    node("swiglu", "SwiGLU", "mlp", 3, 0, 5.1, 1.6, 1.1, { expansion: 8 }),
    node("norm", "RMSNorm", "norm", 3, 1, 5.1, -1.5, -0.3, { type: "rms" }),
    node("logits", "Logits", "logits", 4, 0, 8.6, 0, 0, { vocab: 32000 }),
  ];
  const edges: GraphEdge[] = [
    edge("ll-1", "token-bos", "rope"),
    edge("ll-2", "token-neuro", "rope"),
    edge("ll-3", "token-loom", "rope"),
    edge("ll-4", "token-glows", "rope"),
    edge("ll-5", "rope", "gqa"),
    edge("ll-6", "gqa", "residual"),
    edge("ll-7", "residual", "swiglu"),
    edge("ll-8", "swiglu", "norm"),
    edge("ll-9", "norm", "logits"),
  ];

  const timeline: TraceFrame[] = [];
  const payloads = new Map<string, string>();
  const payloadCatalog: PayloadCatalogEntry[] = [];

  for (let frame = 0; frame < frameCount; frame++) {
    const phase = frame < 8 ? "forward" : frame < 13 ? "decode" : "backward";
    const progress = frame / (frameCount - 1);
    const attention = round(clamp(0.38 + Math.sin(progress * Math.PI * 1.2 + 0.24) * 0.22 + 0.2, 0, 1));
    const residual = round(clamp(0.32 + Math.cos(progress * Math.PI * 1.18 + 0.5) * 0.18 + 0.24, 0, 1));
    const decodeConfidence = round(clamp(0.24 + progress * 0.58 + (phase === "decode" ? 0.08 : 0), 0, 1));
    const lossValue = round(clamp(0.92 - progress * 0.52 + (phase === "backward" ? 0.1 : 0), 0.12, 1));
    const renderId = payloadId(modelId, "render", frame);
    const inspectId = payloadId(modelId, "inspect", frame);
    const headMatrices = createAttentionHeads(tokens.length, progress + (phase === "decode" ? 0.08 : 0));
    const meanMatrix = averageMatrices(headMatrices);
    const overviewMatrix = createWaveMatrix(8, 8, progress * 0.82 + 0.16, 1.08, 0.03);

    const renderPayload = {
      headline: phase === "decode" ? "Grouped-query attention converges on the next token" : "Token embeddings rotate, mix, and settle into logits",
      series: [
        { label: "attention", value: attention },
        { label: "residual", value: residual },
        { label: "decode_conf", value: decodeConfidence },
      ],
      matrix: meanMatrix,
      tokens,
    };

    const inspectPayload = {
      headline: `Frame ${frame + 1} · Tiny Llama block replay`,
      series: [
        { label: "loss", value: lossValue },
        { label: "attention", value: attention },
        { label: "decode_conf", value: decodeConfidence },
      ],
      matrix: overviewMatrix,
      tokens,
      heads: headMatrices.map((matrix, index) => ({
        id: `head-${index + 1}`,
        label: `Head ${index + 1}`,
        matrix,
        focusTokenIndex: (index + frame) % tokens.length,
        score: clamp(attention - index * 0.06, 0, 1),
      })),
      topTokens: [
        { token: "stage", probability: decodeConfidence },
        { token: "glow", probability: clamp(0.58 - progress * 0.14, 0, 1) },
        { token: "trace", probability: clamp(0.42 - progress * 0.1, 0, 1) },
      ],
      selectionDetails: {
        "token-bos": detail("<bos>", "The sequence anchor that seeds causal attention.", [
          metric("Position", 0),
          metric("Attention received", 0.22 + attention * 0.18),
        ]),
        "token-neuro": detail("Token: neuro", "A mid-sequence token feeding grouped-query attention.", [
          metric("Position", 0.33),
          metric("Key share", 0.32 + attention * 0.26),
        ]),
        "token-loom": detail("Token: loom", "The token whose context is redistributed by the attention block.", [
          metric("Position", 0.66),
          metric("Context gain", 0.22 + attention * 0.32),
        ]),
        "token-glows": detail("Token: glows", "The final visible token before the decode head resolves its next step.", [
          metric("Position", 1),
          metric("Decode pull", decodeConfidence),
        ]),
        rope: detail("RoPE", "Rotary embeddings rotate token coordinates instead of adding absolute positions.", [
          metric("Phase angle", 0.28 + progress * 0.32),
          metric("Token coupling", attention),
        ]),
        gqa: detail("Grouped-query attention", "Grouped-query attention shares keys and values while preserving query diversity.", [
          metric("Head score", attention),
          metric("KV sharing", 0.5),
        ]),
        residual: detail("Residual stream", "The residual path retains token identity while carrying attention output forward.", [
          metric("Residual carry", residual),
          metric("Signal blend", 0.26 + attention * 0.34),
        ]),
        swiglu: detail("SwiGLU", "The gated MLP branch expands token capacity before projection.", [
          metric("Expansion", 0.54 + residual * 0.24),
          metric("Gate openness", 0.32 + decodeConfidence * 0.3),
        ]),
        norm: detail("RMSNorm", "Normalization recenters the residual stream before logits.", [
          metric("Stability", 0.44 + progress * 0.24),
          metric("Scale", 0.36 + residual * 0.18),
        ]),
        logits: detail("Logits", "The decode head turns the hidden state into next-token probabilities.", [
          metric("Top probability", decodeConfidence),
          metric("Loss", lossValue),
        ]),
      },
    };

    payloads.set(renderId, JSON.stringify(renderPayload));
    payloads.set(inspectId, JSON.stringify(inspectPayload));
    payloadCatalog.push(payloadEntry(renderId, "render"), payloadEntry(inspectId, "inspect"));

    timeline.push({
      frame_id: frame,
      step: frame,
      substep: 0,
      phase,
      camera_anchor: phase === "decode" ? "decode" : "overview",
      node_states: [
        nodeState("token-bos", 0.22 + attention * 0.34, emphasisForPhase(phase, 0.62), inspectId),
        nodeState("token-neuro", 0.24 + attention * 0.4, emphasisForPhase(phase, 0.66), inspectId),
        nodeState("token-loom", 0.28 + residual * 0.36, emphasisForPhase(phase, 0.64), inspectId),
        nodeState("token-glows", 0.3 + decodeConfidence * 0.44, emphasisForPhase(phase, 0.7), inspectId),
        nodeState("rope", phase === "backward" ? -0.14 - attention * 0.44 : attention * 0.82, emphasisForPhase(phase, 0.78), inspectId),
        nodeState("gqa", phase === "backward" ? -0.2 - attention * 0.56 : attention, emphasisForPhase(phase, 0.94), inspectId),
        nodeState("residual", phase === "backward" ? -0.14 - residual * 0.5 : residual, emphasisForPhase(phase, 0.88), inspectId),
        nodeState("swiglu", phase === "backward" ? -0.22 - residual * 0.52 : residual * 0.92, emphasisForPhase(phase, 0.86), inspectId),
        nodeState("norm", phase === "backward" ? -0.12 - residual * 0.42 : residual * 0.76, emphasisForPhase(phase, 0.78), inspectId),
        nodeState("logits", decodeConfidence * (phase === "backward" ? -0.68 : 1), emphasisForPhase(phase, 0.96), inspectId),
      ],
      edge_states: [
        edgeState("ll-1", phase, 0.34 + attention * 0.18, 0.54),
        edgeState("ll-2", phase, 0.36 + attention * 0.2, 0.56),
        edgeState("ll-3", phase, 0.38 + attention * 0.22, 0.58),
        edgeState("ll-4", phase, 0.4 + decodeConfidence * 0.18, 0.6),
        edgeState("ll-5", phase, 0.48 + attention * 0.34, 0.86),
        edgeState("ll-6", phase, 0.46 + residual * 0.28, 0.8),
        edgeState("ll-7", phase, 0.44 + residual * 0.32, 0.84),
        edgeState("ll-8", phase, 0.42 + residual * 0.24, 0.76),
        edgeState("ll-9", phase, 0.48 + decodeConfidence * 0.36, 0.92),
      ],
      metric_refs: [
        { id: "loss", label: "Loss", value: lossValue },
        { id: "attention", label: "Attention", value: attention },
        { id: "decode", label: "Decode", value: decodeConfidence },
      ],
      payload_refs: [renderId, inspectId],
      note:
        phase === "forward"
          ? "Tokens rotate through RoPE, group into attention, then re-enter the residual stream."
          : phase === "decode"
            ? "The decode head sharpens a single next-token decision while grouped-query attention stays compact."
            : "Backward pressure drains confidence from the logits head and returns through the block.",
    });
  }

  return {
    manifest: {
      trace_version: "1.0.0",
      family: "transformer",
      model_id: modelId,
      dataset_id: "synthetic-token-sequence",
      title: "Tiny Llama",
      summary: "RoPE, grouped-query attention, and SwiGLU in a compact decoder block.",
      phase_set: ["forward", "decode", "backward"],
      frame_count: frameCount,
      camera_presets: [
        camera("overview", "Overview", { x: 0.8, y: 5.6, z: 12.8 }, { x: 0.8, y: 0, z: 0 }, 30),
        camera("decode", "Decode", { x: 6.8, y: 3.2, z: 10.8 }, { x: 8.6, y: 0, z: 0 }, 26),
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
    payloads,
    narrative: {
      intro: "This replay compresses a decoder-only block into one guided pass across RoPE, grouped-query attention, the residual stream, and the logits head.",
      chapters: [
        chapter("rope-and-gqa", "Forward", [0, 7], "gqa", "Watch the token rail rotate through RoPE before grouped-query attention redistributes context."),
        chapter("decode-head", "Decode", [8, 12], "logits", "The replay narrows onto the logits head as next-token confidence stabilizes."),
        chapter("backward-return", "Backward", [13, 15], "swiglu", "A short backward phase pushes pressure from the decode head back through the block."),
      ],
    },
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

function payloadId(modelId: string, kind: "render" | "inspect", frame: number) {
  return `${modelId}-${kind}-${frame}`;
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

function edgeState(edgeId: string, phase: TraceFrame["phase"], intensity: number, emphasis: number) {
  return {
    edgeId,
    intensity: round(clamp(intensity, 0, 1)),
    direction: phase === "backward" ? ("backward" as const) : ("forward" as const),
    emphasis: round(clamp(emphasis, 0, 1)),
  };
}

function createWaveMatrix(rows: number, columns: number, t: number, frequency: number, bias: number) {
  return Array.from({ length: rows }, (_, row) =>
    Array.from({ length: columns }, (_, column) => {
      const x = column / Math.max(columns - 1, 1);
      const y = row / Math.max(rows - 1, 1);
      const wave = Math.sin((x * 2.4 + y * 1.7 + t * frequency) * Math.PI);
      const contour = Math.cos((x - y + t * 0.7) * Math.PI * 1.35) * 0.28;
      return round(clamp(wave * 0.62 + contour + bias, -1, 1));
    }),
  );
}

function sliceMatrix(matrix: number[][], maxRows: number, maxColumns: number) {
  return matrix.slice(0, maxRows).map((row) => row.slice(0, maxColumns));
}

function shiftMatrix(matrix: number[][], offset: number) {
  return matrix.map((row, rowIndex) =>
    row.map((value, columnIndex) => round(clamp(value + Math.sin((rowIndex + columnIndex) * 0.5) * offset, -1, 1))),
  );
}

function createAttentionHeads(size: number, t: number) {
  return Array.from({ length: 4 }, (_, headIndex) =>
    Array.from({ length: size }, (_, row) =>
      Array.from({ length: size }, (_, column) => {
        const locality = Math.max(0, 0.68 - Math.abs(row - column) * (0.12 + headIndex * 0.02));
        const drift = Math.max(0, Math.sin(t * (4.4 + headIndex * 0.35) + row * 0.9 + column * (0.22 + headIndex * 0.08))) * 0.24;
        const focusBias = row === (headIndex + Math.round(t * 4)) % size ? 0.18 : 0;
        return round(clamp(0.08 + locality * (0.56 + t * 0.22) + drift + focusBias, 0, 1));
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
  if (phase === "backward") return base * 0.9;
  if (phase === "update") return base * 0.7;
  if (phase === "loss") return base * 0.95;
  if (phase === "decode") return base * 1.02;
  return base;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function round(value: number) {
  return Math.round(value * 1000) / 1000;
}
