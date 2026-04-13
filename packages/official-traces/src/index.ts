import type { TraceBundle, TraceFrame } from "@neuroloom/core";

export const qwenOfficialTraceId = "qwen3.5-0.8b-sample";
export const officialTraceIds = [qwenOfficialTraceId] as const;
export const qwenRunnerModelId = "Qwen/Qwen3.5-0.8B";
export const qwenBlockCount = 24;
export const qwenHeadGroupCount = 6;
export const qwenClustersPerLane = 3;

export type OfficialTraceId = (typeof officialTraceIds)[number];
export type QwenLane = "residual" | "attention" | "delta" | "ffn";
export type QwenTopLogit = { token: string; score: number };
export type QwenBlockDigest = {
  block: number;
  residual: number;
  attention: number;
  delta: number;
  ffn: number;
};
export type QwenSampleUnit = {
  id: string;
  label: string;
  nodeId: string;
  block: number;
  lane: QwenLane;
  cluster: number;
  intensity: number;
  polarity: number;
  tokenAffinity: number;
};
export type QwenFramePayload = {
  kind: "qwen-frame";
  model: string;
  prompt: string;
  completion: string;
  token: string;
  tokenIndex: number;
  tokenWindow: string[];
  layerNorms: number[];
  residualBands: number[];
  headGroupScores: number[][];
  attentionRow: number[];
  sampledUnits: QwenSampleUnit[];
  topLogits: QwenTopLogit[];
  blockDigest: QwenBlockDigest[];
  cameraAnchor: string;
};
export type QwenRenderPayload = {
  headline: string;
  prompt: string;
  completion: string;
  token: string;
  tokenIndex: number;
  layerSweep: number[];
  sampledUnits: QwenSampleUnit[];
  topLogits: QwenTopLogit[];
};
export type QwenLayoutMeta = {
  blockCount: number;
  headGroupCount: number;
  clustersPerLane: number;
  tokenWindow: number;
};
export type QwenSessionSeed = Pick<TraceBundle, "manifest" | "graph" | "narrative">;
export type QwenSessionStartedEvent = {
  type: "session_started";
  sessionId: string;
  prompt: string;
  model: string;
  startedAt: number;
  layout: QwenLayoutMeta;
  seed: QwenSessionSeed;
};
export type QwenTokenStepEvent = {
  type: "token_step";
  sessionId: string;
  token: string;
  tokenIndex: number;
  completion: string;
  frame: TraceFrame;
  renderPayloadId: string;
  renderPayload: QwenRenderPayload;
  inspectPayloadId: string;
  inspectPayload: QwenFramePayload;
};
export type QwenSessionCompletedEvent = {
  type: "session_completed";
  sessionId: string;
  tokenCount: number;
  durationMs: number;
  traceFileName: string;
};
export type QwenLiveEvent = QwenSessionStartedEvent | QwenTokenStepEvent | QwenSessionCompletedEvent;

type GraphNode = TraceBundle["graph"]["nodes"][number];
type GraphEdge = TraceBundle["graph"]["edges"][number];
type PayloadCatalogEntry = TraceBundle["manifest"]["payload_catalog"][number];

const qwenVisualSemantics = {
  positive: "#2fe5ff",
  negative: "#ffb85f",
  focus: "#d7ff63",
  neutral: "#eef2ff",
  bloomStrength: 1.9,
  fogDensity: 0.075,
} satisfies TraceBundle["manifest"]["visual_semantics"];

const qwenSamplePrompt = "Describe how NeuroLoom turns a Qwen conversation into a starfield in one vivid paragraph.";
const qwenSampleResponse =
  "NeuroLoom turns each new Qwen token into a pulse that crosses a layered starfield, where attention flares, recurrent memory glides under the surface, and the reply condenses into a visible river of light you can replay frame by frame.";

export function createOfficialTraceBundles(): TraceBundle[] {
  return [createQwenOfficialTraceBundle()];
}

export function createOfficialTraceBundle(id: OfficialTraceId): TraceBundle {
  if (id !== qwenOfficialTraceId) {
    throw new Error(`Unsupported official trace id: ${id}`);
  }
  return createQwenOfficialTraceBundle();
}

export function createQwenOfficialTraceBundle(): TraceBundle {
  return createQwenReplayBundle({
    sessionId: qwenOfficialTraceId,
    prompt: qwenSamplePrompt,
    responseText: qwenSampleResponse,
    title: "Qwen3.5-0.8B Starfield Demo",
    summary: "A single Qwen replay rendered as a dense starfield of attention, residual flow, and logits.",
  });
}

export function createQwenReplayBundle(input: {
  sessionId: string;
  prompt: string;
  responseText: string;
  title?: string;
  summary?: string;
}): TraceBundle {
  const recorder = new QwenSessionRecorder({
    sessionId: input.sessionId,
    prompt: input.prompt,
    title: input.title,
    summary: input.summary,
  });
  const tokens = tokenizeCompletion(input.responseText);
  for (const token of tokens) {
    recorder.pushToken(token);
  }
  recorder.complete();
  return recorder.exportBundle();
}

export function hydrateBundleFromLiveStart(event: QwenSessionStartedEvent): TraceBundle {
  return {
    manifest: cloneJson(event.seed.manifest),
    graph: cloneJson(event.seed.graph),
    narrative: cloneJson(event.seed.narrative),
    timeline: [],
    payloads: new Map<string, string>(),
  };
}

export function applyLiveTokenStep(bundle: TraceBundle, event: QwenTokenStepEvent): TraceBundle {
  bundle.timeline.push(cloneJson(event.frame));
  bundle.payloads.set(event.renderPayloadId, JSON.stringify(event.renderPayload));
  bundle.payloads.set(event.inspectPayloadId, JSON.stringify(event.inspectPayload));
  bundle.manifest.payload_catalog.push(
    payloadCatalogEntry(event.renderPayloadId, "render"),
    payloadCatalogEntry(event.inspectPayloadId, "inspect"),
  );
  bundle.manifest.frame_count = bundle.timeline.length;
  bundle.narrative = buildNarrative(bundle.timeline.length, event.inspectPayload.prompt);
  bundle.manifest.summary = buildSummary(event.inspectPayload.prompt);
  return bundle;
}

export class QwenSessionRecorder {
  readonly sessionId: string;
  readonly prompt: string;
  readonly startedAt: number;

  private readonly tokens: string[] = [];
  private readonly bundle: TraceBundle;

  constructor(input: {
    sessionId: string;
    prompt: string;
    title?: string;
    summary?: string;
    startedAt?: number;
  }) {
    this.sessionId = input.sessionId;
    this.prompt = input.prompt;
    this.startedAt = input.startedAt ?? Date.now();
    this.bundle = createEmptyBundle({
      sessionId: input.sessionId,
      prompt: input.prompt,
      title: input.title,
      summary: input.summary,
    });
  }

  createStartEvent(): QwenSessionStartedEvent {
    return {
      type: "session_started",
      sessionId: this.sessionId,
      prompt: this.prompt,
      model: qwenRunnerModelId,
      startedAt: this.startedAt,
      layout: {
        blockCount: qwenBlockCount,
        headGroupCount: qwenHeadGroupCount,
        clustersPerLane: qwenClustersPerLane,
        tokenWindow: 16,
      },
      seed: {
        manifest: cloneJson(this.bundle.manifest),
        graph: cloneJson(this.bundle.graph),
        narrative: cloneJson(this.bundle.narrative),
      },
    };
  }

  pushToken(token: string): QwenTokenStepEvent {
    const tokenIndex = this.tokens.length;
    this.tokens.push(token);
    const completion = this.tokens.join("");
    const payload = buildInspectPayload({
      prompt: this.prompt,
      token,
      tokenIndex,
      completion,
      totalTokens: this.tokens.length,
    });
    const renderPayload = buildRenderPayload(payload);
    const frame = buildFrame({
      sessionId: this.sessionId,
      tokenIndex,
      prompt: this.prompt,
      token,
      payload,
      graph: this.bundle.graph,
    });
    const renderPayloadId = payloadId(this.sessionId, tokenIndex, "render");
    const inspectPayloadId = payloadId(this.sessionId, tokenIndex, "inspect");

    this.bundle.timeline.push(frame);
    this.bundle.payloads.set(renderPayloadId, JSON.stringify(renderPayload));
    this.bundle.payloads.set(inspectPayloadId, JSON.stringify(payload));
    this.bundle.manifest.payload_catalog.push(
      payloadCatalogEntry(renderPayloadId, "render"),
      payloadCatalogEntry(inspectPayloadId, "inspect"),
    );
    this.bundle.manifest.frame_count = this.bundle.timeline.length;
    this.bundle.narrative = buildNarrative(this.bundle.timeline.length, this.prompt);

    return {
      type: "token_step",
      sessionId: this.sessionId,
      token,
      tokenIndex,
      completion,
      frame: cloneJson(frame),
      renderPayloadId,
      renderPayload: cloneJson(renderPayload),
      inspectPayloadId,
      inspectPayload: cloneJson(payload),
    };
  }

  complete(): QwenSessionCompletedEvent {
    this.bundle.manifest.frame_count = this.bundle.timeline.length;
    this.bundle.narrative = buildNarrative(this.bundle.timeline.length, this.prompt);
    return {
      type: "session_completed",
      sessionId: this.sessionId,
      tokenCount: this.tokens.length,
      durationMs: Date.now() - this.startedAt,
      traceFileName: `${this.sessionId}.loomtrace`,
    };
  }

  exportBundle(): TraceBundle {
    return cloneBundle(this.bundle);
  }
}

export function buildSyntheticQwenResponse(prompt: string): string {
  const trimmed = prompt.trim();
  if (!trimmed) {
    return "A quiet Qwen reply enters the stage as a thin current, then thickens into a visible stream of residual light.";
  }

  const lower = trimmed.toLowerCase();
  if (lower.includes("star") || lower.includes("flow")) {
    return "Qwen answers as a star river: attention sparks jump between recent words, the recurrent lane carries memory underneath, and the next token condenses at the edge of the stage.";
  }
  if (lower.includes("why") || lower.includes("how")) {
    return "The live stage works because each token is reduced into structural summaries, then stretched into motion so the model remains readable without pretending to expose every neuron.";
  }
  if (lower.includes("qwen")) {
    return "Qwen threads the reply through stacked hybrid blocks, turning token context into a layered current that bends toward the next predicted word.";
  }

  return `Qwen receives "${trimmed.slice(0, 64)}" and returns a measured reply, with attention flares above the residual river and the decode head brightening as the sentence settles.`;
}

export function tokenizeCompletion(text: string): string[] {
  return text
    .split(/\s+/)
    .filter(Boolean)
    .map((word, index) => (index === 0 ? word : ` ${word}`));
}

function createEmptyBundle(input: {
  sessionId: string;
  prompt: string;
  title?: string;
  summary?: string;
}): TraceBundle {
  const modelId = input.sessionId === qwenOfficialTraceId ? qwenOfficialTraceId : `qwen-session-${input.sessionId}`;
  return {
    manifest: {
      trace_version: "1.0.0",
      family: "transformer",
      model_id: modelId,
      dataset_id: "qwen-live-session",
      title: input.title ?? "Qwen3.5-0.8B Live Session",
      summary: input.summary ?? buildSummary(input.prompt),
      phase_set: ["decode"],
      frame_count: 0,
      camera_presets: [
        cameraPreset("ingress", "Token Ingress", { x: -8.8, y: 4.6, z: 23.4 }, { x: -8.8, y: 0.2, z: 0 }, 31),
        cameraPreset("braid", "Residual Braid", { x: 0.2, y: 2.1, z: 24.8 }, { x: 0.8, y: -0.8, z: 0 }, 28),
        cameraPreset("decode", "Decode Head", { x: 11.6, y: 2.8, z: 18.2 }, { x: 13.9, y: -0.4, z: 0 }, 26),
      ],
      visual_semantics: qwenVisualSemantics,
      payload_catalog: [],
      narrative_ref: "narrative.json",
    },
    graph: buildGraph(),
    timeline: [],
    payloads: new Map<string, string>(),
    narrative: buildNarrative(0, input.prompt),
  };
}

function buildGraph() {
  const nodes: GraphNode[] = [
    node("prompt", "Prompt", "token", 0, 0, -15.2, 0, 0, { lane: "prompt", subtype: "prompt" }),
    node("embedding", "Embedding", "embedding", 1, 0, -12.8, 0, 0, { lane: "embedding", subtype: "token_embed" }),
  ];
  const edges: GraphEdge[] = [edge("prompt-embedding", "prompt", "embedding", "token-flow", 1)];

  for (let block = 0; block < qwenBlockCount; block++) {
    const x = -10.8 + block * 0.98;
    const wave = Math.sin(block * 0.35) * 0.35;
    const residualId = blockNodeId("residual", block);
    const attentionId = blockNodeId("attention", block);
    const deltaId = blockNodeId("delta", block);
    const ffnId = blockNodeId("ffn", block);

    nodes.push(
      node(residualId, `Residual ${block + 1}`, "residual", block + 2, 0, x, 0.2 + wave * 0.4, 0, {
        lane: "residual",
        block,
        subtype: "residual_stream",
      }),
      node(attentionId, `Attention ${block + 1}`, "attention", block + 2, 1, x - 0.04, 3.1 + wave, 0.55, {
        lane: "attention",
        block,
        subtype: "grouped_query_attention",
      }),
      node(deltaId, `Delta ${block + 1}`, "delta", block + 2, 2, x + 0.08, -2.4 - wave * 0.5, -0.42, {
        lane: "delta",
        block,
        subtype: "gated_deltanet",
      }),
      node(ffnId, `FFN ${block + 1}`, "mlp", block + 2, 3, x + 0.02, -5.2 + wave * 0.25, 0.72, {
        lane: "ffn",
        block,
        subtype: "swiglu_ffn",
      }),
    );

    edges.push(
      edge(`embedding-residual-${block}`, block === 0 ? "embedding" : blockNodeId("residual", block - 1), residualId, "residual-flow", 1),
      edge(`residual-attention-${block}`, residualId, attentionId, "attention-branch", 0.82),
      edge(`residual-delta-${block}`, residualId, deltaId, "delta-branch", 0.74),
      edge(`residual-ffn-${block}`, residualId, ffnId, "ffn-branch", 0.78),
      edge(`attention-return-${block}`, attentionId, block === qwenBlockCount - 1 ? "logits" : blockNodeId("residual", block + 1), "attention-return", 0.84),
      edge(`delta-return-${block}`, deltaId, block === qwenBlockCount - 1 ? "logits" : blockNodeId("residual", block + 1), "delta-return", 0.72),
      edge(`ffn-return-${block}`, ffnId, block === qwenBlockCount - 1 ? "logits" : blockNodeId("residual", block + 1), "ffn-return", 0.88),
    );
  }

  nodes.push(
    node("logits", "Logits", "logits", qwenBlockCount + 3, 0, 13.9, -0.25, 0, { lane: "logits", subtype: "decode_logits" }),
    node("decode", "Decode", "decode", qwenBlockCount + 4, 0, 16.1, -0.2, 0, { lane: "decode", subtype: "token_emit" }),
  );
  edges.push(edge("logits-decode", "logits", "decode", "decode-flow", 1));

  return {
    nodes,
    edges,
    rootNodeIds: ["prompt"],
  };
}

function buildFrame(input: {
  sessionId: string;
  tokenIndex: number;
  prompt: string;
  token: string;
  payload: QwenFramePayload;
  graph: TraceBundle["graph"];
}): TraceFrame {
  const currentBlock = input.payload.tokenIndex % qwenBlockCount;
  const nodeStates = input.graph.nodes.map((graphNode) => {
    if (graphNode.id === "prompt") {
      return nodeState(graphNode.id, 0.28 + input.payload.attentionRow.length * 0.02, 0.55);
    }
    if (graphNode.id === "embedding") {
      return nodeState(graphNode.id, 0.42 + input.payload.residualBands[0] * 0.3, 0.72);
    }
    if (graphNode.id === "logits") {
      return nodeState(graphNode.id, input.payload.topLogits[0]?.score ?? 0.35, 0.92);
    }
    if (graphNode.id === "decode") {
      return nodeState(graphNode.id, clamp(0.55 + input.payload.topLogits[0]?.score * 0.28, 0, 1), 0.96);
    }

    const block = Number(graphNode.metadata.block ?? 0);
    const blockDigest = input.payload.blockDigest[block];
    if (!blockDigest) {
      return nodeState(graphNode.id, 0.12, 0.3);
    }

    const lane = String(graphNode.metadata.lane);
    const activation =
      lane === "residual"
        ? blockDigest.residual
        : lane === "attention"
          ? blockDigest.attention
          : lane === "delta"
            ? blockDigest.delta
            : blockDigest.ffn;
    const distance = Math.abs(block - currentBlock);
    const emphasis = clamp(0.38 + Math.exp(-distance / 4) * 0.54, 0, 1);
    return nodeState(graphNode.id, activation, emphasis);
  });

  const nodeMap = new Map(nodeStates.map((state) => [state.nodeId, state]));
  const edgeStates = input.graph.edges.map((graphEdge) => {
    const source = nodeMap.get(graphEdge.source);
    const target = nodeMap.get(graphEdge.target);
    const intensity = clamp(((Math.abs(source?.activation ?? 0) + Math.abs(target?.activation ?? 0)) / 2) * graphEdge.weight + 0.05, 0, 1);
    const direction = source && target && source.activation > target.activation + 0.08 ? "backward" : "forward";
    const emphasis = clamp(0.26 + intensity * 0.68, 0, 1);
    return edgeState(graphEdge.id, intensity, direction, emphasis);
  });

  const averageResidual =
    input.payload.residualBands.reduce((total, value) => total + value, 0) / Math.max(input.payload.residualBands.length, 1);
  const averageAttention =
    input.payload.blockDigest.reduce((total, block) => total + block.attention, 0) / Math.max(input.payload.blockDigest.length, 1);
  const renderPayloadId = payloadId(input.sessionId, input.tokenIndex, "render");
  const inspectPayloadId = payloadId(input.sessionId, input.tokenIndex, "inspect");

  return {
    frame_id: input.tokenIndex,
    step: input.tokenIndex,
    substep: 0,
    phase: "decode",
    camera_anchor: input.payload.cameraAnchor,
    node_states: nodeStates,
    edge_states: edgeStates,
    metric_refs: [
      metric("token_index", "Token", input.tokenIndex + 1),
      metric("residual", "Residual", round(averageResidual)),
      metric("attention", "Attention", round(averageAttention)),
      metric("logit", "Top Logit", input.payload.topLogits[0]?.score ?? 0),
    ],
    payload_refs: [renderPayloadId, inspectPayloadId],
    note: `Token ${input.tokenIndex + 1} "${input.token.trim()}" ripples through grouped attention, DeltaNet memory, and the decode head.`,
  };
}

function buildInspectPayload(input: {
  prompt: string;
  token: string;
  tokenIndex: number;
  completion: string;
  totalTokens: number;
}): QwenFramePayload {
  const tokenSeed = hashString(`${input.prompt}:${input.token}:${input.tokenIndex}`);
  const focusBlock = input.tokenIndex % qwenBlockCount;
  const tokenWindow = tokenizeCompletion(input.completion).slice(-16);
  const blockDigest = Array.from({ length: qwenBlockCount }, (_, block) => {
    const distance = Math.abs(block - focusBlock);
    const focus = Math.exp(-distance / 4.2);
    const residual = clamp(0.18 + focus * 0.54 + wave(tokenSeed, block, 0.18) * 0.18, 0.04, 1);
    const attention = clamp(0.14 + focus * 0.48 + wave(tokenSeed, block, 0.47) * 0.22, 0.04, 1);
    const delta = clamp(0.16 + focus * 0.41 + wave(tokenSeed, block, 0.73) * 0.24, 0.04, 1);
    const ffn = clamp(0.2 + focus * 0.44 + wave(tokenSeed, block, 0.91) * 0.2, 0.04, 1);
    return {
      block,
      residual: round(residual),
      attention: round(attention),
      delta: round(delta),
      ffn: round(ffn),
    };
  });

  const layerNorms = blockDigest.map((block) => round(clamp(block.residual * 0.48 + block.ffn * 0.22 + 0.12, 0, 1)));
  const residualBands = blockDigest.map((block) => round(clamp(block.residual * 0.72 + block.delta * 0.16 + block.attention * 0.12, 0, 1)));
  const headGroupScores = Array.from({ length: qwenBlockCount }, (_, block) =>
    Array.from({ length: qwenHeadGroupCount }, (_, head) =>
      round(clamp(0.14 + blockDigest[block]!.attention * 0.56 + wave(tokenSeed + head * 7, block, head * 0.31) * 0.22, 0, 1)),
    ),
  );

  const attentionRaw = tokenWindow.map((_, index) => {
    const recency = Math.exp(-(tokenWindow.length - 1 - index) / 3.4);
    return 0.08 + recency * 0.78 + positiveWave(tokenSeed, index, 0.29) * 0.18;
  });
  const attentionRow = normalize(attentionRaw).map(round);
  const topLogits = buildTopLogits(input.token.trim() || "token", tokenSeed);
  const sampledUnits: QwenSampleUnit[] = [];

  for (let block = 0; block < qwenBlockCount; block++) {
    for (const lane of ["residual", "attention", "delta", "ffn"] as const) {
      const digest = blockDigest[block]!;
      const laneValue = lane === "residual" ? digest.residual : lane === "attention" ? digest.attention : lane === "delta" ? digest.delta : digest.ffn;
      for (let cluster = 0; cluster < qwenClustersPerLane; cluster++) {
        const local = clamp(laneValue * (0.76 + cluster * 0.12) + wave(tokenSeed + cluster * 19, block, cluster * 0.22) * 0.16, 0, 1);
        sampledUnits.push({
          id: `cluster:${lane}:${block}:${cluster}`,
          label: `${lane} ${block + 1}.${cluster + 1}`,
          nodeId: blockNodeId(lane === "ffn" ? "ffn" : lane, block),
          block,
          lane,
          cluster,
          intensity: round(local),
          polarity: round(wave(tokenSeed + cluster * 13, block, 0.18)),
          tokenAffinity: round(clamp(Math.exp(-Math.abs(block - focusBlock) / 3.8), 0, 1)),
        });
      }
    }
  }

  return {
    kind: "qwen-frame",
    model: qwenRunnerModelId,
    prompt: input.prompt,
    completion: input.completion,
    token: input.token,
    tokenIndex: input.tokenIndex,
    tokenWindow,
    layerNorms,
    residualBands,
    headGroupScores,
    attentionRow,
    sampledUnits,
    topLogits,
    blockDigest,
    cameraAnchor: input.tokenIndex < 4 ? "ingress" : input.tokenIndex < 12 ? "braid" : "decode",
  };
}

function buildRenderPayload(payload: QwenFramePayload): QwenRenderPayload {
  return {
    headline:
      payload.tokenIndex < 4
        ? "Ingress: tokens begin threading into the hybrid stack."
        : payload.tokenIndex < 12
          ? "Braid: grouped attention and recurrent memory tighten into a residual river."
          : "Decode: the starfield narrows and the next word condenses at the edge.",
    prompt: payload.prompt,
    completion: payload.completion,
    token: payload.token,
    tokenIndex: payload.tokenIndex,
    layerSweep: payload.residualBands,
    sampledUnits: payload.sampledUnits.filter((unit) => unit.block % 3 === payload.tokenIndex % 3 || unit.tokenAffinity > 0.42),
    topLogits: payload.topLogits,
  };
}

function buildNarrative(frameCount: number, prompt: string): TraceBundle["narrative"] {
  if (frameCount === 0) {
    return {
      intro: `Live Qwen session seeded from prompt: "${prompt.trim() || "Awaiting input"}".`,
      chapters: [
        {
          id: "awaiting",
          label: "Awaiting Tokens",
          frameRange: [0, 0],
          defaultSelection: "embedding",
          description: "The stage is primed but the first decode step has not arrived yet.",
        },
      ],
    };
  }

  const earlyEnd = Math.min(frameCount - 1, Math.max(0, Math.floor(frameCount * 0.25)));
  const midStart = Math.min(frameCount - 1, earlyEnd + 1);
  const midEnd = Math.min(frameCount - 1, Math.max(midStart, Math.floor(frameCount * 0.72)));
  const finalStart = Math.min(frameCount - 1, Math.max(midStart, midEnd));

  return {
    intro: `Qwen3.5-0.8B transforms the prompt into a live starfield, then preserves the whole exchange as a replayable loomtrace.`,
    chapters: [
      {
        id: "ingress",
        label: "Ingress",
        frameRange: [0, earlyEnd],
        defaultSelection: "embedding",
        description: "The opening tokens cross the embedding gate and wake the first hybrid blocks.",
      },
      {
        id: "braid",
        label: "Braid",
        frameRange: [midStart, midEnd],
        defaultSelection: blockNodeId("attention", Math.min(8, qwenBlockCount - 1)),
        description: "Grouped attention and DeltaNet memory braid into the residual river at mid-stack.",
      },
      {
        id: "decode",
        label: "Decode",
        frameRange: [finalStart, frameCount - 1],
        defaultSelection: "logits",
        description: "The live response narrows into the decode head and leaves a replay trail behind it.",
      },
    ],
  };
}

function buildSummary(prompt: string) {
  return `Live-first replay for a single Qwen3.5-0.8B conversation. Prompt seed: "${prompt.trim().slice(0, 72)}".`;
}

function buildTopLogits(token: string, seed: number): QwenTopLogit[] {
  const stem = token.replace(/^[\s]+/, "") || "token";
  const candidates = [
    stem,
    " attention",
    " residual",
    " starfield",
    " memory",
    " decode",
  ];
  return candidates
    .map((candidate, index) => ({
      token: candidate,
      score: round(clamp(0.22 + positiveWave(seed + index * 17, index, 0.19) * 0.58 + (index === 0 ? 0.16 : 0), 0.01, 0.99)),
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 5);
}

function cameraPreset(
  id: string,
  label: string,
  position: { x: number; y: number; z: number },
  target: { x: number; y: number; z: number },
  fov: number,
) {
  return { id, label, position, target, fov };
}

function payloadId(seed: string, frameIndex: number, kind: "render" | "inspect") {
  const safeSeed = seed.replace(/[^a-zA-Z0-9_-]/g, "-");
  return `${safeSeed}-frame-${String(frameIndex).padStart(4, "0")}-${kind}`;
}

function payloadCatalogEntry(id: string, kind: "render" | "inspect"): PayloadCatalogEntry {
  return {
    id,
    kind,
    mimeType: "application/json",
    path: `payload/${kind}/${id}.json`,
  };
}

function blockNodeId(kind: "residual" | "attention" | "delta" | "ffn", block: number) {
  return `${kind}-${String(block).padStart(2, "0")}`;
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
    position: { x: round3(x), y: round3(y), z: round3(z) },
    metadata,
  };
}

function edge(id: string, source: string, target: string, type: string, weight: number): GraphEdge {
  return { id, source, target, type, weight };
}

function nodeState(nodeId: string, activation: number, emphasis: number) {
  return {
    nodeId,
    activation: round(clamp(activation, -1, 1)),
    emphasis: round(clamp(emphasis, 0, 1)),
  };
}

function edgeState(edgeId: string, intensity: number, direction: "forward" | "backward" | "neutral", emphasis: number) {
  return {
    edgeId,
    intensity: round(clamp(intensity, 0, 1)),
    direction,
    emphasis: round(clamp(emphasis, 0, 1)),
  };
}

function metric(id: string, label: string, value: number) {
  return { id, label, value: round(clamp(value, 0, 999)) };
}

function normalize(values: number[]) {
  const total = values.reduce((sum, value) => sum + value, 0) || 1;
  return values.map((value) => value / total);
}

function wave(seed: number, layer: number, offset: number) {
  return Math.sin(seed * 0.0009 + layer * 0.44 + offset);
}

function positiveWave(seed: number, layer: number, offset: number) {
  return (wave(seed, layer, offset) + 1) / 2;
}

function hashString(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0);
}

function cloneBundle(bundle: TraceBundle): TraceBundle {
  return {
    manifest: cloneJson(bundle.manifest),
    graph: cloneJson(bundle.graph),
    narrative: cloneJson(bundle.narrative),
    timeline: cloneJson(bundle.timeline),
    payloads: new Map(bundle.payloads),
    preview: bundle.preview ? new Uint8Array(bundle.preview) : undefined,
  };
}

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function round(value: number) {
  return Number(value.toFixed(4));
}

function round3(value: number) {
  return Number(value.toFixed(3));
}
