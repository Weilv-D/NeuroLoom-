import type { TraceBundle } from "@neuroloom/core";

import { ffnGridHeight, ffnGridWidth, qwenAttnHeadsPerBlock, qwenBlockCount, qwenFfnNeuronsPerBlock } from "./constants";
import { clamp, edge, node, round, round3, seeded, wave } from "./helpers";
import type { GraphEdge, GraphNode, NeuronDef, QwenBlockDigest } from "./types";

export function buildGraph() {
  const nodes: GraphNode[] = [
    node("prompt", "Prompt", "token", 0, 0, -15.2, 0, 0, { lane: "prompt", subtype: "prompt" }),
    node("embedding", "Embedding", "embedding", 1, 0, -12.8, 0, 0, { lane: "embedding", subtype: "token_embed" }),
  ];
  const edges: GraphEdge[] = [edge("prompt-embedding", "prompt", "embedding", "token-flow", 1)];
  const neurons: NeuronDef[] = [];
  const neuronPositions: Record<string, [number, number, number]> = {};

  for (let block = 0; block < qwenBlockCount; block++) {
    const x = -10.8 + block * 0.98;
    const waveOffset = Math.sin(block * 0.35) * 0.35;
    const residualId = blockNodeId("residual", block);
    const attentionId = blockNodeId("attention", block);
    const deltaId = blockNodeId("delta", block);
    const ffnId = blockNodeId("ffn", block);

    nodes.push(
      node(residualId, `Residual ${block + 1}`, "residual", block + 2, 0, x, 0.2 + waveOffset * 0.4, 0, {
        lane: "residual",
        block,
        subtype: "residual_stream",
      }),
      node(attentionId, `Attention ${block + 1}`, "attention", block + 2, 1, x - 0.04, 3.1 + waveOffset, 0.55, {
        lane: "attention",
        block,
        subtype: "grouped_query_attention",
      }),
      node(deltaId, `Delta ${block + 1}`, "delta", block + 2, 2, x + 0.08, -2.4 - waveOffset * 0.5, -0.42, {
        lane: "delta",
        block,
        subtype: "gated_deltanet",
      }),
      node(ffnId, `FFN ${block + 1}`, "mlp", block + 2, 3, x + 0.02, -5.2 + waveOffset * 0.25, 0.72, {
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
      edge(
        `attention-return-${block}`,
        attentionId,
        block === qwenBlockCount - 1 ? "logits" : blockNodeId("residual", block + 1),
        "attention-return",
        0.84,
      ),
      edge(
        `delta-return-${block}`,
        deltaId,
        block === qwenBlockCount - 1 ? "logits" : blockNodeId("residual", block + 1),
        "delta-return",
        0.72,
      ),
      edge(`ffn-return-${block}`, ffnId, block === qwenBlockCount - 1 ? "logits" : blockNodeId("residual", block + 1), "ffn-return", 0.88),
    );

    // FFN neurons: natural Gaussian ring (Torus) galaxy transitioning smoothly into sparse space
    const ffnRng = seeded(`neurons:${block}`, 42);
    const ffnCenterX = x + 0.05;
    const ffnCenterY = Math.sin(block * 0.2) * 0.2;
    
    for (let idx = 0; idx < qwenFfnNeuronsPerBlock; idx++) {
      const neuronId = `neuron:${block}:${idx}`;
      
      // Box-Muller transform for natural Gaussian distribution (no harsh square edges)
      const u1 = Math.max(ffnRng(), 0.00001);
      const u2 = ffnRng();
      const g1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2); // Gaussian X
      const g2 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2); // Gaussian Y
      const g3 = Math.sqrt(-2.0 * Math.log(Math.max(ffnRng(), 0.00001))) * Math.cos(2.0 * Math.PI * ffnRng()); // Gaussian Z

      // Create a majestic elliptical ring
      // ringRadius pulls the dense core out from the absolute center, creating a visible "hole" or "ring"
      const ringRadius = 2.8; 
      
      // angle around the Z-Y plane (the face of the ring)
      const angle = ffnRng() * Math.PI * 2.0;

      // X jitter using pure Gaussian so blocks merge organically without looking like stacked boxes
      const xJitter = g1 * 0.65;
      
      // Thickness of the ring spreads out naturally using Gaussian noise
      // Squashed Y creates an elliptical/galactic disk orientation
      const ySpread = g2 * 0.45;
      const zSpread = g3 * 1.5;
      
      const ny = Math.sin(angle) * ringRadius * 0.35 + ySpread;
      const nz = Math.cos(angle) * ringRadius * 1.0 + zSpread;
      
      const pos: [number, number, number] = [
        round3(ffnCenterX + xJitter),
        round3(ffnCenterY + ny),
        round3(nz)
      ];
      neurons.push({ id: neuronId, block, index: idx, lane: "ffn" });
      neuronPositions[neuronId] = pos;
    }

    // Attention heads: glowing orbital nodes around the ring
    const attnRng = seeded(`attn:${block}`, 7);
    for (let head = 0; head < qwenAttnHeadsPerBlock; head++) {
      const neuronId = `attn_head:${block}:${head}`;
      
      const angle = (head / qwenAttnHeadsPerBlock) * Math.PI * 2.0 + block * 0.5;
      const radius = 4.5 + attnRng() * 1.5;
      const xJitter = (attnRng() - 0.5) * 1.2;
      
      const hx = ffnCenterX + xJitter;
      const hy = ffnCenterY + radius * Math.sin(angle) * 0.45;
      const hz = radius * Math.cos(angle) * 1.2;
      
      neurons.push({ id: neuronId, block, index: head, lane: "attn_head" });
      neuronPositions[neuronId] = [round3(hx), round3(hy), round3(hz)];
    }
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
    neurons,
    neuronPositions,
  };
}

export function neuronArcPosition(
  centerX: number,
  centerY: number,
  idx: number,
  gridW: number,
  gridH: number,
  rng: () => number,
): [number, number, number] {
  const col = idx % gridW;
  const row = Math.floor(idx / gridW);
  const u = col / (gridW - 1) - 0.5;
  const v = row / (gridH - 1) - 0.5;

  const arcRadius = 0.65 + Math.abs(u) * 0.3;
  const angle = v * Math.PI * 0.85 + u * 0.4;
  const spreadX = u * 0.48;
  const spreadY = Math.sin(angle) * arcRadius * 0.55;
  const spreadZ = Math.cos(angle) * arcRadius * 0.25;

  const jitter = 0.12;

  return [
    round3(centerX + spreadX + (rng() - 0.5) * jitter),
    round3(centerY + spreadY + (rng() - 0.5) * jitter * 0.7),
    round3(spreadZ + (rng() - 0.5) * jitter * 0.5),
  ];
}

export function buildNeuronStates(
  graph: TraceBundle["graph"],
  tokenSeed: number,
  focusBlock: number,
  blockDigest: QwenBlockDigest[],
): { id: string; activation: number }[] {
  const neurons = graph.neurons;
  if (!neurons || neurons.length === 0) return [];

  const result: { id: string; activation: number }[] = [];
  for (let i = 0; i < neurons.length; i++) {
    const neuron = neurons[i]!;
    const distance = Math.abs(neuron.block - focusBlock);
    const focus = Math.exp(-distance / 3.5);

    if (neuron.lane === "ffn") {
      const hash = neuronIndexHash(neuron.block, neuron.index);
      const sparse = Math.abs(Math.sin(hash * 0.00013 + tokenSeed * 0.00007));
      const isActivated = sparse > 0.88;
      const activation = isActivated
        ? clamp(focus * 0.6 + wave(tokenSeed, neuron.block, hash * 0.001) * 0.35 + 0.15, -1, 1)
        : clamp(wave(tokenSeed, neuron.block, hash * 0.001) * 0.06, -0.05, 0.08);
      result.push({ id: neuron.id, activation: round(activation) });
    } else {
      const blockAct = blockDigest[neuron.block]?.attention ?? 0.14;
      const headWave = wave(tokenSeed + neuron.index * 7, neuron.block, neuron.index * 0.31);
      const activation = clamp(0.1 + focus * blockAct * 0.5 + headWave * 0.25, -1, 1);
      result.push({ id: neuron.id, activation: round(activation) });
    }
  }
  return result;
}

export function neuronIndexHash(block: number, index: number): number {
  let h = block * 374761393 + index * 668265263;
  h = (h ^ (h >>> 13)) * 1274126177;
  h = h ^ (h >>> 16);
  return Math.abs(h >>> 0);
}

function blockNodeId(kind: "residual" | "attention" | "delta" | "ffn", block: number) {
  return `${kind}-${String(block).padStart(2, "0")}`;
}
