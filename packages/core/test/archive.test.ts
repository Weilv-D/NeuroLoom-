import { describe, expect, test } from "vitest";

import { createLoomTraceArchive, loadLoomTraceArchive } from "../src/archive.js";

const sampleBundle = {
  manifest: {
    trace_version: "1.0.0" as const,
    family: "mlp" as const,
    model_id: "spiral-2d-mlp",
    dataset_id: "spiral-2d",
    title: "Spiral MLP",
    summary: "Replay sample",
    phase_set: ["forward", "loss", "backward", "update"] as const,
    frame_count: 2,
    camera_presets: [
      {
        id: "overview",
        label: "Overview",
        position: { x: 0, y: 1, z: 7 },
        target: { x: 0, y: 0, z: 0 },
        fov: 35,
      },
    ],
    visual_semantics: {
      positive: "#15f0ff",
      negative: "#ffb45b",
      focus: "#d8ff66",
      neutral: "#eef2ff",
      bloomStrength: 1.2,
      fogDensity: 0.04,
    },
    payload_catalog: [
      { id: "render-0", kind: "render" as const, mimeType: "application/json", path: "payload/render-0.json" },
      { id: "inspect-0", kind: "inspect" as const, mimeType: "application/json", path: "payload/inspect-0.json" },
    ],
    narrative_ref: "narrative.json",
  },
  graph: {
    nodes: [
      {
        id: "input",
        label: "Input",
        type: "input",
        layerIndex: 0,
        order: 0,
        position: { x: -2, y: 0, z: 0 },
        metadata: {},
      },
      {
        id: "output",
        label: "Output",
        type: "output",
        layerIndex: 1,
        order: 0,
        position: { x: 2, y: 0, z: 0 },
        metadata: {},
      },
    ],
    edges: [{ id: "edge-0", source: "input", target: "output", type: "flow", weight: 1 }],
    rootNodeIds: ["input"],
  },
  timeline: [
    {
      frame_id: 0,
      step: 0,
      substep: 0,
      phase: "forward" as const,
      camera_anchor: "overview",
      node_states: [
        { nodeId: "input", activation: 0.6, emphasis: 0.6, payloadRef: "inspect-0" },
        { nodeId: "output", activation: 0.1, emphasis: 0.4, payloadRef: "inspect-0" },
      ],
      edge_states: [{ edgeId: "edge-0", intensity: 0.7, direction: "forward" as const, emphasis: 0.5 }],
      metric_refs: [{ id: "loss", label: "Loss", value: 0.9 }],
      payload_refs: ["render-0", "inspect-0"],
      note: "forward pulse",
    },
    {
      frame_id: 1,
      step: 0,
      substep: 1,
      phase: "loss" as const,
      camera_anchor: "overview",
      node_states: [
        { nodeId: "input", activation: 0.2, emphasis: 0.3, payloadRef: "inspect-0" },
        { nodeId: "output", activation: 0.9, emphasis: 0.9, payloadRef: "inspect-0" },
      ],
      edge_states: [{ edgeId: "edge-0", intensity: 0.4, direction: "forward" as const, emphasis: 0.4 }],
      metric_refs: [{ id: "loss", label: "Loss", value: 0.4 }],
      payload_refs: ["render-0", "inspect-0"],
      note: "loss anchor",
    },
  ],
  narrative: {
    intro: "intro",
    chapters: [{ id: "start", label: "Start", frameRange: [0, 1] as [number, number], defaultSelection: "input", description: "desc" }],
  },
  payloads: new Map([
    ["render-0", JSON.stringify({ matrix: [[0.1, 0.2]] })],
    ["inspect-0", JSON.stringify({ headline: "Inspect", series: [{ label: "loss", value: 0.4 }] })],
  ]),
};

describe("loomtrace archive", () => {
  test("round-trips archive creation and loading", async () => {
    const archive = await createLoomTraceArchive(sampleBundle);
    const loaded = await loadLoomTraceArchive(archive);

    expect(loaded.manifest.model_id).toBe(sampleBundle.manifest.model_id);
    expect(loaded.timeline).toHaveLength(2);
    expect(loaded.graph.nodes[1]?.id).toBe("output");
    expect(loaded.payloads.get("render-0")).toBe(sampleBundle.payloads.get("render-0"));
  });
});
