import { describe, expect, test } from "vitest";

import { validateTraceBundle } from "../src/validate.js";

const baseBundle = {
  manifest: {
    trace_version: "1.0.0",
    family: "mlp",
    model_id: "spiral-2d-mlp",
    dataset_id: "spiral-2d",
    title: "MLP",
    summary: "Summary",
    phase_set: ["forward", "loss", "backward", "update"],
    frame_count: 1,
    camera_presets: [
      {
        id: "overview",
        label: "Overview",
        position: { x: 0, y: 2, z: 8 },
        target: { x: 0, y: 0, z: 0 },
        fov: 40,
      },
    ],
    visual_semantics: {
      positive: "#0cf2ff",
      negative: "#ffb347",
      focus: "#dcff66",
      neutral: "#eef2ff",
      bloomStrength: 1.2,
      fogDensity: 0.04,
    },
    payload_catalog: [{ id: "payload", kind: "render", mimeType: "application/json", path: "payload/test.json" }],
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
        position: { x: 0, y: 0, z: 0 },
        metadata: {},
      },
    ],
    edges: [],
    rootNodeIds: ["input"],
  },
  timeline: [
    {
      frame_id: 0,
      step: 0,
      substep: 0,
      phase: "forward",
      camera_anchor: "overview",
      node_states: [{ nodeId: "input", activation: 0.4, emphasis: 0.5, payloadRef: "payload" }],
      edge_states: [],
      metric_refs: [{ id: "loss", label: "Loss", value: 0.5 }],
      payload_refs: ["payload"],
    },
  ],
  narrative: {
    intro: "Intro",
    chapters: [{ id: "c1", label: "Start", frameRange: [0, 0], description: "desc", defaultSelection: "input" }],
  },
  payloads: new Map([["payload", JSON.stringify({ values: [0.1, 0.2] })]]),
};

describe("validateTraceBundle", () => {
  test("accepts a valid bundle", () => {
    const result = validateTraceBundle(baseBundle);
    expect(result.ok).toBe(true);
  });

  test("rejects disallowed node types", () => {
    const result = validateTraceBundle({
      ...baseBundle,
      graph: {
        ...baseBundle.graph,
        nodes: [{ ...baseBundle.graph.nodes[0], type: "attention" }],
      },
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors[0]).toContain("not allowed");
    }
  });
});
