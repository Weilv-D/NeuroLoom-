import { z } from "zod";

export const supportedFamilies = ["mlp", "cnn", "transformer"] as const;
export const supportedPhases = ["forward", "loss", "backward", "update", "decode"] as const;

export type TraceFamily = (typeof supportedFamilies)[number];
export type TracePhase = (typeof supportedPhases)[number];

export const vector3Schema = z.object({
  x: z.number(),
  y: z.number(),
  z: z.number(),
});

export const cameraPresetSchema = z.object({
  id: z.string(),
  label: z.string(),
  position: vector3Schema,
  target: vector3Schema,
  fov: z.number().min(10).max(120),
});

export const visualSemanticsSchema = z.object({
  positive: z.string(),
  negative: z.string(),
  focus: z.string(),
  neutral: z.string(),
  bloomStrength: z.number().min(0),
  fogDensity: z.number().min(0),
});

export const payloadCatalogEntrySchema = z.object({
  id: z.string(),
  kind: z.enum(["render", "inspect"]),
  mimeType: z.string(),
  path: z.string(),
});

export const manifestSchema = z.object({
  trace_version: z.literal("1.0.0"),
  family: z.enum(supportedFamilies),
  model_id: z.string(),
  dataset_id: z.string(),
  title: z.string(),
  summary: z.string(),
  phase_set: z.array(z.enum(supportedPhases)).min(1),
  frame_count: z.number().int().positive(),
  camera_presets: z.array(cameraPresetSchema).min(1),
  visual_semantics: visualSemanticsSchema,
  payload_catalog: z.array(payloadCatalogEntrySchema),
  narrative_ref: z.string(),
});

export const graphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: z.string(),
  layerIndex: z.number().int().nonnegative(),
  order: z.number().int().nonnegative(),
  position: vector3Schema,
  metadata: z.record(z.union([z.string(), z.number(), z.boolean()])).default({}),
});

export const graphEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  type: z.string(),
  weight: z.number().default(1),
});

export const neuronDefSchema = z.object({
  id: z.string(),
  block: z.number().int().nonnegative(),
  index: z.number().int().nonnegative(),
  lane: z.enum(["ffn", "attn_head"]),
});

export const graphSchema = z.object({
  nodes: z.array(graphNodeSchema).min(1),
  edges: z.array(graphEdgeSchema),
  rootNodeIds: z.array(z.string()).min(1),
  neurons: z.array(neuronDefSchema).optional(),
  neuronPositions: z.record(z.tuple([z.number(), z.number(), z.number()])).optional(),
});

export const metricSchema = z.object({
  id: z.string(),
  label: z.string(),
  value: z.number(),
  unit: z.string().optional(),
});

export const nodeStateSchema = z.object({
  nodeId: z.string(),
  activation: z.number(),
  emphasis: z.number().min(0).max(1),
  payloadRef: z.string().optional(),
});

export const neuronStateSchema = z.object({
  id: z.string(),
  activation: z.number(),
});

export const edgeStateSchema = z.object({
  edgeId: z.string(),
  intensity: z.number(),
  direction: z.enum(["forward", "backward", "neutral"]),
  emphasis: z.number().min(0).max(1),
});

export const frameSchema = z.object({
  frame_id: z.number().int().nonnegative(),
  step: z.number().int().nonnegative(),
  substep: z.number().int().nonnegative(),
  phase: z.enum(supportedPhases),
  camera_anchor: z.string(),
  node_states: z.array(nodeStateSchema),
  neuron_states: z.array(neuronStateSchema).optional(),
  edge_states: z.array(edgeStateSchema),
  metric_refs: z.array(metricSchema),
  payload_refs: z.array(z.string()),
  note: z.string().optional(),
});

export const narrativeChapterSchema = z.object({
  id: z.string(),
  label: z.string(),
  frameRange: z.tuple([z.number().int().nonnegative(), z.number().int().nonnegative()]),
  defaultSelection: z.string().optional(),
  description: z.string(),
});

export const narrativeSchema = z.object({
  intro: z.string(),
  chapters: z.array(narrativeChapterSchema).min(1),
});

export type TraceManifest = z.infer<typeof manifestSchema>;
export type TraceGraph = z.infer<typeof graphSchema>;
export type TraceFrame = z.infer<typeof frameSchema>;
export type TraceNarrative = z.infer<typeof narrativeSchema>;

export type TraceBundle = {
  manifest: TraceManifest;
  graph: TraceGraph;
  timeline: TraceFrame[];
  narrative: TraceNarrative;
  payloads: Map<string, string>;
  preview?: Uint8Array;
};
