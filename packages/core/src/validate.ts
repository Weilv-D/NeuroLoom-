import { type TraceBundle, type TraceFamily, frameSchema, graphSchema, manifestSchema, narrativeSchema } from "./schema.js";

const familyNodeTypeAllowlist: Record<TraceFamily, Set<string>> = {
  mlp: new Set(["input", "linear", "activation", "output", "loss"]),
  cnn: new Set(["input", "conv", "norm", "activation", "pool", "dense", "output", "loss", "stage"]),
  transformer: new Set(["token", "embedding", "attention", "delta", "residual", "mlp", "norm", "logits", "loss", "decode"]),
};

export type ValidationResult = { ok: true; family: TraceFamily; warnings: string[] } | { ok: false; errors: string[]; warnings: string[] };

export function validateTraceBundle(input: unknown): ValidationResult {
  const warnings: string[] = [];
  const bundle = input as Partial<TraceBundle>;
  const manifestResult = manifestSchema.safeParse(bundle.manifest);
  if (!manifestResult.success) {
    return {
      ok: false,
      errors: manifestResult.error.issues.map((issue) => `manifest.${issue.path.join(".")}: ${issue.message}`),
      warnings,
    };
  }

  const graphResult = graphSchema.safeParse(bundle.graph);
  if (!graphResult.success) {
    return {
      ok: false,
      errors: graphResult.error.issues.map((issue) => `graph.${issue.path.join(".")}: ${issue.message}`),
      warnings,
    };
  }

  const narrativeResult = narrativeSchema.safeParse(bundle.narrative);
  if (!narrativeResult.success) {
    return {
      ok: false,
      errors: narrativeResult.error.issues.map((issue) => `narrative.${issue.path.join(".")}: ${issue.message}`),
      warnings,
    };
  }

  if (!Array.isArray(bundle.timeline)) {
    return { ok: false, errors: ["timeline: expected array"], warnings };
  }

  const timelineErrors = bundle.timeline.flatMap((frame, index) => {
    const result = frameSchema.safeParse(frame);
    return result.success ? [] : result.error.issues.map((issue) => `timeline[${index}].${issue.path.join(".")}: ${issue.message}`);
  });

  if (timelineErrors.length > 0) {
    return { ok: false, errors: timelineErrors, warnings };
  }

  const manifest = manifestResult.data;
  const graph = graphResult.data;
  const narrative = narrativeResult.data;
  const timeline = bundle.timeline.map((frame) => frameSchema.parse(frame));
  const payloads = bundle.payloads ?? new Map<string, string>();

  const allowedNodeTypes = familyNodeTypeAllowlist[manifest.family];
  const semanticErrors: string[] = [];

  graph.nodes.forEach((node) => {
    if (!allowedNodeTypes.has(node.type)) {
      semanticErrors.push(`graph.nodes.${node.id}: node type "${node.type}" is not allowed for family "${manifest.family}"`);
    }
  });

  const nodeIds = new Set(graph.nodes.map((node) => node.id));
  const edgeIds = new Set(graph.edges.map((edge) => edge.id));
  const cameraIds = new Set(manifest.camera_presets.map((camera) => camera.id));

  graph.edges.forEach((edge) => {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
      semanticErrors.push(`graph.edges.${edge.id}: source/target must reference existing nodes`);
    }
  });

  timeline.forEach((frame) => {
    if (!cameraIds.has(frame.camera_anchor)) {
      semanticErrors.push(`timeline.frame_${frame.frame_id}: unknown camera anchor "${frame.camera_anchor}"`);
    }

    frame.node_states.forEach((nodeState) => {
      if (!nodeIds.has(nodeState.nodeId)) {
        semanticErrors.push(`timeline.frame_${frame.frame_id}: unknown node "${nodeState.nodeId}"`);
      }
    });

    frame.edge_states.forEach((edgeState) => {
      if (!edgeIds.has(edgeState.edgeId)) {
        semanticErrors.push(`timeline.frame_${frame.frame_id}: unknown edge "${edgeState.edgeId}"`);
      }
    });

    frame.payload_refs.forEach((payloadRef) => {
      if (!payloads.has(payloadRef)) {
        warnings.push(`timeline.frame_${frame.frame_id}: payload "${payloadRef}" not found in bundle`);
      }
    });
  });

  if (timeline.length !== manifest.frame_count) {
    semanticErrors.push(`manifest.frame_count: expected ${manifest.frame_count}, received ${timeline.length} timeline frames`);
  }

  narrative.chapters.forEach((chapter) => {
    const [start, end] = chapter.frameRange;
    if (end < start || end >= timeline.length) {
      semanticErrors.push(`narrative.chapters.${chapter.id}: invalid frame range ${start}-${end}`);
    }
    if (chapter.defaultSelection && !nodeIds.has(chapter.defaultSelection)) {
      warnings.push(`narrative.chapters.${chapter.id}: default selection "${chapter.defaultSelection}" is unknown`);
    }
  });

  if (semanticErrors.length > 0) {
    return { ok: false, errors: semanticErrors, warnings };
  }

  return { ok: true, family: manifest.family, warnings };
}
