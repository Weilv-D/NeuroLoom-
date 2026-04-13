# `.loomtrace` Profile For NeuroLoom

Version: `1.0.0`

## Purpose

NeuroLoom uses `.loomtrace` to replay a single `Qwen/Qwen3.5-0.8B` conversation after live generation has finished.

The archive stores:

- structural graph anchors
- token-step frames
- render payloads for the starfield stage
- inspect payloads for the side panels
- chapter metadata for replay scrubbing

NeuroLoom the product is single-model, but the archive still declares:

- `family: transformer`

This keeps the replay engine generic while the app remains Qwen-only.

## Archive Layout

Required entries:

```text
manifest.json
graph.json
timeline.ndjson
narrative.json
payload/
```

Optional:

```text
preview.webp
```

## `manifest.json`

Example:

```json
{
  "trace_version": "1.0.0",
  "family": "transformer",
  "model_id": "qwen3.5-0.8b-sample",
  "dataset_id": "qwen-live-session",
  "title": "Qwen3.5-0.8B Starfield Demo",
  "summary": "Live-first replay for a single Qwen3.5-0.8B conversation.",
  "phase_set": ["decode"],
  "frame_count": 29,
  "camera_presets": [],
  "visual_semantics": {},
  "payload_catalog": [],
  "narrative_ref": "narrative.json"
}
```

Key rules:

- `family` stays `transformer`
- `phase_set` is currently `["decode"]`
- `frame_count` equals the number of token frames in `timeline.ndjson`
- payload catalog entries reference both `render` and `inspect` JSON payloads

## `graph.json`

The graph describes the fixed Qwen stage.

Node types used by NeuroLoom:

- `token`
- `embedding`
- `residual`
- `attention`
- `delta`
- `mlp`
- `logits`
- `decode`

The `delta` node type is used for the recurrent memory lane visible in the Qwen starfield.

Each block typically contributes:

- one residual node
- one attention node
- one delta node
- one FFN node

Metadata carries Qwen-specific detail such as:

- `lane`
- `block`
- `subtype`

Example node:

```json
{
  "id": "delta-07",
  "label": "Delta 8",
  "type": "delta",
  "layerIndex": 9,
  "order": 2,
  "position": { "x": -3.94, "y": -2.25, "z": -0.42 },
  "metadata": {
    "lane": "delta",
    "block": 7,
    "subtype": "gated_deltanet"
  }
}
```

## `timeline.ndjson`

Each line is one generated token.

Example:

```json
{
  "frame_id": 12,
  "step": 12,
  "substep": 0,
  "phase": "decode",
  "camera_anchor": "braid",
  "node_states": [],
  "edge_states": [],
  "metric_refs": [],
  "payload_refs": [],
  "note": "Token 13 ripples through grouped attention, DeltaNet memory, and the decode head."
}
```

Rules:

- one token = one frame
- `frame_id` is zero-based and contiguous
- `camera_anchor` must match a preset in `manifest.camera_presets`
- `payload_refs` point to the render and inspect payloads for that token

## Payloads

NeuroLoom writes two JSON payloads per token:

- `render`
  Fast payload for the stage
- `inspect`
  Full payload for the right-side panels and selection details

### Render payload

Contains:

- headline
- prompt
- completion text so far
- current token
- token index
- layer sweep
- sampled units
- top logits

### Inspect payload

Contains:

- `token`
- `tokenIndex`
- `tokenWindow`
- `completion`
- `layerNorms`
- `residualBands`
- `headGroupScores`
- `attentionRow`
- `sampledUnits`
- `topLogits`
- `blockDigest`
- `cameraAnchor`

The important distinction is that `sampledUnits` are not decorative particles. They are the replayable star clusters that the stage uses to represent sampled internal structure.

## `narrative.json`

Replay chapters stay lightweight:

- `Ingress`
- `Braid`
- `Decode`

They give the scrubber stable jump points and default selections.

## Validation

Validation order:

1. schema validation
2. transformer node allowlist validation
3. camera, node, edge, and payload reference validation
4. narrative frame range validation

Validate one or more bundles:

```bash
pnpm --filter @neuroloom/core loomtrace path/to/file.loomtrace
```

## Design Constraint

This `.loomtrace` profile is intentionally narrow.

It exists to preserve one Qwen live session as a deterministic replay artifact, not to become a universal runtime dump format.
