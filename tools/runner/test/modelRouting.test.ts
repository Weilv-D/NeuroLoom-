import assert from "node:assert/strict";
import test from "node:test";

import { resolveRuntimeModel } from "../src/modelRouting.js";

test("synthetic mode keeps the canonical requested model", () => {
  const resolution = resolveRuntimeModel({
    backendModel: "qwen3.5:0.8b",
    canonicalModel: "Qwen/Qwen3.5-0.8B",
    mode: "synthetic",
  });

  assert.equal(resolution.requestedModel, "Qwen/Qwen3.5-0.8B");
  assert.equal(resolution.effectiveModel, "Qwen/Qwen3.5-0.8B");
  assert.equal(resolution.remapped, false);
});

test("adapter mode remaps the canonical model to the configured backend model", () => {
  const resolution = resolveRuntimeModel({
    requestedModel: "Qwen/Qwen3.5-0.8B",
    backendModel: "qwen3.5:0.8b",
    canonicalModel: "Qwen/Qwen3.5-0.8B",
    mode: "adapter",
  });

  assert.equal(resolution.requestedModel, "Qwen/Qwen3.5-0.8B");
  assert.equal(resolution.effectiveModel, "qwen3.5:0.8b");
  assert.equal(resolution.remapped, true);
});

test("adapter mode still falls back to the canonical model if no backend override exists", () => {
  const resolution = resolveRuntimeModel({
    backendModel: "",
    canonicalModel: "Qwen/Qwen3.5-0.8B",
    mode: "adapter",
  });

  assert.equal(resolution.effectiveModel, "Qwen/Qwen3.5-0.8B");
  assert.equal(resolution.remapped, false);
});
