import assert from "node:assert/strict";
import test from "node:test";

import { matchesRequestedModel, resolveModelsEndpoint } from "../src/backendProbe.js";

test("resolveModelsEndpoint converts chat completions endpoint into models endpoint", () => {
  assert.equal(resolveModelsEndpoint("http://127.0.0.1:8000/v1/chat/completions"), "http://127.0.0.1:8000/v1/models");
  assert.equal(resolveModelsEndpoint("http://127.0.0.1:1234/chat/completions"), "http://127.0.0.1:1234/models");
});

test("matchesRequestedModel handles exact and normalized qwen ids", () => {
  assert.equal(matchesRequestedModel("Qwen/Qwen3.5-0.8B", ["Qwen/Qwen3.5-0.8B"]), true);
  assert.equal(matchesRequestedModel("Qwen/Qwen3.5-0.8B", ["qwen3.5-0.8b"]), true);
  assert.equal(matchesRequestedModel("Qwen/Qwen3.5-0.8B", ["models/qwen3.5-0.8b"]), true);
  assert.equal(matchesRequestedModel("Qwen/Qwen3.5-0.8B", ["llama3.2"]), false);
});
