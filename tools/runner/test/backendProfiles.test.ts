import assert from "node:assert/strict";
import test from "node:test";

import { detectBackendProfile, resolveBackendEndpoint } from "../src/backendProfiles.js";

test("detectBackendProfile falls back to synthetic with no backend url", () => {
  const profile = detectBackendProfile("");
  assert.equal(profile.provider, "synthetic");
  assert.equal(profile.endpoint, null);
  assert.equal(profile.detectedFrom, "synthetic");
});

test("detectBackendProfile infers LM Studio from the default local port", () => {
  const profile = detectBackendProfile("http://127.0.0.1:1234");
  assert.equal(profile.provider, "lmstudio");
  assert.equal(profile.endpoint, "http://127.0.0.1:1234/v1/chat/completions");
});

test("detectBackendProfile infers Ollama from the default local port", () => {
  const profile = detectBackendProfile("http://127.0.0.1:11434");
  assert.equal(profile.provider, "ollama");
  assert.equal(profile.endpoint, "http://127.0.0.1:11434/v1/chat/completions");
});

test("detectBackendProfile allows an explicit provider override", () => {
  const profile = detectBackendProfile("http://127.0.0.1:9000", "ollama");
  assert.equal(profile.provider, "ollama");
  assert.equal(profile.detectedFrom, "override");
});

test("resolveBackendEndpoint preserves an explicit chat completions path", () => {
  assert.equal(resolveBackendEndpoint("http://127.0.0.1:8000/v1/chat/completions"), "http://127.0.0.1:8000/v1/chat/completions");
  assert.equal(resolveBackendEndpoint("http://127.0.0.1:11434/chat/completions"), "http://127.0.0.1:11434/chat/completions");
});
