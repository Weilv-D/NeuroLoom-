import assert from "node:assert/strict";
import test from "node:test";

import { mergeProfileEnvironment, parseEnvText } from "../scripts/profileEnv.ts";

test("parseEnvText ignores comments and blank lines", () => {
  const env = parseEnvText(`
# comment
NEUROLOOM_BACKEND_PROVIDER=ollama

NEUROLOOM_BACKEND_URL=http://127.0.0.1:11434
NEUROLOOM_BACKEND_MODEL="qwen3.5:0.8b"
`);

  assert.deepEqual(env, {
    NEUROLOOM_BACKEND_PROVIDER: "ollama",
    NEUROLOOM_BACKEND_URL: "http://127.0.0.1:11434",
    NEUROLOOM_BACKEND_MODEL: "qwen3.5:0.8b",
  });
});

test("mergeProfileEnvironment lets shell overrides win for NeuroLoom variables", () => {
  const merged = mergeProfileEnvironment(
    {
      NEUROLOOM_BACKEND_MODEL: "qwen3.5:0.8b",
      NEUROLOOM_BACKEND_URL: "http://127.0.0.1:11434",
    },
    {
      NEUROLOOM_BACKEND_MODEL: "Qwen/Qwen3.5-0.8B",
      PATH: "/usr/bin",
    },
  );

  assert.equal(merged.NEUROLOOM_BACKEND_MODEL, "Qwen/Qwen3.5-0.8B");
  assert.equal(merged.NEUROLOOM_BACKEND_URL, "http://127.0.0.1:11434");
  assert.equal("PATH" in merged, false);
});
