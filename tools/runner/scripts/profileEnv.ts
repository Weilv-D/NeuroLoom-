import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const runnerProfiles = ["lmstudio", "ollama", "vllm"] as const;

export type RunnerProfile = (typeof runnerProfiles)[number];

const scriptsDir = path.dirname(fileURLToPath(import.meta.url));
const runnerRoot = path.resolve(scriptsDir, "..");
const examplesDir = path.join(runnerRoot, "examples");

export async function loadProfileEnvironment(profile: RunnerProfile) {
  const filePath = resolveProfileExamplePath(profile);
  const raw = await readFile(filePath, "utf8");
  return parseEnvText(raw);
}

export function mergeProfileEnvironment(profileEnv: Record<string, string>, shellEnv: NodeJS.ProcessEnv = process.env) {
  const merged = { ...profileEnv };
  for (const [key, value] of Object.entries(shellEnv)) {
    if (typeof value !== "string" || !value) {
      continue;
    }
    if (key.startsWith("NEUROLOOM_") || key === "OPENAI_API_KEY") {
      merged[key] = value;
    }
  }
  return merged;
}

export function parseProfileArg(raw: string | undefined): RunnerProfile {
  if (raw && isRunnerProfile(raw)) {
    return raw;
  }
  throw new Error(`Expected one of: ${runnerProfiles.join(", ")}`);
}

export function resolveProfileExamplePath(profile: RunnerProfile) {
  return path.join(examplesDir, `${profile}.env.example`);
}

export function resolveRunnerRoot() {
  return runnerRoot;
}

export function parseEnvText(raw: string) {
  const env: Record<string, string> = {};
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const separator = trimmed.indexOf("=");
    if (separator <= 0) {
      continue;
    }
    const key = trimmed.slice(0, separator).trim();
    const value = trimmed
      .slice(separator + 1)
      .trim()
      .replace(/^['"]|['"]$/g, "");
    env[key] = value;
  }
  return env;
}

function isRunnerProfile(value: string): value is RunnerProfile {
  return runnerProfiles.includes(value as RunnerProfile);
}
