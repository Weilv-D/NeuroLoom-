import { spawn } from "node:child_process";

import { loadProfileEnvironment, mergeProfileEnvironment, parseProfileArg, resolveRunnerRoot } from "./profileEnv.js";

const args = process.argv.slice(2);
const profile = parseProfileArg(args[0]);
const useBuiltServer = args.includes("--start");

const baseEnv = await loadProfileEnvironment(profile);
const mergedEnv = mergeProfileEnvironment(baseEnv);
const runnerRoot = resolveRunnerRoot();

console.log(`NeuroLoom Runner profile: ${profile}`);
console.log(`Backend URL: ${mergedEnv.NEUROLOOM_BACKEND_URL ?? "unset"}`);
console.log(`Backend model: ${mergedEnv.NEUROLOOM_BACKEND_MODEL ?? "unset"}`);
console.log(`Streaming: ${mergedEnv.NEUROLOOM_BACKEND_STREAM ?? "true"}`);

const child = spawn(resolvePnpmBinary(), useBuiltServer ? ["start"] : ["exec", "tsx", "watch", "src/server.ts"], {
  cwd: runnerRoot,
  env: {
    ...process.env,
    ...mergedEnv,
  },
  stdio: "inherit",
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

function resolvePnpmBinary() {
  return process.platform === "win32" ? "pnpm.cmd" : "pnpm";
}
