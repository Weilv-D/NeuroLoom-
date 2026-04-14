import { probeBackend } from "../src/backendProbe.js";
import { detectBackendProfile } from "../src/backendProfiles.js";
import { loadProfileEnvironment, mergeProfileEnvironment, parseProfileArg } from "./profileEnv.js";

const args = process.argv.slice(2);
const profile = parseProfileArg(args[0]);

const baseEnv = await loadProfileEnvironment(profile);
const mergedEnv = mergeProfileEnvironment(baseEnv);
const backendUrl = mergedEnv.NEUROLOOM_BACKEND_URL ?? "";
const backendProvider = mergedEnv.NEUROLOOM_BACKEND_PROVIDER ?? profile;
const backendModel = mergedEnv.NEUROLOOM_BACKEND_MODEL ?? "Qwen/Qwen3.5-0.8B";
const backendApiKey = mergedEnv.NEUROLOOM_BACKEND_API_KEY ?? mergedEnv.OPENAI_API_KEY ?? "";

const detected = detectBackendProfile(backendUrl, backendProvider);
const result = await probeBackend({
  profile: detected,
  apiKey: backendApiKey,
  targetModel: backendModel,
});

console.log(`NeuroLoom backend doctor: ${profile}`);
console.log(`Provider: ${result.label}`);
console.log(`Target model: ${result.targetModel}`);
console.log(`Endpoint: ${result.endpoint ?? "n/a"}`);
console.log(`Models endpoint: ${result.modelsEndpoint ?? "n/a"}`);
console.log(`Reachable: ${result.reachable ? "yes" : "no"}`);
console.log(`Model match: ${result.matchedModel ? "yes" : "no"}`);
console.log(`Hint: ${result.error ?? result.hint}`);
if (result.models.length > 0) {
  console.log("Reported models:");
  for (const model of result.models) {
    console.log(`- ${model}`);
  }
}

process.exit(result.ok && result.reachable ? 0 : 1);
