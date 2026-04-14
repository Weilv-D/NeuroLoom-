import type { BackendProfile, BackendProvider } from "./backendProfiles.js";

export type BackendProbeResult = {
  ok: boolean;
  provider: BackendProvider;
  label: string;
  checkedAt: number;
  targetModel: string;
  matchedModel: boolean;
  reachable: boolean;
  endpoint: string | null;
  modelsEndpoint: string | null;
  models: string[];
  statusCode: number | null;
  error: string | null;
  hint: string;
};

export async function probeBackend(input: { profile: BackendProfile; apiKey: string; targetModel: string }): Promise<BackendProbeResult> {
  const { profile, apiKey, targetModel } = input;

  if (profile.provider === "synthetic" || !profile.baseUrl || !profile.endpoint) {
    return {
      ok: true,
      provider: profile.provider,
      label: profile.label,
      checkedAt: Date.now(),
      targetModel,
      matchedModel: true,
      reachable: true,
      endpoint: profile.endpoint,
      modelsEndpoint: null,
      models: [],
      statusCode: null,
      error: null,
      hint: "Synthetic mode does not need a remote backend probe.",
    };
  }

  const modelsEndpoint = resolveModelsEndpoint(profile.endpoint);
  const headers = {
    Accept: "application/json",
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  };

  try {
    const primary = await fetch(modelsEndpoint, {
      method: "GET",
      headers,
      signal: AbortSignal.timeout(4_000),
    });

    let models = primary.ok ? await extractModelIds(primary) : [];
    let statusCode = primary.status;

    if (!primary.ok && profile.provider === "ollama") {
      const fallbackEndpoint = resolveOllamaTagsEndpoint(profile.baseUrl);
      const fallback = await fetch(fallbackEndpoint, {
        method: "GET",
        headers,
        signal: AbortSignal.timeout(4_000),
      });
      statusCode = fallback.status;
      models = fallback.ok ? await extractOllamaModelIds(fallback) : [];
      if (!fallback.ok) {
        return createProbeFailure(
          profile,
          targetModel,
          modelsEndpoint,
          fallback.status,
          `Backend probe failed: ${fallback.status} ${fallback.statusText}`,
        );
      }
    } else if (!primary.ok) {
      return createProbeFailure(
        profile,
        targetModel,
        modelsEndpoint,
        primary.status,
        `Backend probe failed: ${primary.status} ${primary.statusText}`,
      );
    }

    const matchedModel = matchesRequestedModel(targetModel, models);
    return {
      ok: true,
      provider: profile.provider,
      label: profile.label,
      checkedAt: Date.now(),
      targetModel,
      matchedModel,
      reachable: true,
      endpoint: profile.endpoint,
      modelsEndpoint,
      models,
      statusCode,
      error: null,
      hint: matchedModel
        ? `${profile.label} is reachable and reports a model compatible with ${targetModel}.`
        : `${profile.label} is reachable, but ${targetModel} was not found in the reported model list.`,
    };
  } catch (error) {
    return createProbeFailure(profile, targetModel, modelsEndpoint, null, normalizeProbeError(error));
  }
}

export function resolveModelsEndpoint(chatEndpoint: string) {
  return chatEndpoint.replace(/\/chat\/completions\/?$/, "/models");
}

export function matchesRequestedModel(targetModel: string, reportedModels: string[]) {
  if (!targetModel.trim() || reportedModels.length === 0) {
    return false;
  }
  const expected = normalizeModelId(targetModel);
  return reportedModels.some((candidate) => {
    const normalized = normalizeModelId(candidate);
    return normalized === expected || normalized.endsWith(expected) || expected.endsWith(normalized);
  });
}

function createProbeFailure(
  profile: BackendProfile,
  targetModel: string,
  modelsEndpoint: string | null,
  statusCode: number | null,
  error: string,
): BackendProbeResult {
  return {
    ok: false,
    provider: profile.provider,
    label: profile.label,
    checkedAt: Date.now(),
    targetModel,
    matchedModel: false,
    reachable: false,
    endpoint: profile.endpoint,
    modelsEndpoint,
    models: [],
    statusCode,
    error,
    hint: profile.setupHint,
  };
}

async function extractModelIds(response: Response) {
  const json = (await response.json()) as {
    data?: Array<{ id?: string | null }>;
  };
  return (json.data ?? []).map((entry) => entry.id ?? "").filter(Boolean);
}

async function extractOllamaModelIds(response: Response) {
  const json = (await response.json()) as {
    models?: Array<{ name?: string | null; model?: string | null }>;
  };
  return (json.models ?? []).flatMap((entry) => [entry.name ?? "", entry.model ?? ""]).filter(Boolean);
}

function resolveOllamaTagsEndpoint(baseUrl: string) {
  return `${baseUrl.replace(/\/$/, "")}/api/tags`;
}

function normalizeModelId(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/^qwen\//, "")
    .replace(/^models\//, "");
}

function normalizeProbeError(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unknown backend probe error.";
}
