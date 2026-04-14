export type BackendProvider = "synthetic" | "lmstudio" | "ollama" | "vllm" | "custom";

export type BackendProfile = {
  provider: BackendProvider;
  label: string;
  detectedFrom: "synthetic" | "override" | "url";
  baseUrl: string | null;
  endpoint: string | null;
  setupHint: string;
};

const providerLabels: Record<BackendProvider, string> = {
  synthetic: "Synthetic Demo",
  lmstudio: "LM Studio",
  ollama: "Ollama",
  vllm: "vLLM",
  custom: "OpenAI-Compatible",
};

const providerHints: Record<Exclude<BackendProvider, "synthetic">, string> = {
  lmstudio: "Launch the OpenAI-compatible local server in LM Studio, then point NeuroLoom at http://127.0.0.1:1234.",
  ollama: "Start Ollama's OpenAI-compatible server and load the Qwen model, then point NeuroLoom at http://127.0.0.1:11434.",
  vllm: "Serve Qwen with vLLM's OpenAI entrypoint, then point NeuroLoom at http://127.0.0.1:8000.",
  custom: "Use any OpenAI-compatible chat endpoint that can stream SSE token deltas.",
};

export function detectBackendProfile(rawUrl: string, explicitProviderRaw?: string): BackendProfile {
  const url = rawUrl.trim();
  if (!url) {
    return {
      provider: "synthetic",
      label: providerLabels.synthetic,
      detectedFrom: "synthetic",
      baseUrl: null,
      endpoint: null,
      setupHint: "No backend URL configured. NeuroLoom will use the built-in synthetic Qwen session source.",
    };
  }

  const explicitProvider = normalizeProvider(explicitProviderRaw);
  const provider = explicitProvider ?? detectProviderFromUrl(url);
  return {
    provider,
    label: providerLabels[provider],
    detectedFrom: explicitProvider ? "override" : "url",
    baseUrl: url,
    endpoint: resolveBackendEndpoint(url),
    setupHint: providerHints[provider] ?? providerHints.custom,
  };
}

export function resolveBackendEndpoint(rawUrl: string) {
  const trimmed = rawUrl.trim();
  if (!trimmed) {
    return "";
  }
  if (/(?:\/v1)?\/chat\/completions\/?$/.test(trimmed)) {
    return trimmed;
  }
  return `${trimmed.replace(/\/$/, "")}/v1/chat/completions`;
}

function normalizeProvider(providerRaw?: string | null): Exclude<BackendProvider, "synthetic"> | null {
  const value = providerRaw?.trim().toLowerCase();
  if (!value) {
    return null;
  }
  if (value === "lmstudio" || value === "ollama" || value === "vllm" || value === "custom") {
    return value;
  }
  if (value === "openai" || value === "openai-compatible" || value === "openai_compatible") {
    return "custom";
  }
  return null;
}

function detectProviderFromUrl(rawUrl: string): Exclude<BackendProvider, "synthetic"> {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return "custom";
  }

  const host = parsed.hostname.toLowerCase();
  const path = parsed.pathname.toLowerCase();
  const port = parsed.port;
  const full = `${host}${path}`;

  if (host.includes("lmstudio") || port === "1234") {
    return "lmstudio";
  }
  if (host.includes("ollama") || port === "11434" || path.startsWith("/api/")) {
    return "ollama";
  }
  if (host.includes("vllm") || port === "8000") {
    return "vllm";
  }
  return "custom";
}
