import type { QwenLiveEvent } from "@neuroloom/official-traces";

const defaultRunnerUrl = ((import.meta.env as { VITE_NEUROLOOM_RUNNER_URL?: string }).VITE_NEUROLOOM_RUNNER_URL ??
  "http://127.0.0.1:7778") as string;

export type RunnerHealth = {
  ok: boolean;
  mode: "synthetic" | "adapter";
  model: string;
  backendModel: string;
  liveEndpoint: string;
};

export type SessionStartResponse = {
  id: string;
  neuroloom: {
    session_id: string;
    websocket_url: string;
    trace_url: string;
  };
};

export async function checkRunnerHealth(): Promise<RunnerHealth | null> {
  try {
    const response = await fetch(`${defaultRunnerUrl}/health`);
    if (!response.ok) return null;
    return (await response.json()) as RunnerHealth;
  } catch {
    return null;
  }
}

export async function startChatSession(prompt: string): Promise<SessionStartResponse> {
  const response = await fetch(`${defaultRunnerUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "Qwen/Qwen3.5-0.8B",
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Runner refused the session: ${response.status} ${text}`);
  }
  return (await response.json()) as SessionStartResponse;
}

export function connectToSession(
  sessionId: string,
  handlers: {
    onEvent(event: QwenLiveEvent): void;
    onError(message: string): void;
    onClose?(): void;
  },
) {
  const baseWsUrl = defaultRunnerUrl.replace(/^http/, "ws");
  const socket = new WebSocket(`${baseWsUrl}/live/${sessionId}`);

  socket.addEventListener("message", (message) => {
    try {
      handlers.onEvent(JSON.parse(message.data as string) as QwenLiveEvent);
    } catch (error) {
      handlers.onError((error as Error).message);
    }
  });
  socket.addEventListener("error", () => {
    handlers.onError("Runner WebSocket connection failed.");
  });
  socket.addEventListener("close", () => {
    handlers.onClose?.();
  });

  return () => socket.close();
}

export async function downloadTraceFromRunner(traceUrl: string): Promise<Uint8Array> {
  const response = await fetch(traceUrl);
  if (!response.ok) {
    throw new Error(`Failed to download session trace: ${response.status} ${response.statusText}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}
