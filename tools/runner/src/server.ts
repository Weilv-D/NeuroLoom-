import { randomUUID } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";

import { createLoomTraceArchive } from "@neuroloom/core";
import {
  QwenSessionRecorder,
  buildSyntheticQwenResponse,
  qwenRunnerModelId,
  tokenizeCompletion,
  type QwenLiveEvent,
} from "@neuroloom/official-traces";
import { WebSocket, WebSocketServer } from "ws";

import { probeBackend } from "./backendProbe.js";
import { detectBackendProfile, resolveBackendEndpoint } from "./backendProfiles.js";
import { resolveRuntimeModel, type RunnerMode } from "./modelRouting.js";

type ChatCompletionRequest = {
  model?: string;
  messages?: Array<{ role?: string; content?: string | Array<{ type?: string; text?: string }> }>;
  max_tokens?: number;
  temperature?: number;
};

type SessionRecord = {
  id: string;
  prompt: string;
  recorder: QwenSessionRecorder;
  events: QwenLiveEvent[];
  sockets: Set<WebSocket>;
  archive: Uint8Array | null;
  status: "booting" | "live" | "complete" | "error" | "cancelled";
  error: string | null;
  completion: string;
  createdAt: number;
  updatedAt: number;
  abortController: AbortController;
  finishReason: "completed" | "cancelled" | "error";
};

const runnerPort = Number(process.env.NEUROLOOM_RUNNER_PORT ?? "7778");
const backendUrl = process.env.NEUROLOOM_BACKEND_URL?.trim() ?? "";
const backendApiKey = process.env.NEUROLOOM_BACKEND_API_KEY?.trim() ?? process.env.OPENAI_API_KEY?.trim() ?? "";
const backendModel = process.env.NEUROLOOM_BACKEND_MODEL?.trim() ?? qwenRunnerModelId;
const backendStreamingRequested = process.env.NEUROLOOM_BACKEND_STREAM?.trim() !== "false";
const backendProvider = process.env.NEUROLOOM_BACKEND_PROVIDER?.trim() ?? "";
const mode: RunnerMode = backendUrl ? "adapter" : "synthetic";
const sessionRetention = Number(process.env.NEUROLOOM_SESSION_RETENTION ?? "12");
const backendProfile = detectBackendProfile(backendUrl, backendProvider);

const sessions = new Map<string, SessionRecord>();

const server = createServer((request, response) => {
  void routeRequest(request, response).catch((error) => {
    console.error("runner request failed", error);
    sendJson(response, 500, { error: (error as Error).message });
  });
});

const wss = new WebSocketServer({ noServer: true });

server.on("upgrade", (request, socket, head) => {
  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "127.0.0.1"}`);
  const match = url.pathname.match(/^\/live\/([^/]+)$/);
  if (!match) {
    socket.destroy();
    return;
  }

  const session = sessions.get(match[1]);
  if (!session) {
    socket.destroy();
    return;
  }

  wss.handleUpgrade(request, socket, head, (websocket) => {
    wss.emit("connection", websocket, request, session.id);
  });
});

wss.on("connection", (socket: WebSocket, _request: IncomingMessage, sessionId: string) => {
  const session = sessions.get(sessionId);
  if (!session) {
    socket.close();
    return;
  }

  session.sockets.add(socket);
  for (const event of session.events) {
    socket.send(JSON.stringify(event));
  }

  socket.on("close", () => {
    session.sockets.delete(socket);
  });
});

server.listen(runnerPort, "127.0.0.1", () => {
  console.log(`NeuroLoom Runner listening on http://127.0.0.1:${runnerPort} (${mode})`);
  if (mode === "adapter") {
    console.log(`Adapter target: ${backendProfile.label} -> ${backendProfile.endpoint}`);
  }
});

async function routeRequest(request: IncomingMessage, response: ServerResponse) {
  setCorsHeaders(response);
  if (request.method === "OPTIONS") {
    response.writeHead(204);
    response.end();
    return;
  }

  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "127.0.0.1"}`);

  if (request.method === "GET" && url.pathname === "/health") {
    const modelResolution = resolveRuntimeModel({
      backendModel,
      canonicalModel: qwenRunnerModelId,
      mode,
    });
    sendJson(response, 200, {
      ok: true,
      mode,
      model: qwenRunnerModelId,
      backendModel: backendModel,
      effectiveModel: modelResolution.effectiveModel,
      modelRemapped: modelResolution.remapped,
      streaming: Boolean(backendUrl) && backendStreamingRequested,
      backendUrl: backendUrl || null,
      backendEndpoint: backendProfile.endpoint,
      backendProvider: backendProfile.provider,
      backendLabel: backendProfile.label,
      backendDetectedFrom: backendProfile.detectedFrom,
      backendSetupHint: backendProfile.setupHint,
      sessions: sessions.size,
      liveEndpoint: `/live/:sessionId`,
      probeEndpoint: `/backend/probe`,
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/backend/probe") {
    sendJson(
      response,
      200,
      await probeBackend({
        profile: backendProfile,
        apiKey: backendApiKey,
        targetModel: backendModel,
      }),
    );
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
    const body = (await readJsonBody(request)) as ChatCompletionRequest;
    const prompt = extractPrompt(body.messages);
    if (!prompt) {
      sendJson(response, 400, { error: "Expected a user message with text content." });
      return;
    }
    const modelResolution = resolveRuntimeModel({
      requestedModel: body.model,
      backendModel,
      canonicalModel: qwenRunnerModelId,
      mode,
    });

    const sessionId = `session-${randomUUID().replace(/-/g, "").slice(0, 12)}`;
    const recorder = new QwenSessionRecorder({ sessionId, prompt });
    const session: SessionRecord = {
      id: sessionId,
      prompt,
      recorder,
      events: [],
      sockets: new Set<WebSocket>(),
      archive: null,
      status: "booting",
      error: null,
      completion: "",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      abortController: new AbortController(),
      finishReason: "completed",
    };
    sessions.set(sessionId, session);
    trimSessions();

    const startEvent = recorder.createStartEvent();
    session.events.push(startEvent);
    session.status = "live";
    broadcastEvent(session, startEvent);
    void runSession(session, body, modelResolution.effectiveModel);

    sendJson(response, 200, {
      id: `chatcmpl_${sessionId}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: modelResolution.effectiveModel,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "" },
          finish_reason: null,
        },
      ],
      neuroloom: {
        session_id: sessionId,
        websocket_url: `ws://127.0.0.1:${runnerPort}/live/${sessionId}`,
        trace_url: `http://127.0.0.1:${runnerPort}/sessions/${sessionId}/trace`,
        requested_model: modelResolution.requestedModel,
        effective_model: modelResolution.effectiveModel,
        model_remapped: modelResolution.remapped,
      },
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/sessions") {
    sendJson(response, 200, {
      sessions: [...sessions.values()].sort((left, right) => right.createdAt - left.createdAt).map((session) => serializeSession(session)),
    });
    return;
  }

  const traceMatch = url.pathname.match(/^\/sessions\/([^/]+)\/trace$/);
  if (request.method === "GET" && traceMatch) {
    const session = sessions.get(traceMatch[1]);
    if (!session) {
      sendJson(response, 404, { error: "Unknown session id." });
      return;
    }
    if (!session.archive) {
      sendJson(response, 409, { error: "Session replay is not ready yet." });
      return;
    }
    response.writeHead(200, {
      "Content-Type": "application/octet-stream",
      "Content-Disposition": `attachment; filename="${session.id}.loomtrace"`,
      "Content-Length": String(session.archive.byteLength),
    });
    response.end(Buffer.from(session.archive));
    return;
  }

  const sessionMatch = url.pathname.match(/^\/sessions\/([^/]+)$/);
  if (request.method === "GET" && sessionMatch) {
    const session = sessions.get(sessionMatch[1]);
    if (!session) {
      sendJson(response, 404, { error: "Unknown session id." });
      return;
    }
    sendJson(response, 200, serializeSession(session));
    return;
  }

  const cancelMatch = url.pathname.match(/^\/sessions\/([^/]+)\/cancel$/);
  if (request.method === "POST" && cancelMatch) {
    const session = sessions.get(cancelMatch[1]);
    if (!session) {
      sendJson(response, 404, { error: "Unknown session id." });
      return;
    }
    if (session.status !== "live" && session.status !== "booting") {
      sendJson(response, 409, { error: `Session is already ${session.status}.` });
      return;
    }
    session.finishReason = "cancelled";
    session.abortController.abort("Session cancelled by user");
    session.updatedAt = Date.now();
    sendJson(response, 202, serializeSession(session));
    return;
  }

  sendJson(response, 404, { error: "Not found." });
}

async function runSession(session: SessionRecord, request: ChatCompletionRequest, effectiveModel: string) {
  try {
    if (!backendUrl) {
      const completionText = buildSyntheticQwenResponse(session.prompt);
      await emitCompletionAsTokens(session, completionText, true);
      return;
    }

    if (backendStreamingRequested) {
      const completionText = await streamBackendCompletion(session, request, effectiveModel);
      if (!completionText.trim()) {
        throw new Error("Streaming adapter produced an empty completion.");
      }
      if (session.status !== "complete") {
        await finishSession(session);
      }
      return;
    }

    const completionText = await resolveBufferedCompletionText(session.prompt, request, effectiveModel);
    await emitCompletionAsTokens(session, completionText, false);
  } catch (error) {
    const reason = session.abortController.signal.aborted ? "cancelled" : "error";
    session.status = reason;
    session.finishReason = reason;
    session.error = reason === "cancelled" ? null : (error as Error).message;
    session.updatedAt = Date.now();
    const fallbackText = buildSyntheticQwenResponse(session.prompt);
    if (reason === "cancelled") {
      await finishSession(session);
      return;
    }
    await emitCompletionAsTokens(session, fallbackText, true);
  }
}

async function emitCompletionAsTokens(session: SessionRecord, completionText: string, syntheticDelay: boolean) {
  session.completion = completionText;
  const tokens = tokenizeCompletion(completionText);
  for (const token of tokens) {
    if (session.abortController.signal.aborted) {
      break;
    }
    if (syntheticDelay) {
      await sleep(stepDelay(token));
    }
    if (session.abortController.signal.aborted) {
      break;
    }
    const event = session.recorder.pushToken(token);
    session.events.push(event);
    session.updatedAt = Date.now();
    broadcastEvent(session, event);
  }
  await finishSession(session);
}

async function finishSession(session: SessionRecord) {
  if (session.archive) {
    return;
  }
  const completed = session.recorder.complete();
  session.events.push(completed);
  session.archive = await createLoomTraceArchive(session.recorder.exportBundle());
  session.status = session.finishReason === "cancelled" ? "cancelled" : session.finishReason === "error" ? "error" : "complete";
  session.updatedAt = Date.now();
  broadcastEvent(session, completed);
}

async function streamBackendCompletion(session: SessionRecord, request: ChatCompletionRequest, effectiveModel: string) {
  const response = await fetch(resolveBackendEndpoint(backendUrl), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(backendApiKey ? { Authorization: `Bearer ${backendApiKey}` } : {}),
    },
    body: JSON.stringify({
      model: effectiveModel,
      messages: request.messages,
      max_tokens: request.max_tokens ?? 160,
      temperature: request.temperature ?? 0.7,
      stream: true,
    }),
  });
  if (!response.ok) {
    throw new Error(`Adapter backend failed: ${response.status} ${response.statusText}`);
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("text/event-stream") || !response.body) {
    const buffered = await extractBufferedCompletionFromResponse(response);
    await emitCompletionAsTokens(session, buffered, false);
    return buffered;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completion = "";
  let emittedTokenCount = 0;
  let isDone = false;

  while (!isDone) {
    if (session.abortController.signal.aborted) {
      await reader.cancel();
      break;
    }
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const rawEvent of events) {
      const parsed = parseSseEvent(rawEvent);
      if (!parsed) continue;
      if (parsed.done) {
        isDone = true;
        break;
      }
      if (!parsed.content) continue;
      completion += parsed.content;
      session.completion = completion;
      session.updatedAt = Date.now();
      emittedTokenCount = emitFreshTokens(session, completion, emittedTokenCount);
    }
  }

  if (buffer.trim()) {
    const parsed = parseSseEvent(buffer);
    if (parsed?.content) {
      completion += parsed.content;
      session.completion = completion;
      session.updatedAt = Date.now();
      emittedTokenCount = emitFreshTokens(session, completion, emittedTokenCount);
    }
  }

  emitFreshTokens(session, completion, emittedTokenCount, true);
  session.completion = completion;
  return completion;
}

async function resolveBufferedCompletionText(_prompt: string, request: ChatCompletionRequest, effectiveModel: string) {
  const endpoint = resolveBackendEndpoint(backendUrl);
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(backendApiKey ? { Authorization: `Bearer ${backendApiKey}` } : {}),
    },
    body: JSON.stringify({
      model: effectiveModel,
      messages: request.messages,
      max_tokens: request.max_tokens ?? 160,
      temperature: request.temperature ?? 0.7,
      stream: false,
    }),
  });
  if (!response.ok) {
    throw new Error(`Adapter backend failed: ${response.status} ${response.statusText}`);
  }
  return extractBufferedCompletionFromResponse(response);
}

function extractPrompt(messages: ChatCompletionRequest["messages"]) {
  if (!messages || messages.length === 0) return "";
  const lastUserMessage = [...messages].reverse().find((message) => message.role === "user");
  if (!lastUserMessage) return "";
  if (typeof lastUserMessage.content === "string") {
    return lastUserMessage.content;
  }
  if (Array.isArray(lastUserMessage.content)) {
    return lastUserMessage.content
      .map((part) => part.text ?? "")
      .join("")
      .trim();
  }
  return "";
}

function broadcastEvent(session: SessionRecord, event: QwenLiveEvent) {
  const payload = JSON.stringify(event);
  for (const socket of session.sockets) {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(payload);
    }
  }
}

async function extractBufferedCompletionFromResponse(response: Response) {
  const json = (await response.json()) as {
    choices?: Array<{
      delta?: { content?: string | Array<{ text?: string; type?: string }> };
      message?: { content?: string | Array<{ text?: string; type?: string }> };
    }>;
  };
  const choice = json.choices?.[0];
  const content = choice?.message?.content ?? choice?.delta?.content;
  const extracted = extractContentString(content);
  if (extracted.trim()) {
    return extracted;
  }
  throw new Error("Adapter backend returned an empty completion.");
}

function parseSseEvent(rawEvent: string) {
  const data = rawEvent
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n")
    .trim();
  if (!data) {
    return null;
  }
  if (data === "[DONE]") {
    return { done: true, content: "" };
  }
  const json = JSON.parse(data) as {
    choices?: Array<{
      delta?: { content?: string | Array<{ text?: string; type?: string }> };
      message?: { content?: string | Array<{ text?: string; type?: string }> };
      finish_reason?: string | null;
    }>;
  };
  const choice = json.choices?.[0];
  return {
    done: choice?.finish_reason === "stop",
    content: extractContentString(choice?.delta?.content ?? choice?.message?.content),
  };
}

function extractContentString(content: string | Array<{ text?: string; type?: string }> | undefined) {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map((part) => part.text ?? "").join("");
  }
  return "";
}

function emitFreshTokens(session: SessionRecord, completion: string, emittedTokenCount: number, flushLast = false) {
  const nextTokens = tokenizeCompletion(completion);
  const readyCount = flushLast ? nextTokens.length : completedTokenCount(completion, nextTokens.length);
  for (let index = emittedTokenCount; index < readyCount; index++) {
    if (session.abortController.signal.aborted) {
      break;
    }
    const token = nextTokens[index];
    if (!token) continue;
    const event = session.recorder.pushToken(token);
    session.events.push(event);
    session.updatedAt = Date.now();
    broadcastEvent(session, event);
  }
  return session.abortController.signal.aborted ? emittedTokenCount : readyCount;
}

function completedTokenCount(completion: string, tokenCount: number) {
  const trailing = completion.at(-1) ?? "";
  if (!trailing || /\s/.test(trailing) || /[.,!?;:)\]"'`]/.test(trailing)) {
    return tokenCount;
  }
  return Math.max(0, tokenCount - 1);
}

async function readJsonBody(request: IncomingMessage) {
  const chunks: Uint8Array[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw ? JSON.parse(raw) : {};
}

function setCorsHeaders(response: ServerResponse) {
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

function sendJson(response: ServerResponse, statusCode: number, payload: unknown) {
  response.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

function stepDelay(token: string) {
  return Math.max(72, Math.min(180, 86 + token.length * 9));
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function serializeSession(session: SessionRecord) {
  return {
    id: session.id,
    prompt: session.prompt,
    status: session.status,
    finishReason: session.finishReason,
    error: session.error,
    events: session.events.length,
    completion: session.completion,
    archiveReady: Boolean(session.archive),
    tokenCount: session.events.filter((event) => event.type === "token_step").length,
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    traceUrl: `http://127.0.0.1:${runnerPort}/sessions/${session.id}/trace`,
  };
}

function trimSessions() {
  if (sessions.size <= sessionRetention) {
    return;
  }
  const removable = [...sessions.values()]
    .sort((left, right) => left.createdAt - right.createdAt)
    .slice(0, sessions.size - sessionRetention);
  for (const session of removable) {
    if (session.status === "live" || session.status === "booting") {
      continue;
    }
    sessions.delete(session.id);
  }
}
