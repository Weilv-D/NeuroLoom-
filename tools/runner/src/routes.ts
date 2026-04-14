import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

import { qwenRunnerModelId } from "@neuroloom/official-traces";

import { probeBackend } from "./backendProbe.js";
import { resolveRuntimeModel, type RunnerMode } from "./modelRouting.js";
import type { SessionStore } from "./sessionManager.js";
import type { ChatCompletionRequest } from "./types.js";

export function setCorsHeaders(response: ServerResponse) {
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

export function sendJson(response: ServerResponse, statusCode: number, payload: unknown) {
  response.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

export async function readJsonBody(request: IncomingMessage) {
  const chunks: Uint8Array[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw ? JSON.parse(raw) : {};
}

export function extractPrompt(messages: ChatCompletionRequest["messages"]) {
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

export async function routeRequest(
  request: IncomingMessage,
  response: ServerResponse,
  context: {
    store: SessionStore;
    mode: RunnerMode;
    backendModel: string;
    backendProfile: import("./backendProfiles.js").BackendProfile;
    backendThink: boolean | string | undefined;
    backendUrl: string;
    backendApiKey: string;
    backendStreamingRequested: boolean;
  },
) {
  setCorsHeaders(response);
  if (request.method === "OPTIONS") {
    response.writeHead(204);
    response.end();
    return;
  }

  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "127.0.0.1"}`);
  const { store, mode, backendModel, backendProfile, backendThink, backendUrl, backendApiKey, backendStreamingRequested } = context;

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
      backendThink: backendThink ?? null,
      sessions: store.sessions.size,
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
    const recorder = new (await import("@neuroloom/official-traces")).QwenSessionRecorder({ sessionId, prompt });
    const session: import("./types.js").SessionRecord = {
      id: sessionId,
      prompt,
      recorder,
      events: [],
      sockets: new Set(),
      archive: null,
      status: "booting",
      error: null,
      completion: "",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      abortController: new AbortController(),
      finishReason: "completed",
    };
    store.sessions.set(sessionId, session);
    store.trimSessions();

    const startEvent = recorder.createStartEvent();
    session.events.push(startEvent);
    session.status = "live";
    store.broadcastEvent(session, startEvent);
    void store.runSession(session, body, modelResolution.effectiveModel);

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
        websocket_url: `ws://${request.headers.host ?? `127.0.0.1:${store.runnerPort}`}/live/${sessionId}`,
        trace_url: `http://${request.headers.host ?? `127.0.0.1:${store.runnerPort}`}/sessions/${sessionId}/trace`,
        requested_model: modelResolution.requestedModel,
        effective_model: modelResolution.effectiveModel,
        model_remapped: modelResolution.remapped,
      },
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/sessions") {
    sendJson(response, 200, {
      sessions: [...store.sessions.values()]
        .sort((left, right) => right.createdAt - left.createdAt)
        .map((session) => store.serializeSession(session)),
    });
    return;
  }

  const traceMatch = url.pathname.match(/^\/sessions\/([^/]+)\/trace$/);
  if (request.method === "GET" && traceMatch) {
    const session = store.sessions.get(traceMatch[1]);
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
    const session = store.sessions.get(sessionMatch[1]);
    if (!session) {
      sendJson(response, 404, { error: "Unknown session id." });
      return;
    }
    sendJson(response, 200, store.serializeSession(session));
    return;
  }

  const cancelMatch = url.pathname.match(/^\/sessions\/([^/]+)\/cancel$/);
  if (request.method === "POST" && cancelMatch) {
    const session = store.sessions.get(cancelMatch[1]);
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
    sendJson(response, 202, store.serializeSession(session));
    return;
  }

  sendJson(response, 404, { error: "Not found." });
}
