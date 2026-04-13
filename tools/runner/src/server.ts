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

type RunnerMode = "synthetic" | "adapter";

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
  status: "booting" | "live" | "complete" | "error";
  error: string | null;
};

const runnerPort = Number(process.env.NEUROLOOM_RUNNER_PORT ?? "7778");
const backendUrl = process.env.NEUROLOOM_BACKEND_URL?.trim() ?? "";
const backendApiKey = process.env.NEUROLOOM_BACKEND_API_KEY?.trim() ?? process.env.OPENAI_API_KEY?.trim() ?? "";
const backendModel = process.env.NEUROLOOM_BACKEND_MODEL?.trim() ?? qwenRunnerModelId;
const mode: RunnerMode = backendUrl ? "adapter" : "synthetic";

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
    console.log(`Adapter target: ${backendUrl}`);
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
    sendJson(response, 200, {
      ok: true,
      mode,
      model: qwenRunnerModelId,
      backendModel: backendModel,
      liveEndpoint: `/live/:sessionId`,
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
    const body = (await readJsonBody(request)) as ChatCompletionRequest;
    const prompt = extractPrompt(body.messages);
    if (!prompt) {
      sendJson(response, 400, { error: "Expected a user message with text content." });
      return;
    }

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
    };
    sessions.set(sessionId, session);

    const startEvent = recorder.createStartEvent();
    session.events.push(startEvent);
    session.status = "live";
    broadcastEvent(session, startEvent);
    void runSession(session, body);

    sendJson(response, 200, {
      id: `chatcmpl_${sessionId}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: body.model ?? qwenRunnerModelId,
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
      },
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
    sendJson(response, 200, {
      id: session.id,
      prompt: session.prompt,
      status: session.status,
      error: session.error,
      events: session.events.length,
      archiveReady: Boolean(session.archive),
    });
    return;
  }

  sendJson(response, 404, { error: "Not found." });
}

async function runSession(session: SessionRecord, request: ChatCompletionRequest) {
  try {
    const completionText = await resolveCompletionText(session.prompt, request);
    const tokens = tokenizeCompletion(completionText);

    for (const token of tokens) {
      await sleep(stepDelay(token));
      const event = session.recorder.pushToken(token);
      session.events.push(event);
      broadcastEvent(session, event);
    }

    const completed = session.recorder.complete();
    session.events.push(completed);
    session.archive = await createLoomTraceArchive(session.recorder.exportBundle());
    session.status = "complete";
    broadcastEvent(session, completed);
  } catch (error) {
    session.status = "error";
    session.error = (error as Error).message;
    const fallbackText = buildSyntheticQwenResponse(session.prompt);
    const tokens = tokenizeCompletion(fallbackText);
    for (const token of tokens) {
      await sleep(stepDelay(token));
      const event = session.recorder.pushToken(token);
      session.events.push(event);
      broadcastEvent(session, event);
    }
    const completed = session.recorder.complete();
    session.events.push(completed);
    session.archive = await createLoomTraceArchive(session.recorder.exportBundle());
    session.status = "complete";
    broadcastEvent(session, completed);
  }
}

async function resolveCompletionText(prompt: string, request: ChatCompletionRequest) {
  if (!backendUrl) {
    return buildSyntheticQwenResponse(prompt);
  }

  const endpoint = backendUrl.endsWith("/chat/completions") ? backendUrl : `${backendUrl.replace(/\/$/, "")}/v1/chat/completions`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(backendApiKey ? { Authorization: `Bearer ${backendApiKey}` } : {}),
    },
    body: JSON.stringify({
      model: request.model ?? backendModel,
      messages: request.messages,
      max_tokens: request.max_tokens ?? 160,
      temperature: request.temperature ?? 0.7,
      stream: false,
    }),
  });
  if (!response.ok) {
    throw new Error(`Adapter backend failed: ${response.status} ${response.statusText}`);
  }
  const json = (await response.json()) as {
    choices?: Array<{ message?: { content?: string | Array<{ text?: string }> } }>;
  };
  const content = json.choices?.[0]?.message?.content;
  if (typeof content === "string" && content.trim()) {
    return content;
  }
  if (Array.isArray(content)) {
    const joined = content
      .map((part) => part.text ?? "")
      .join("")
      .trim();
    if (joined) {
      return joined;
    }
  }
  throw new Error("Adapter backend returned an empty completion.");
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
