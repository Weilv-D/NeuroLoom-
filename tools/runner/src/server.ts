import { createServer, type IncomingMessage } from "node:http";

import { qwenRunnerModelId } from "@neuroloom/official-traces";
import { WebSocket, WebSocketServer } from "ws";

import { detectBackendProfile } from "./backendProfiles.js";
import { resolveRuntimeModel, type RunnerMode } from "./modelRouting.js";
import { routeRequest, sendJson } from "./routes.js";
import { resolveBackendThinkSetting } from "./sseParser.js";
import { SessionStore } from "./sessionManager.js";

const runnerPort = Number(process.env.NEUROLOOM_RUNNER_PORT ?? "7778");
const backendUrl = process.env.NEUROLOOM_BACKEND_URL?.trim() ?? "";
const backendApiKey = process.env.NEUROLOOM_BACKEND_API_KEY?.trim() ?? process.env.OPENAI_API_KEY?.trim() ?? "";
const backendModel = process.env.NEUROLOOM_BACKEND_MODEL?.trim() ?? qwenRunnerModelId;
const backendStreamingRequested = process.env.NEUROLOOM_BACKEND_STREAM?.trim() !== "false";
const backendProvider = process.env.NEUROLOOM_BACKEND_PROVIDER?.trim() ?? "";
const mode: RunnerMode = backendUrl ? "adapter" : "synthetic";
const sessionRetention = Number(process.env.NEUROLOOM_SESSION_RETENTION ?? "12");
const backendProfile = detectBackendProfile(backendUrl, backendProvider);
const backendThink = resolveBackendThinkSetting(process.env.NEUROLOOM_BACKEND_THINK, backendProfile.provider);

const store = new SessionStore({
  retention: sessionRetention,
  runnerPort,
  backendUrl,
  backendApiKey,
  backendThink,
  backendStreamingRequested,
});

const server = createServer((request, response) => {
  void routeRequest(request, response, {
    store,
    mode,
    backendModel,
    backendProfile,
    backendThink,
    backendUrl,
    backendApiKey,
    backendStreamingRequested,
  }).catch((error) => {
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

  const session = store.sessions.get(match[1]);
  if (!session) {
    socket.destroy();
    return;
  }

  wss.handleUpgrade(request, socket, head, (websocket) => {
    wss.emit("connection", websocket, request, session.id);
  });
});

wss.on("connection", (socket: WebSocket, _request: IncomingMessage, sessionId: string) => {
  const session = store.sessions.get(sessionId);
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

const runnerHost = process.env.NEUROLOOM_RUNNER_HOST ?? "::"; // Listen on both IPv6 and IPv4

server.listen(runnerPort, runnerHost, () => {
  console.log(`NeuroLoom Runner listening on http://${runnerHost === "::" ? "localhost" : runnerHost}:${runnerPort} (${mode})`);
  if (mode === "adapter") {
    console.log(`Adapter target: ${backendProfile.label} -> ${backendProfile.endpoint}`);
  }
});
