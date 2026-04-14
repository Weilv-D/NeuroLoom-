import { createLoomTraceArchive, ReplayEngine, type TraceBundle, type TraceFrame } from "@neuroloom/core";
import {
  applyLiveTokenStep,
  createQwenOfficialTraceBundle,
  hydrateBundleFromLiveStart,
  type QwenFramePayload,
  type QwenTokenStepEvent,
} from "@neuroloom/official-traces";
import { startTransition, useEffect, useId, useMemo, useRef, useState } from "react";

import { SceneCanvas } from "./SceneCanvas";
import {
  type BackendProbe,
  cancelRunnerSession,
  checkRunnerHealth,
  connectToSession,
  downloadTraceFromRunner,
  listRunnerSessions,
  probeRunnerBackend,
  startChatSession,
  type RunnerHealth,
  type RunnerSession,
} from "./runnerClient";
import { qwenSampleTrace } from "./sampleTraces";
import { loadTraceFromFile, loadTraceFromUrl } from "./traceLoader";
import type { SelectionState } from "./types";

const playbackIntervalMs = 160;
const defaultPrompt = "Explain how NeuroLoom should make a Qwen3.5-0.8B conversation feel like light moving through a dense starfield.";

type SessionMode = "sample" | "connecting" | "live" | "replay";

export function App() {
  const uploadId = useId();
  const stageRef = useRef<HTMLDivElement | null>(null);
  const disconnectRef = useRef<(() => void) | null>(null);
  const liveFollowRef = useRef(true);

  const [bundle, setBundle] = useState<TraceBundle | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selection, setSelection] = useState<SelectionState>(null);
  const [liveFollow, setLiveFollow] = useState(true);
  const [runnerHealth, setRunnerHealth] = useState<RunnerHealth | null>(null);
  const [runnerChecked, setRunnerChecked] = useState(false);
  const [runnerSessions, setRunnerSessions] = useState<RunnerSession[]>([]);
  const [backendProbe, setBackendProbe] = useState<BackendProbe | null>(null);
  const [sessionMode, setSessionMode] = useState<SessionMode>("sample");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [traceUrl, setTraceUrl] = useState<string | null>(null);
  const [promptDraft, setPromptDraft] = useState(defaultPrompt);
  const [activePrompt, setActivePrompt] = useState(defaultPrompt);
  const [assistantText, setAssistantText] = useState("");
  const [statusLine, setStatusLine] = useState("Loading the official Qwen replay…");
  const [loadingLabel, setLoadingLabel] = useState<string | null>("Loading the official Qwen replay…");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    liveFollowRef.current = liveFollow;
  }, [liveFollow]);

  async function refreshRunnerStatus(options?: { includeProbe?: boolean }) {
    const includeProbe = options?.includeProbe ?? true;
    const health = await checkRunnerHealth();
    setRunnerHealth(health);
    setRunnerChecked(true);

    try {
      setRunnerSessions(await listRunnerSessions());
    } catch {
      setRunnerSessions([]);
    }

    if (includeProbe && health?.mode === "adapter") {
      setBackendProbe(await probeRunnerBackend());
      return;
    }

    if (!health || health.mode !== "adapter") {
      setBackendProbe(null);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      setLoadingLabel("Loading the official Qwen replay…");
      try {
        const nextBundle = await loadTraceFromUrl(qwenSampleTrace.path);
        if (cancelled) return;
        setBundle(nextBundle);
        setAssistantText(readLastCompletion(nextBundle));
        setStatusLine("Demo replay ready. Start the local runner for live sessions.");
      } catch {
        const fallbackBundle = createQwenOfficialTraceBundle();
        if (cancelled) return;
        setBundle(fallbackBundle);
        setAssistantText(readLastCompletion(fallbackBundle));
        setStatusLine("Demo replay loaded from the bundled fallback trace.");
      } finally {
        if (!cancelled) {
          setLoadingLabel(null);
        }
      }
      if (!cancelled) {
        await refreshRunnerStatus();
      }
    }

    void boot();
    const healthInterval = window.setInterval(() => {
      if (!cancelled) {
        void refreshRunnerStatus({ includeProbe: false });
      }
    }, 10_000);

    return () => {
      cancelled = true;
      window.clearInterval(healthInterval);
      disconnectRef.current?.();
    };
  }, []);

  useEffect(() => {
    if (!playing || !bundle || bundle.timeline.length === 0) return;
    const intervalId = window.setInterval(() => {
      setFrameIndex((current) => {
        const lastFrame = bundle.timeline.length - 1;
        if (current >= lastFrame) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, playbackIntervalMs);

    return () => window.clearInterval(intervalId);
  }, [playing, bundle]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target && ["TEXTAREA", "INPUT", "BUTTON", "SELECT"].includes(target.tagName)) {
        return;
      }
      if (!bundle || bundle.timeline.length === 0) return;
      if (event.key === " ") {
        event.preventDefault();
        setPlaying((current) => !current);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        stepFrame(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        stepFrame(1);
      } else if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        void exportPng();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [bundle]);

  const engine = useMemo(() => (bundle && bundle.timeline.length > 0 ? new ReplayEngine(bundle) : null), [bundle]);
  const safeFrameIndex = bundle ? clamp(frameIndex, 0, Math.max(bundle.timeline.length - 1, 0)) : 0;
  const frame = engine ? engine.getFrame(safeFrameIndex) : null;
  const currentChapter = frame && engine ? engine.getChapterForFrame(safeFrameIndex) : (bundle?.narrative.chapters[0] ?? null);
  const currentPayload = bundle && frame ? getInspectPayload(bundle, frame) : null;
  const currentMetrics = frame?.metric_refs ?? [];
  const currentTokenIndex = currentPayload?.tokenIndex ?? null;
  const currentTokenWindow = currentPayload?.tokenWindow ?? [];
  const currentTopLogits = currentPayload?.topLogits ?? [];
  const focusBlock = currentPayload ? currentPayload.tokenIndex % 24 : 0;
  const focusDigest = currentPayload ? currentPayload.blockDigest.slice(Math.max(0, focusBlock - 2), Math.min(24, focusBlock + 3)) : [];
  const selectionDetail = describeSelection({ bundle, frame, payload: currentPayload, selection });
  const chapterIndex = currentChapter && bundle ? bundle.narrative.chapters.findIndex((chapter) => chapter.id === currentChapter.id) : -1;
  const currentRunnerSession = sessionId ? (runnerSessions.find((entry) => entry.id === sessionId) ?? null) : null;

  function stepFrame(delta: number) {
    if (!bundle || bundle.timeline.length === 0) return;
    setFrameIndex((current) => clamp(current + delta, 0, bundle.timeline.length - 1));
    setPlaying(false);
    setLiveFollow(false);
  }

  async function loadSampleReplay() {
    disconnectRef.current?.();
    setLoadingLabel("Reloading the official Qwen replay…");
    setError(null);
    setSessionMode("sample");
    setSessionId(null);
    setTraceUrl(null);
    try {
      const nextBundle = await loadTraceFromUrl(qwenSampleTrace.path);
      startTransition(() => {
        setBundle(nextBundle);
        setFrameIndex(0);
        setSelection(null);
        setAssistantText(readLastCompletion(nextBundle));
        setActivePrompt(defaultPrompt);
        setStatusLine("Demo replay ready. Start the local runner for live sessions.");
      });
    } catch {
      const fallbackBundle = createQwenOfficialTraceBundle();
      startTransition(() => {
        setBundle(fallbackBundle);
        setFrameIndex(0);
        setSelection(null);
        setAssistantText(readLastCompletion(fallbackBundle));
        setActivePrompt(defaultPrompt);
        setStatusLine("Demo replay loaded from the bundled fallback trace.");
      });
    } finally {
      setLoadingLabel(null);
    }
  }

  async function startLiveSession() {
    if (!runnerHealth) {
      setError("The local NeuroLoom Runner is not reachable. Start it first, then try again.");
      return;
    }

    disconnectRef.current?.();
    setError(null);
    setPlaying(false);
    setSelection(null);
    setLiveFollow(true);
    setLoadingLabel("Connecting to the local runner…");
    setStatusLine("Creating a live Qwen session…");
    setSessionMode("connecting");
    setActivePrompt(promptDraft);
    setAssistantText("");

    try {
      const response = await startChatSession(promptDraft);
      setSessionId(response.neuroloom.session_id);
      setTraceUrl(response.neuroloom.trace_url);
      disconnectRef.current = connectToSession(response.neuroloom.session_id, {
        onEvent(event) {
          if (event.type === "session_started") {
            startTransition(() => {
              setBundle(hydrateBundleFromLiveStart(event));
              setFrameIndex(0);
              setSessionMode("live");
              setStatusLine("Runner connected. Waiting for the first token pulse…");
            });
            return;
          }

          if (event.type === "token_step") {
            handleTokenStep(event);
            return;
          }

          setSessionMode("replay");
          setLoadingLabel(null);
          setStatusLine(`Live session completed in ${(event.durationMs / 1000).toFixed(1)}s. Replay is ready.`);
          void listRunnerSessions()
            .then(setRunnerSessions)
            .catch(() => undefined);
        },
        onError(message) {
          setLoadingLabel(null);
          setError(message);
        },
        onClose() {
          setLoadingLabel(null);
        },
      });
    } catch (startError) {
      setLoadingLabel(null);
      setSessionMode("sample");
      setError((startError as Error).message);
    }
  }

  function handleTokenStep(event: QwenTokenStepEvent) {
    startTransition(() => {
      setBundle((previous) => {
        const base = previous ? cloneBundle(previous) : null;
        if (!base) return previous;
        return applyLiveTokenStep(base, event);
      });
      setAssistantText(event.completion);
      setStatusLine(`Live token ${event.tokenIndex + 1} flowing through the stage.`);
      setLoadingLabel(null);
      if (liveFollowRef.current) {
        setFrameIndex(event.frame.frame_id);
      }
    });
  }

  async function stopLiveSession() {
    if (!sessionId) return;
    try {
      setStatusLine("Stopping the live session…");
      await cancelRunnerSession(sessionId);
      setRunnerSessions(await listRunnerSessions());
    } catch (stopError) {
      setError((stopError as Error).message);
    }
  }

  async function importTrace(file: File) {
    setLoadingLabel(`Importing ${file.name}…`);
    setError(null);
    try {
      const nextBundle = await loadTraceFromFile(file);
      startTransition(() => {
        setBundle(nextBundle);
        setFrameIndex(0);
        setSelection(null);
        setSessionMode("replay");
        setAssistantText(readLastCompletion(nextBundle));
        setStatusLine(`Imported ${file.name}.`);
      });
    } catch (importError) {
      setError((importError as Error).message);
    } finally {
      setLoadingLabel(null);
    }
  }

  async function exportPng() {
    const canvas = stageRef.current?.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) {
      setError("The stage canvas is unavailable for PNG export.");
      return;
    }
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) {
      setError("The browser could not generate a PNG snapshot.");
      return;
    }
    const url = URL.createObjectURL(blob);
    triggerDownload(url, `qwen-starfield-frame-${String(safeFrameIndex).padStart(3, "0")}.png`);
    URL.revokeObjectURL(url);
  }

  async function exportReplay() {
    try {
      if (traceUrl && sessionMode !== "sample") {
        const archive = await downloadTraceFromRunner(traceUrl);
        triggerDownload(
          URL.createObjectURL(new Blob([archive], { type: "application/octet-stream" })),
          `${sessionId ?? "qwen-session"}.loomtrace`,
        );
        return;
      }
      if (!bundle) {
        setError("There is no replay loaded to export.");
        return;
      }
      const archive = await createLoomTraceArchive(bundle);
      triggerDownload(
        URL.createObjectURL(new Blob([archive], { type: "application/octet-stream" })),
        `${bundle.manifest.model_id}.loomtrace`,
      );
    } catch (exportError) {
      setError((exportError as Error).message);
    }
  }

  async function openRunnerReplay(session: RunnerSession) {
    try {
      setLoadingLabel(`Loading replay ${session.id}…`);
      const archive = await downloadTraceFromRunner(session.traceUrl);
      const file = new File([archive], `${session.id}.loomtrace`, { type: "application/octet-stream" });
      const nextBundle = await loadTraceFromFile(file);
      startTransition(() => {
        setBundle(nextBundle);
        setFrameIndex(0);
        setSelection(null);
        setSessionMode("replay");
        setSessionId(session.id);
        setTraceUrl(session.traceUrl);
        setActivePrompt(session.prompt);
        setAssistantText(session.completion);
        setStatusLine(`Loaded replay ${session.id}.`);
      });
    } catch (openError) {
      setError((openError as Error).message);
    } finally {
      setLoadingLabel(null);
    }
  }

  function jumpToChapter(offset: number) {
    if (!bundle || chapterIndex < 0) return;
    const chapters = bundle.narrative.chapters;
    const nextIndex = clamp(chapterIndex + offset, 0, chapters.length - 1);
    const nextChapter = chapters[nextIndex];
    if (!nextChapter) return;
    setFrameIndex(nextChapter.frameRange[0]);
    setPlaying(false);
    setLiveFollow(false);
    setSelection(nextChapter.defaultSelection ? { kind: "node", id: nextChapter.defaultSelection } : null);
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">NeuroLoom</p>
          <h1>Qwen3.5-0.8B, rendered as a live starfield.</h1>
          <p className="hero-text">
            A single-model stage for Qwen conversations. Live sessions stream into a dense field of residual light, grouped attention,
            DeltaNet memory, and replayable decode traces.
          </p>
        </div>
        <div className="header-badges">
          <span className="status-pill status-pill--accent">{qwenSampleTrace.label}</span>
          <span className="status-pill">{bundle?.timeline.length ?? 0} frames</span>
          <span className={`status-pill ${runnerHealth ? "status-pill--live" : "status-pill--muted"}`}>
            {runnerHealth ? `Runner ${runnerHealth.mode}` : runnerChecked ? "Runner offline" : "Runner probing"}
          </span>
          <span className={`status-pill ${sessionMode === "live" ? "status-pill--live" : ""}`}>{sessionMode}</span>
        </div>
      </header>

      <main className="workspace">
        <aside className="panel panel--left">
          <section className="card">
            <p className="eyebrow">Session Prompt</p>
            <form
              className="prompt-form"
              onSubmit={(event) => {
                event.preventDefault();
                void startLiveSession();
              }}
            >
              <textarea
                value={promptDraft}
                onChange={(event) => setPromptDraft(event.target.value)}
                placeholder="Ask the runner for a live Qwen session…"
                rows={7}
              />
              <div className="prompt-actions">
                <button type="submit" className="primary-button" disabled={!runnerHealth || Boolean(loadingLabel)}>
                  Start Live Session
                </button>
                <button type="button" className="secondary-button" onClick={() => void loadSampleReplay()}>
                  Load Demo Replay
                </button>
                <button type="button" className="secondary-button" onClick={() => void refreshRunnerStatus()}>
                  Refresh Runner
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={async () => setBackendProbe(await probeRunnerBackend())}
                  disabled={!runnerHealth || runnerHealth.mode !== "adapter"}
                >
                  Probe Backend
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => void stopLiveSession()}
                  disabled={!sessionId || sessionMode !== "live"}
                >
                  Stop Live
                </button>
              </div>
            </form>
            <div className="status-list">
              <div>
                <span>Transport</span>
                <strong>
                  {runnerHealth
                    ? `local runner · ${runnerHealth.mode}${runnerHealth.streaming ? " · streaming" : " · buffered"}`
                    : "replay fallback"}
                </strong>
              </div>
              <div>
                <span>Model</span>
                <strong>{runnerHealth?.model ?? qwenSampleTrace.model}</strong>
              </div>
              <div>
                <span>Backend</span>
                <strong>{runnerHealth?.backendModel ?? "fallback replay only"}</strong>
              </div>
              <div>
                <span>Runtime Model</span>
                <strong>{runnerHealth?.effectiveModel ?? qwenSampleTrace.model}</strong>
              </div>
              <div>
                <span>Provider</span>
                <strong>
                  {runnerHealth
                    ? `${runnerHealth.backendLabel}${runnerHealth.backendDetectedFrom === "override" ? " · forced" : ""}`
                    : "synthetic"}
                </strong>
              </div>
              <div>
                <span>Target</span>
                <strong>{runnerHealth?.backendUrl ?? "local synthetic runner"}</strong>
              </div>
              <div>
                <span>Endpoint</span>
                <strong>{runnerHealth?.backendEndpoint ?? "not in use"}</strong>
              </div>
              <div>
                <span>Status</span>
                <strong>{statusLine}</strong>
              </div>
              <div>
                <span>Session</span>
                <strong>{currentRunnerSession?.status ?? sessionMode}</strong>
              </div>
            </div>
            {runnerHealth?.modelRemapped ? (
              <p className="helper-copy">
                NeuroLoom requested the canonical Qwen profile and remapped it to the configured backend model for live inference.
              </p>
            ) : null}
            {runnerHealth?.backendSetupHint ? <p className="helper-copy">{runnerHealth.backendSetupHint}</p> : null}
            {backendProbe ? (
              <div className={backendProbe.ok ? "probe-card" : "probe-card probe-card--error"}>
                <div className="card-heading">
                  <p className="eyebrow">Backend Probe</p>
                  <span>{new Date(backendProbe.checkedAt).toLocaleTimeString()}</span>
                </div>
                <div className="probe-grid">
                  <div>
                    <span>Reachable</span>
                    <strong>{backendProbe.reachable ? "yes" : "no"}</strong>
                  </div>
                  <div>
                    <span>Model Match</span>
                    <strong>{backendProbe.matchedModel ? "yes" : "no"}</strong>
                  </div>
                  <div>
                    <span>Target Model</span>
                    <strong>{backendProbe.targetModel}</strong>
                  </div>
                  <div>
                    <span>Models Endpoint</span>
                    <strong>{backendProbe.modelsEndpoint ?? "not required"}</strong>
                  </div>
                </div>
                <p className="helper-copy">{backendProbe.error ?? backendProbe.hint}</p>
                <div className="probe-models">
                  {backendProbe.models.length === 0 ? <span className="empty-state">No model list reported.</span> : null}
                  {backendProbe.models.slice(0, 6).map((modelId) => (
                    <span key={modelId} className="token-pill">
                      {modelId}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
            {error ? <p className="error-text">{error}</p> : null}
            {loadingLabel ? <p className="loading-text">{loadingLabel}</p> : null}
          </section>

          <section className="card">
            <p className="eyebrow">Conversation</p>
            <div className="message message--user">
              <span className="message-role">User</span>
              <p>{activePrompt}</p>
            </div>
            <div className="message message--assistant">
              <span className="message-role">Qwen</span>
              <p>{assistantText || "No tokens yet. Start a live session or load the demo replay."}</p>
            </div>
          </section>

          <section className="card">
            <div className="card-heading">
              <p className="eyebrow">Token Window</p>
              <span>{currentTokenWindow.length} tokens</span>
            </div>
            <div className="token-cloud">
              {currentTokenWindow.length === 0 ? <span className="empty-state">Waiting for token flow.</span> : null}
              {currentTokenWindow.map((token, index) => {
                const absoluteIndex = (currentPayload?.tokenIndex ?? 0) - currentTokenWindow.length + index + 1;
                const tokenId = `token-${absoluteIndex}`;
                const isActive = selection?.kind === "token" && selection.id === tokenId;
                return (
                  <button
                    key={tokenId}
                    type="button"
                    className={isActive ? "token-pill token-pill--active" : "token-pill"}
                    onClick={() => setSelection({ kind: "token", id: tokenId })}
                  >
                    {token}
                  </button>
                );
              })}
            </div>
          </section>

          <section className="card">
            <div className="card-heading">
              <p className="eyebrow">Recent Sessions</p>
              <span>{runnerSessions.length}</span>
            </div>
            <div className="session-list">
              {runnerSessions.length === 0 ? <span className="empty-state">No runner sessions yet.</span> : null}
              {runnerSessions.map((session) => (
                <article key={session.id} className={session.id === sessionId ? "session-row session-row--active" : "session-row"}>
                  <div className="session-row__meta">
                    <strong>{session.id}</strong>
                    <span>{session.status}</span>
                  </div>
                  <p>{session.prompt}</p>
                  <div className="session-row__actions">
                    <span>{session.tokenCount} tokens</span>
                    {session.archiveReady ? (
                      <button type="button" className="secondary-button" onClick={() => void openRunnerReplay(session)}>
                        Open Replay
                      </button>
                    ) : null}
                    {(session.status === "live" || session.status === "booting") && session.id === sessionId ? (
                      <button type="button" className="secondary-button" onClick={() => void stopLiveSession()}>
                        Cancel
                      </button>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
          </section>
        </aside>

        <section className="center-column">
          <section className="stage-card" ref={stageRef}>
            <div className="stage-meta">
              <div>
                <p className="eyebrow">Frame</p>
                <strong>
                  {bundle?.timeline.length ? safeFrameIndex + 1 : 0} / {bundle?.timeline.length ?? 0}
                </strong>
              </div>
              <div>
                <p className="eyebrow">Current Token</p>
                <strong>{currentPayload?.token?.trim() || "idle"}</strong>
              </div>
              <div>
                <p className="eyebrow">Chapter</p>
                <strong>{currentChapter?.label ?? "Awaiting Tokens"}</strong>
              </div>
              <div>
                <p className="eyebrow">Focus</p>
                <strong>{selection ? `${selection.kind} · ${selection.id}` : "none"}</strong>
              </div>
            </div>

            {bundle ? (
              <SceneCanvas
                bundle={bundle}
                frame={frame}
                payload={currentPayload}
                selection={selection}
                onSelect={setSelection}
                live={sessionMode === "live"}
              />
            ) : (
              <div className="stage-placeholder">Preparing the starfield…</div>
            )}
          </section>

          <section className="scrubber-card">
            <div className="scrubber-actions">
              <button
                type="button"
                className="secondary-button"
                onClick={() => stepFrame(-1)}
                disabled={!bundle || bundle.timeline.length === 0}
              >
                Prev
              </button>
              <button
                type="button"
                className="primary-button"
                onClick={() => setPlaying((current) => !current)}
                disabled={!bundle || bundle.timeline.length === 0}
              >
                {playing ? "Pause" : "Play"}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => stepFrame(1)}
                disabled={!bundle || bundle.timeline.length === 0}
              >
                Next
              </button>
              <button
                type="button"
                className={liveFollow ? "secondary-button is-active" : "secondary-button"}
                onClick={() => setLiveFollow((current) => !current)}
              >
                Follow Live
              </button>
              <button type="button" className="secondary-button" onClick={() => void exportPng()}>
                Export PNG
              </button>
              <button type="button" className="secondary-button" onClick={() => void exportReplay()}>
                Export `.loomtrace`
              </button>
              <label htmlFor={uploadId} className="secondary-button secondary-button--file">
                Import Replay
              </label>
              <input
                id={uploadId}
                type="file"
                accept=".loomtrace"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void importTrace(file);
                  }
                }}
              />
            </div>

            <input
              className="scrubber-range"
              type="range"
              min={0}
              max={Math.max((bundle?.timeline.length ?? 1) - 1, 0)}
              value={safeFrameIndex}
              onChange={(event) => {
                setFrameIndex(Number(event.target.value));
                setPlaying(false);
                setLiveFollow(false);
              }}
              disabled={!bundle || bundle.timeline.length === 0}
            />

            <div className="scrubber-footer">
              <div className="range-caption">
                <span>Space play · ←/→ step · S export</span>
                <strong>{frame?.phase ?? "idle"}</strong>
              </div>
              <div className="chapter-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => jumpToChapter(-1)}
                  disabled={!bundle || chapterIndex <= 0}
                >
                  Prev Chapter
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => jumpToChapter(1)}
                  disabled={!bundle || chapterIndex < 0 || chapterIndex >= bundle.narrative.chapters.length - 1}
                >
                  Next Chapter
                </button>
              </div>
            </div>
          </section>
        </section>

        <aside className="panel panel--right">
          <section className="card">
            <div className="card-heading">
              <p className="eyebrow">Current Frame</p>
              <span>{currentTokenIndex !== null ? `token ${currentTokenIndex + 1}` : "idle"}</span>
            </div>
            <div className="metric-grid">
              {currentMetrics.length === 0 ? <span className="empty-state">No metrics yet.</span> : null}
              {currentMetrics.map((metric) => (
                <div key={metric.id} className="metric-card">
                  <span>{metric.label}</span>
                  <strong>{metric.value.toFixed(3)}</strong>
                </div>
              ))}
            </div>
            <div className="logit-list">
              <div className="card-heading">
                <p className="eyebrow">Top Logits</p>
                <span>{currentTopLogits.length} candidates</span>
              </div>
              {currentTopLogits.length === 0 ? <span className="empty-state">Waiting for decode logits.</span> : null}
              {currentTopLogits.map((logit) => (
                <div key={`${logit.token}-${logit.score}`} className="logit-row">
                  <span>{logit.token}</span>
                  <div className="logit-bar">
                    <div style={{ width: `${Math.max(8, logit.score * 100)}%` }} />
                  </div>
                  <strong>{logit.score.toFixed(3)}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="card">
            <div className="card-heading">
              <p className="eyebrow">Focus Detail</p>
              <span>{selection ? selection.kind : "none"}</span>
            </div>
            <strong className="selection-title">{selectionDetail.title}</strong>
            <p className="selection-copy">{selectionDetail.description}</p>
            {selectionDetail.metrics.length > 0 ? (
              <div className="mini-metrics">
                {selectionDetail.metrics.map((metric) => (
                  <div key={metric.label} className="mini-metric">
                    <span>{metric.label}</span>
                    <strong>{metric.value}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <span className="empty-state">Select a token, structural block, or sample cluster.</span>
            )}
          </section>

          <section className="card">
            <div className="card-heading">
              <p className="eyebrow">Focus Blocks</p>
              <span>around block {focusBlock + 1}</span>
            </div>
            {focusDigest.length === 0 ? <span className="empty-state">Layer summaries appear after the first token.</span> : null}
            {focusDigest.map((digest) => (
              <div key={digest.block} className="digest-row">
                <strong>Block {digest.block + 1}</strong>
                <div className="digest-bars">
                  <DigestBar label="res" value={digest.residual} />
                  <DigestBar label="attn" value={digest.attention} />
                  <DigestBar label="delta" value={digest.delta} />
                  <DigestBar label="ffn" value={digest.ffn} />
                </div>
              </div>
            ))}
          </section>
        </aside>
      </main>
    </div>
  );
}

function DigestBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="digest-bar">
      <span>{label}</span>
      <div className="digest-bar__track">
        <div style={{ width: `${Math.max(8, value * 100)}%` }} />
      </div>
      <strong>{value.toFixed(2)}</strong>
    </div>
  );
}

function getInspectPayload(bundle: TraceBundle, frame: TraceFrame) {
  const payloadId = frame.payload_refs.find((ref) =>
    bundle.manifest.payload_catalog.find((entry) => entry.id === ref && entry.kind === "inspect"),
  );
  if (!payloadId) return null;
  const raw = bundle.payloads.get(payloadId);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as QwenFramePayload;
  } catch {
    return null;
  }
}

function readLastCompletion(bundle: TraceBundle) {
  const lastFrame = bundle.timeline.at(-1);
  if (!lastFrame) return "";
  return getInspectPayload(bundle, lastFrame)?.completion ?? "";
}

function describeSelection(input: {
  bundle: TraceBundle | null;
  frame: TraceFrame | null;
  payload: QwenFramePayload | null;
  selection: SelectionState;
}) {
  if (!input.selection) {
    return {
      title: "No focus",
      description: "Click a token, structural block, or star cluster to lock the stage around it.",
      metrics: [],
    };
  }

  if (input.selection.kind === "token") {
    const index = Number(input.selection.id.replace("token-", ""));
    const payload = input.payload;
    const localIndex = payload ? payload.tokenWindow.length - (payload.tokenIndex - index + 1) : -1;
    const token = payload && localIndex >= 0 ? payload.tokenWindow[localIndex] : input.selection.id;
    const attention = payload && localIndex >= 0 ? (payload.attentionRow[localIndex] ?? 0) : 0;
    return {
      title: `Token ${index + 1}`,
      description: "A token focus brightens the local rail and pulls attention weights toward its recent neighborhood.",
      metrics: [
        { label: "token", value: token.trim() || "space" },
        { label: "attention", value: attention.toFixed(3) },
      ],
    };
  }

  if (input.selection.kind === "cluster") {
    const unit = input.payload?.sampledUnits.find((entry) => entry.id === input.selection.id);
    return {
      title: unit?.label ?? input.selection.id,
      description:
        "Sample clusters are the visible star grains inside each hybrid sub-block. They move with the same token pulse as their parent lane.",
      metrics: unit
        ? [
            { label: "lane", value: unit.lane },
            { label: "intensity", value: unit.intensity.toFixed(3) },
            { label: "affinity", value: unit.tokenAffinity.toFixed(3) },
          ]
        : [],
    };
  }

  const node = input.bundle?.graph.nodes.find((entry) => entry.id === input.selection.id);
  const nodeState = input.frame?.node_states.find((entry) => entry.nodeId === input.selection.id);
  return {
    title: node?.label ?? input.selection.id,
    description:
      "Structural nodes are the stable anchors of the live stage. Their star clusters show block-scale energy rather than an abstract module box.",
    metrics: node
      ? [
          { label: "type", value: node.type },
          { label: "lane", value: String(node.metadata.lane ?? "n/a") },
          { label: "activation", value: (nodeState?.activation ?? 0).toFixed(3) },
        ]
      : [],
  };
}

function triggerDownload(url: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

function cloneBundle(bundle: TraceBundle): TraceBundle {
  return {
    manifest: JSON.parse(JSON.stringify(bundle.manifest)) as TraceBundle["manifest"],
    graph: JSON.parse(JSON.stringify(bundle.graph)) as TraceBundle["graph"],
    narrative: JSON.parse(JSON.stringify(bundle.narrative)) as TraceBundle["narrative"],
    timeline: JSON.parse(JSON.stringify(bundle.timeline)) as TraceBundle["timeline"],
    payloads: new Map(bundle.payloads),
    preview: bundle.preview ? new Uint8Array(bundle.preview) : undefined,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
