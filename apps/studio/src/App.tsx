import type { TraceBundle, TraceFrame } from "@neuroloom/core";
import { startTransition, useDeferredValue, useEffect, useId, useRef, useState } from "react";
import { scaleLinear } from "d3-scale";

import { SceneCanvas } from "./SceneCanvas";
import { officialTraces } from "./sampleTraces";
import { type SelectionState, useStudioStore } from "./state";
import { loadTraceFromFile, loadTraceFromUrl } from "./traceLoader";

type BrowserRuntimeState = {
  label: string;
  detail: string;
  webgl: boolean;
  webgpu: boolean;
};

export function App() {
  const {
    mode,
    traceId,
    bundle,
    engine,
    loadingLabel,
    error,
    frameIndex,
    playing,
    selection,
    frozenSelection,
    activeChapterId,
    setMode,
    beginLoading,
    finishLoading,
    failLoading,
    setFrameIndex,
    step,
    togglePlaying,
    setPlaying,
    setSelection,
    toggleFreezeSelection,
    clearFrozenSelection,
    jumpToChapter
  } = useStudioStore();
  const uploadId = useId();
  const stageFrameRef = useRef<HTMLDivElement | null>(null);
  const [browserRuntime, setBrowserRuntime] = useState<BrowserRuntimeState | null>(null);

  async function ingestTrace(nextTraceId: string, loader: () => Promise<TraceBundle>) {
    beginLoading(nextTraceId);
    try {
      const nextBundle = await loader();
      startTransition(() => {
        finishLoading(nextTraceId, nextBundle);
      });
    } catch (loadError) {
      failLoading((loadError as Error).message);
    }
  }

  async function regenerateOfficialTrace(targetTraceId: string) {
    beginLoading(`${targetTraceId} (browser)`);
    try {
      const { createOfficialTraceBundle, isOfficialTraceId } = await import("@neuroloom/official-traces");
      if (!isOfficialTraceId(targetTraceId)) {
        throw new Error(`Browser regeneration is only available for official traces. Received "${targetTraceId}".`);
      }
      const nextBundle = createOfficialTraceBundle(targetTraceId);
      startTransition(() => {
        finishLoading(targetTraceId, nextBundle);
      });
    } catch (loadError) {
      failLoading((loadError as Error).message);
    }
  }

  useEffect(() => {
    if (bundle || loadingLabel) return;
    const firstTrace = officialTraces[0];
    void ingestTrace(firstTrace.id, () => loadTraceFromUrl(firstTrace.path));
  }, [bundle, loadingLabel]);

  useEffect(() => {
    if (!playing || !engine) return;
    const intervalId = window.setInterval(() => {
      const state = useStudioStore.getState();
      if (!state.engine) return;
      if (state.frameIndex >= state.engine.frameCount - 1) {
        state.setPlaying(false);
        return;
      }
      state.step(1);
    }, 520);

    return () => window.clearInterval(intervalId);
  }, [playing, engine]);

  useEffect(() => {
    let cancelled = false;

    void detectBrowserRuntime().then((runtime) => {
      if (!cancelled) {
        setBrowserRuntime(runtime);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName)) {
        return;
      }
      if (!useStudioStore.getState().engine) return;
      if (event.key === " ") {
        event.preventDefault();
        useStudioStore.getState().togglePlaying();
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        useStudioStore.getState().step(-1);
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        useStudioStore.getState().step(1);
        return;
      }
      if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        void exportStageSnapshot();
        return;
      }
      if (event.key.toLowerCase() === "f") {
        event.preventDefault();
        useStudioStore.getState().toggleFreezeSelection();
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        const state = useStudioStore.getState();
        if (state.frozenSelection) {
          state.clearFrozenSelection();
          return;
        }
        state.setSelection(null);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const frame = engine ? engine.getFrame(frameIndex) : null;
  const deferredFrame = useDeferredValue(frame);
  const currentChapter =
    bundle?.narrative.chapters.find((chapter) => chapter.id === activeChapterId) ??
    (engine ? engine.getChapterForFrame(frameIndex) ?? null : null);
  const activeTrace = officialTraces.find((trace) => trace.id === traceId) ?? officialTraces.find((trace) => trace.family === bundle?.manifest.family);
  const currentChapterIndex = currentChapter && bundle ? bundle.narrative.chapters.findIndex((chapter) => chapter.id === currentChapter.id) : -1;
  const renderPayloadId =
    bundle && deferredFrame
      ? bundle.manifest.payload_catalog.find((entry) => entry.kind === "render" && deferredFrame.payload_refs.includes(entry.id))?.id ?? null
      : null;
  const renderPayload = bundle && renderPayloadId ? parsePayload(bundle.payloads.get(renderPayloadId)) : null;
  const sceneSelection = frozenSelection ?? selection;
  const regenerationTarget = bundle?.manifest.model_id ?? traceId ?? null;
  const canRegenerateInBrowser = regenerationTarget ? officialTraces.some((trace) => trace.id === regenerationTarget) : false;
  const frozenSelectionLabel = bundle && frozenSelection ? formatSelectionLabel(bundle, frozenSelection) : null;

  async function exportStageSnapshot() {
    const state = useStudioStore.getState();
    const traceId = state.traceId;
    const engine = state.engine;
    const frameIndex = state.frameIndex;
    if (!traceId || !engine) return;
    const currentFrame = engine.getFrame(frameIndex);
    if (!currentFrame) return;
    
    const canvas = stageFrameRef.current?.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) {
      console.warn("Snapshot export failed: stage canvas is unavailable.");
      return;
    }
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) {
      console.warn("Snapshot export failed: browser could not create a PNG.");
      return;
    }
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${traceId}-frame-${String(currentFrame.frame_id).padStart(3, "0")}.png`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">NeuroLoom v1</p>
          <h1>Neural networks, replayed as a precise 2.5D stage.</h1>
          <p className="hero-text">
            NeuroLoom is a replay-first explainer for <code>MLP</code>, <code>CNN</code>, and standard
            <code> GPT-style Transformer</code> traces. It reconstructs one training or inference run into a
            controllable visual scene where glow, motion, and numeric truth stay in sync.
          </p>
          <div className="hero-definition">
            <span>Definition</span>
            <strong>NeuroLoom is a neural network replay explainer.</strong>
          </div>
        </div>
        <div className="hero-stats">
          <StatCard label="Families" value="3 official" detail="MLP / CNN / Transformer" />
          <StatCard label="Modes" value="Story + Studio" detail="Guided narrative and frame-by-frame analysis" />
          <StatCard label="Input" value=".loomtrace" detail="Controlled replay bundle with schema validation" />
        </div>
      </header>

      <section className="trace-library">
        {officialTraces.map((trace) => (
          <button
            key={trace.id}
            type="button"
            className={`trace-card trace-card--${trace.accent} ${traceId === trace.id ? "is-active" : ""}`}
            onClick={() => void ingestTrace(trace.id, () => loadTraceFromUrl(trace.path))}
          >
            <span className="trace-card__family">{trace.family}</span>
            <strong>{trace.label}</strong>
            <p>{trace.summary}</p>
          </button>
        ))}
      </section>

      <section className="toolbar">
        <div className="toolbar__group">
          <button type="button" className={mode === "story" ? "chip is-active" : "chip"} onClick={() => setMode("story")}>
            Story Mode
          </button>
          <button type="button" className={mode === "studio" ? "chip is-active" : "chip"} onClick={() => setMode("studio")}>
            Studio Mode
          </button>
        </div>
        <div className="toolbar__group toolbar__group--meta">
          {bundle ? (
            <>
              <span className="meta-pill">{bundle.manifest.title}</span>
              <span className="meta-pill">{bundle.manifest.family}</span>
              <span className="meta-pill">{bundle.manifest.frame_count} frames</span>
              {frozenSelectionLabel ? <span className="meta-pill meta-pill--focus">Focus locked: {frozenSelectionLabel}</span> : null}
            </>
          ) : null}
          {browserRuntime ? (
            <span className="meta-pill meta-pill--runtime" title={browserRuntime.detail}>
              {browserRuntime.label}
            </span>
          ) : null}
        </div>
        <div className="toolbar__group">
          <button type="button" className={frozenSelection ? "chip is-active" : "chip"} onClick={() => toggleFreezeSelection()} disabled={!selection && !frozenSelection}>
            {frozenSelection ? "Unfreeze Focus" : "Freeze Focus"}
          </button>
          <button type="button" className="chip chip--ghost" onClick={() => clearFrozenSelection()} disabled={!frozenSelection}>
            Clear Focus
          </button>
          <button
            type="button"
            className="chip"
            onClick={() => {
              if (!regenerationTarget || !canRegenerateInBrowser) return;
              void regenerateOfficialTrace(regenerationTarget);
            }}
            disabled={!canRegenerateInBrowser}
          >
            Rebuild In Browser
          </button>
          <button type="button" className="chip" onClick={() => void exportStageSnapshot()} disabled={!bundle}>
            Export PNG
          </button>
          <label htmlFor={uploadId} className="chip chip--file">
            Import `.loomtrace`
          </label>
          <input
            id={uploadId}
            className="visually-hidden"
            type="file"
            accept=".loomtrace"
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              if (!file) return;
              void ingestTrace(file.name, () => loadTraceFromFile(file));
              event.currentTarget.value = "";
            }}
          />
        </div>
      </section>

      {loadingLabel ? <div className="banner banner--info">Loading {loadingLabel}…</div> : null}
      {error ? <div className="banner banner--error">{error}</div> : null}

      {bundle && deferredFrame ? (
        <main className="workspace">
          <aside className="panel panel--left">
            <section className="panel-section">
              <header className="panel-section__header">
                <span>Trace</span>
                <strong>{bundle.manifest.summary}</strong>
              </header>
              <p className="muted-copy">{bundle.narrative.intro}</p>
            </section>

            {mode === "story" ? (
              <>
                <section className="panel-section">
                  <header className="panel-section__header">
                    <span>Narrative Track</span>
                    <strong>{bundle.narrative.chapters.length} chapters</strong>
                  </header>
                  <div className="stack-list">
                    {bundle.narrative.chapters.map((chapter, index) => (
                      <button
                        key={chapter.id}
                        type="button"
                        className={chapter.id === currentChapter?.id ? "stack-item is-active stack-item--story" : "stack-item stack-item--story"}
                        onClick={() => jumpToChapter(chapter.id)}
                      >
                        <div>
                          <span>{chapter.label}</span>
                          <small>{chapter.description}</small>
                        </div>
                        <small>
                          {index + 1}/{bundle.narrative.chapters.length}
                        </small>
                      </button>
                    ))}
                  </div>
                </section>

                {activeTrace ? (
                  <section className="panel-section">
                    <header className="panel-section__header">
                      <span>Watch For</span>
                      <strong>{activeTrace.family}</strong>
                    </header>
                    <p className="story-title">{activeTrace.storyTitle}</p>
                    <KeyList items={activeTrace.watchFor} />
                  </section>
                ) : null}
              </>
            ) : (
              <>
                <section className="panel-section">
                  <header className="panel-section__header">
                    <span>Story Anchors</span>
                    <strong>{bundle.narrative.chapters.length} stops</strong>
                  </header>
                  <div className="stack-list">
                    {bundle.narrative.chapters.map((chapter) => (
                      <button
                        key={chapter.id}
                        type="button"
                        className={chapter.id === currentChapter?.id ? "stack-item is-active" : "stack-item"}
                        onClick={() => jumpToChapter(chapter.id)}
                      >
                        <span>{chapter.label}</span>
                        <small>
                          {chapter.frameRange[0]}–{chapter.frameRange[1]}
                        </small>
                      </button>
                    ))}
                  </div>
                </section>

                <section className="panel-section">
                  <header className="panel-section__header">
                    <span>Structure</span>
                    <strong>{bundle.graph.nodes.length} nodes</strong>
                  </header>
                  <StructureList bundle={bundle} selection={selection} onSelect={setSelection} />
                </section>

                {activeTrace ? (
                  <section className="panel-section">
                    <header className="panel-section__header">
                      <span>Studio Tips</span>
                      <strong>3 prompts</strong>
                    </header>
                    <KeyList items={activeTrace.studioTips} />
                  </section>
                ) : null}
              </>
            )}
          </aside>

          <section className="stage-column">
            <div className="stage-frame" ref={stageFrameRef}>
              <div className="stage-frame__overlay">
                <div>
                  <span className="overlay-label">Phase</span>
                  <strong>{deferredFrame.phase}</strong>
                </div>
                <div>
                  <span className="overlay-label">Step</span>
                  <strong>
                    {deferredFrame.step}.{deferredFrame.substep}
                  </strong>
                </div>
                {currentChapter ? (
                  <div>
                    <span className="overlay-label">Chapter</span>
                    <strong>{currentChapter.label}</strong>
                  </div>
                ) : null}
                {frozenSelectionLabel ? (
                  <div>
                    <span className="overlay-label">Focus</span>
                    <strong>{frozenSelectionLabel}</strong>
                  </div>
                ) : null}
              </div>
              <div className="stage-frame__lens">
                <RenderLens payload={renderPayload} family={bundle.manifest.family} mode={mode} />
              </div>
              <div className="stage-frame__legend">
                <LegendPill colorClass="is-electric" label="Activation / forward flow" />
                <LegendPill colorClass="is-amber" label="Compression / backward pressure" />
                <LegendPill colorClass="is-lime" label="Selection / frozen focus" />
              </div>
              <SceneCanvas bundle={bundle} frame={deferredFrame} selection={sceneSelection} onSelect={setSelection} />
            </div>
            <TimelineBar
              frame={deferredFrame}
              frameIndex={frameIndex}
              frameCount={bundle.manifest.frame_count}
              playing={playing}
              chapter={currentChapter?.label ?? null}
              onSeek={setFrameIndex}
              onPrev={() => step(-1)}
              onNext={() => step(1)}
              onTogglePlay={togglePlaying}
              onExport={() => void exportStageSnapshot()}
              onPrevChapter={() => {
                if (!bundle || currentChapterIndex <= 0) return;
                jumpToChapter(bundle.narrative.chapters[currentChapterIndex - 1]!.id);
              }}
              onNextChapter={() => {
                if (!bundle || currentChapterIndex < 0 || currentChapterIndex >= bundle.narrative.chapters.length - 1) return;
                jumpToChapter(bundle.narrative.chapters[currentChapterIndex + 1]!.id);
              }}
            />
          </section>

          <aside className="panel panel--right">
            {mode === "story" ? (
              <StoryPanel
                bundle={bundle}
                frame={deferredFrame}
                chapter={currentChapter}
                activeTraceTitle={activeTrace?.storyTitle ?? null}
                watchFor={activeTrace?.watchFor ?? []}
              />
            ) : (
              <InspectorPanel
                bundle={bundle}
                frame={deferredFrame}
                selection={selection}
                chapter={currentChapter?.description ?? null}
                activeTrace={activeTrace ?? null}
                onSelect={setSelection}
              />
            )}
          </aside>
        </main>
      ) : null}
    </div>
  );
}

function StatCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <article className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </article>
  );
}

function TimelineBar({
  frame,
  frameIndex,
  frameCount,
  playing,
  chapter,
  onSeek,
  onPrev,
  onNext,
  onTogglePlay,
  onExport,
  onPrevChapter,
  onNextChapter
}: {
  frame: TraceFrame;
  frameIndex: number;
  frameCount: number;
  playing: boolean;
  chapter: string | null;
  onSeek(index: number): void;
  onPrev(): void;
  onNext(): void;
  onTogglePlay(): void;
  onExport(): void;
  onPrevChapter(): void;
  onNextChapter(): void;
}) {
  return (
    <div className="timeline">
      <div className="timeline__controls">
        <button type="button" className="chip" onClick={onPrev}>
          Prev
        </button>
        <button type="button" className="chip chip--play" onClick={onTogglePlay}>
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" className="chip" onClick={onNext}>
          Next
        </button>
        <button type="button" className="chip" onClick={onExport}>
          PNG
        </button>
      </div>
      <div className="timeline__track">
        <input
          type="range"
          min={0}
          max={frameCount - 1}
          value={frameIndex}
          onChange={(event) => onSeek(Number(event.currentTarget.value))}
        />
        <div className="timeline__meta">
          <span>
            Frame {frame.frame_id + 1} / {frameCount}
          </span>
          <span>{chapter ?? "Free scrub"}</span>
          <span className="timeline__hotkeys">`Space` play · `←/→` step · `S` export · `F` freeze · `Esc` clear</span>
          <div className="timeline__chapter-nav">
            <button type="button" className="chip chip--ghost" onClick={onPrevChapter}>
              Prev Chapter
            </button>
            <button type="button" className="chip chip--ghost" onClick={onNextChapter}>
              Next Chapter
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function StructureList({
  bundle,
  selection,
  onSelect
}: {
  bundle: TraceBundle;
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
}) {
  const layerIndexes = Array.from(new Set(bundle.graph.nodes.map((node) => node.layerIndex))).sort((left, right) => left - right);

  return (
    <div className="layer-groups">
      {layerIndexes.map((layerIndex) => (
        <div key={layerIndex} className="layer-group">
          <span className="layer-group__label">Layer {layerIndex}</span>
          {bundle.graph.nodes
            .filter((node) => node.layerIndex === layerIndex)
            .sort((left, right) => left.order - right.order)
            .map((node) => (
              <button
                key={node.id}
                type="button"
                className={selection?.id === node.id ? "structure-pill is-active" : "structure-pill"}
                onClick={() => onSelect({ id: node.id, kind: "node" })}
              >
                <span>{node.label}</span>
                <small>{node.type}</small>
              </button>
            ))}
        </div>
      ))}
    </div>
  );
}

function InspectorPanel({
  bundle,
  frame,
  selection,
  chapter,
  activeTrace,
  onSelect
}: {
  bundle: TraceBundle;
  frame: TraceFrame;
  selection: SelectionState;
  chapter: string | null;
  activeTrace: (typeof officialTraces)[number] | null;
  onSelect(selection: SelectionState): void;
}) {
  const inspectPayloadId =
    bundle.manifest.payload_catalog.find((entry) => entry.kind === "inspect" && frame.payload_refs.includes(entry.id))?.id ?? null;
  const inspectPayload = inspectPayloadId ? parsePayload(bundle.payloads.get(inspectPayloadId)) : null;
  const selectedDetail =
    selection?.kind === "node" && inspectPayload && typeof inspectPayload === "object" && "selectionDetails" in inspectPayload
      ? (inspectPayload.selectionDetails as any)?.[selection.id]
      : null;

  return (
    <>
      <FamilyFocusPanel bundle={bundle} selection={selection} onSelect={onSelect} payload={inspectPayload} activeTrace={activeTrace} />

      <section className="panel-section">
        <header className="panel-section__header">
          <span>Narrative Notes</span>
          <strong>{chapter ? "Current chapter" : "Frame note"}</strong>
        </header>
        <p className="muted-copy">{chapter ?? frame.note ?? "No note for this frame."}</p>
      </section>

      <section className="panel-section">
        <header className="panel-section__header">
          <span>Frame Metrics</span>
          <strong>{frame.metric_refs.length} values</strong>
        </header>
        <div className="metric-grid">
          {frame.metric_refs.map((metric) => (
            <article key={metric.id} className="metric-card">
              <span>{metric.label}</span>
              <strong>{formatMetric(metric.value)}</strong>
            </article>
          ))}
        </div>
      </section>

      <section className="panel-section">
        <header className="panel-section__header">
          <span>Tensor Slice</span>
          <strong>{(inspectPayload?.headline as string) ?? "No payload"}</strong>
        </header>
        {inspectPayload ? <PayloadView payload={inspectPayload} /> : <p className="muted-copy">No inspect payload.</p>}
      </section>

      {bundle.manifest.family === "mlp" ? <MlpBoundaryPanel payload={inspectPayload} /> : null}
      {bundle.manifest.family === "cnn" ? <CnnFeaturePanel payload={inspectPayload} onSelect={onSelect} /> : null}

      {bundle.manifest.family === "transformer" ? (
        <TransformerAttentionPanel bundle={bundle} payload={inspectPayload} selection={selection} onSelect={onSelect} />
      ) : null}

      <section className="panel-section">
        <header className="panel-section__header">
          <span>Structure</span>
          <strong>{selection ? selection.id : "Nothing selected"}</strong>
        </header>
        {selectedDetail ? (
          <div className="detail-card">
            <strong>{selectedDetail.title}</strong>
            <p>{selectedDetail.blurb}</p>
            <div className="detail-stats">
              {selectedDetail.stats.map((stat: { label: string; value: number }) => (
                <div key={stat.label} className="detail-stat">
                  <span>{stat.label}</span>
                  <strong>{formatMetric(stat.value)}</strong>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="muted-copy">Select a node to inspect family-specific details.</p>
        )}
      </section>
    </>
  );
}

function TransformerAttentionPanel({
  bundle,
  payload,
  selection,
  onSelect
}: {
  bundle: TraceBundle;
  payload: Record<string, unknown> | null;
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
}) {
  const heads = Array.isArray(payload?.heads)
    ? payload.heads.flatMap((head) => {
        if (!head || typeof head !== "object") return [];
        const id = "id" in head && typeof head.id === "string" ? head.id : "head";
        const label = "label" in head && typeof head.label === "string" ? head.label : id;
        const matrix = "matrix" in head && Array.isArray(head.matrix) ? (head.matrix as number[][]) : [];
        const focusTokenIndex =
          "focusTokenIndex" in head && typeof head.focusTokenIndex === "number" ? head.focusTokenIndex : 0;
        const score = "score" in head && typeof head.score === "number" ? head.score : null;
        return [{ id, label, matrix, focusTokenIndex, score }];
      })
    : [];
  const tokens = Array.isArray(payload?.tokens) ? payload.tokens.filter((entry): entry is string => typeof entry === "string") : [];
  const topTokens = Array.isArray(payload?.topTokens)
    ? payload.topTokens.flatMap((entry) => {
        if (!entry || typeof entry !== "object") return [];
        const token = "token" in entry && typeof entry.token === "string" ? entry.token : null;
        const probability = "probability" in entry && typeof entry.probability === "number" ? entry.probability : null;
        return token && probability !== null ? [{ token, probability }] : [];
      })
    : [];
  const [selectedHeadIndex, setSelectedHeadIndex] = useState(0);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0);

  useEffect(() => {
    setSelectedHeadIndex(0);
  }, [bundle.manifest.model_id, payload?.headline]);

  useEffect(() => {
    if (heads.length === 0) return;
    const suggestedToken = heads[selectedHeadIndex]?.focusTokenIndex ?? 0;
    setSelectedTokenIndex(Math.max(0, Math.min(suggestedToken, Math.max(tokens.length - 1, 0))));
  }, [selectedHeadIndex, heads.length, payload?.headline, tokens.length]);

  const selectedHead = heads[Math.min(selectedHeadIndex, Math.max(heads.length - 1, 0))] ?? null;
  const selectedToken = tokens[Math.min(selectedTokenIndex, Math.max(tokens.length - 1, 0))] ?? null;
  const tokenNodes = bundle.graph.nodes.filter((node) => node.type === "token").sort((left, right) => left.order - right.order);
  const selectedRow =
    selectedHead && selectedHead.matrix[selectedTokenIndex]
      ? selectedHead.matrix[selectedTokenIndex]!.map((value, index) => ({
          token: tokens[index] ?? `token-${index}`,
          value
        }))
      : [];

  return (
    <section className="panel-section">
      <header className="panel-section__header">
        <span>Attention Explorer</span>
        <strong>{selectedHead?.label ?? "No head"}</strong>
      </header>
      <div className="focus-groups">
        <div className="focus-group">
          <span className="focus-group__label">Heads</span>
          <div className="focus-group__chips">
            {heads.map((head, index) => (
              <button
                key={head.id}
                type="button"
                data-testid={`attention-head-${head.id}`}
                className={selectedHeadIndex === index ? "focus-chip is-active" : "focus-chip"}
                onClick={() => setSelectedHeadIndex(index)}
              >
                {head.label}
              </button>
            ))}
          </div>
        </div>
        <div className="focus-group">
          <span className="focus-group__label">Tokens</span>
          <div className="focus-group__chips">
            {tokenNodes.map((node, index) => (
              <button
                key={node.id}
                type="button"
                data-testid={`attention-token-${node.id}`}
                className={selectedTokenIndex === index || selection?.id === node.id ? "focus-chip is-active" : "focus-chip"}
                onClick={() => {
                  setSelectedTokenIndex(index);
                  onSelect({ id: node.id, kind: "node" });
                }}
              >
                {node.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      {selectedHead ? (
        <div className="family-slice">
          <div className="family-slice__meta">
            <span>{selectedToken ? `${selectedHead.label} on ${selectedToken}` : selectedHead.label}</span>
            <strong>{selectedHead.score !== null ? formatMetric(selectedHead.score) : "focused row"}</strong>
          </div>
          <MatrixHeatmap matrix={selectedHead.matrix} />
        </div>
      ) : null}
      {selectedRow.length > 0 ? (
        <div className="token-bars">
          {selectedRow.map((entry) => (
            <div key={entry.token} className="token-bars__item">
              <div className="token-bars__meta">
                <span>{entry.token}</span>
                <strong>{formatMetric(entry.value)}</strong>
              </div>
              <div className="series-bar__track">
                <div className="series-bar__fill" style={{ width: `${Math.max(6, entry.value * 100)}%` }} />
              </div>
            </div>
          ))}
        </div>
      ) : null}
      {topTokens.length > 0 ? (
        <div className="detail-card">
          <strong>Decode candidates</strong>
          <div className="detail-stats">
            {topTokens.map((entry) => (
              <div key={entry.token} className="detail-stat">
                <span>{entry.token}</span>
                <strong>{formatMetric(entry.probability)}</strong>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function MlpBoundaryPanel({ payload }: { payload: Record<string, unknown> | null }) {
  const snapshots = Array.isArray(payload?.boundarySnapshots)
    ? payload.boundarySnapshots.flatMap((snapshot) => {
        if (!snapshot || typeof snapshot !== "object") return [];
        const id = "id" in snapshot && typeof snapshot.id === "string" ? snapshot.id : "snapshot";
        const label = "label" in snapshot && typeof snapshot.label === "string" ? snapshot.label : id;
        const matrix = "matrix" in snapshot && Array.isArray(snapshot.matrix) ? (snapshot.matrix as number[][]) : [];
        return [{ id, label, matrix }];
      })
    : [];
  const regions = Array.isArray(payload?.regions)
    ? payload.regions.flatMap((entry) => {
        if (!entry || typeof entry !== "object") return [];
        const label = "label" in entry && typeof entry.label === "string" ? entry.label : null;
        const value = "value" in entry && typeof entry.value === "number" ? entry.value : null;
        return label && value !== null ? [{ label, value }] : [];
      })
    : [];
  const [selectedSnapshotIndex, setSelectedSnapshotIndex] = useState(0);
  const selectedSnapshot = snapshots[Math.min(selectedSnapshotIndex, Math.max(snapshots.length - 1, 0))] ?? null;

  return (
    <section className="panel-section">
      <header className="panel-section__header">
        <span>Decision Boundary</span>
        <strong>{selectedSnapshot?.label ?? "No snapshot"}</strong>
      </header>
      <div className="focus-group__chips">
        {snapshots.map((snapshot, index) => (
          <button
            key={snapshot.id}
            type="button"
            className={selectedSnapshotIndex === index ? "focus-chip is-active" : "focus-chip"}
            onClick={() => setSelectedSnapshotIndex(index)}
          >
            {snapshot.label}
          </button>
        ))}
      </div>
      {selectedSnapshot ? (
        <div className="family-slice">
          <div className="family-slice__meta">
            <span>Boundary slice</span>
            <strong>{selectedSnapshot.label}</strong>
          </div>
          <MatrixHeatmap matrix={selectedSnapshot.matrix} />
        </div>
      ) : null}
      {regions.length > 0 ? (
        <div className="token-bars">
          {regions.map((region) => (
            <div key={region.label} className="token-bars__item">
              <div className="token-bars__meta">
                <span>{region.label}</span>
                <strong>{formatMetric(region.value)}</strong>
              </div>
              <div className="series-bar__track">
                <div className="series-bar__fill" style={{ width: `${Math.max(6, region.value * 100)}%` }} />
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function CnnFeaturePanel({
  payload,
  onSelect
}: {
  payload: Record<string, unknown> | null;
  onSelect(selection: SelectionState): void;
}) {
  const stages = Array.isArray(payload?.stages)
    ? payload.stages.flatMap((stage) => {
        if (!stage || typeof stage !== "object") return [];
        const id = "id" in stage && typeof stage.id === "string" ? stage.id : "stage";
        const label = "label" in stage && typeof stage.label === "string" ? stage.label : id;
        const matrix = "matrix" in stage && Array.isArray(stage.matrix) ? (stage.matrix as number[][]) : [];
        const channels =
          "channels" in stage && Array.isArray(stage.channels)
            ? stage.channels.flatMap((channel: any) => {
                if (!channel || typeof channel !== "object") return [];
                const channelId = "id" in channel && typeof channel.id === "string" ? channel.id : "channel";
                const channelLabel = "label" in channel && typeof channel.label === "string" ? channel.label : channelId;
                const nodeId = "nodeId" in channel && typeof channel.nodeId === "string" ? channel.nodeId : null;
                const channelMatrix = "matrix" in channel && Array.isArray(channel.matrix) ? (channel.matrix as number[][]) : [];
                const score = "score" in channel && typeof channel.score === "number" ? channel.score : null;
                return [{ id: channelId, label: channelLabel, nodeId, matrix: channelMatrix, score }];
              })
            : [];
        return [{ id, label, matrix, channels }];
      })
    : [];
  const topClasses = Array.isArray(payload?.topClasses)
    ? payload.topClasses.flatMap((entry) => {
        if (!entry || typeof entry !== "object") return [];
        const label = "label" in entry && typeof entry.label === "string" ? entry.label : null;
        const value = "value" in entry && typeof entry.value === "number" ? entry.value : null;
        return label && value !== null ? [{ label, value }] : [];
      })
    : [];
  const [selectedStageIndex, setSelectedStageIndex] = useState(0);
  const selectedStage = stages[Math.min(selectedStageIndex, Math.max(stages.length - 1, 0))] ?? null;
  const [selectedChannelIndex, setSelectedChannelIndex] = useState(0);
  const selectedChannel = selectedStage?.channels[Math.min(selectedChannelIndex, Math.max((selectedStage?.channels.length ?? 1) - 1, 0))] ?? null;

  useEffect(() => {
    setSelectedChannelIndex(0);
  }, [selectedStageIndex, payload?.headline]);

  return (
    <section className="panel-section">
      <header className="panel-section__header">
        <span>Feature Explorer</span>
        <strong>{selectedStage?.label ?? "No stage"}</strong>
      </header>
      <div className="focus-group">
        <span className="focus-group__label">Stages</span>
        <div className="focus-group__chips">
          {stages.map((stage, index) => (
            <button
              key={stage.id}
              type="button"
              className={selectedStageIndex === index ? "focus-chip is-active" : "focus-chip"}
              onClick={() => {
                setSelectedStageIndex(index);
                onSelect({ id: stage.id, kind: "node" });
              }}
            >
              {stage.label}
            </button>
          ))}
        </div>
      </div>
      {selectedStage ? (
        <div className="family-slice">
          <div className="family-slice__meta">
            <span>Stage map</span>
            <strong>{selectedStage.label}</strong>
          </div>
          <MatrixHeatmap matrix={selectedStage.matrix} />
        </div>
      ) : null}
      {selectedStage?.channels.length ? (
        <div className="focus-group">
          <span className="focus-group__label">Channels</span>
          <div className="focus-group__chips">
            {selectedStage.channels.map((channel: any, index: number) => (
              <button
                key={channel.id}
                type="button"
                className={selectedChannelIndex === index ? "focus-chip is-active" : "focus-chip"}
                onClick={() => {
                  setSelectedChannelIndex(index);
                  if (channel.nodeId) onSelect({ id: channel.nodeId, kind: "node" });
                }}
              >
                {channel.label}
              </button>
            ))}
          </div>
        </div>
      ) : null}
      {selectedChannel ? (
        <div className="family-slice">
          <div className="family-slice__meta">
            <span>Channel response</span>
            <strong>{selectedChannel.score !== null ? formatMetric(selectedChannel.score) : selectedChannel.label}</strong>
          </div>
          <MatrixHeatmap matrix={selectedChannel.matrix} />
        </div>
      ) : null}
      {topClasses.length > 0 ? (
        <div className="detail-card">
          <strong>Top classes</strong>
          <div className="detail-stats">
            {topClasses.map((entry) => (
              <div key={entry.label} className="detail-stat">
                <span>{entry.label}</span>
                <strong>{formatMetric(entry.value)}</strong>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function FamilyFocusPanel({
  bundle,
  selection,
  onSelect,
  payload,
  activeTrace
}: {
  bundle: TraceBundle;
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
  payload: Record<string, unknown> | null;
  activeTrace: (typeof officialTraces)[number] | null;
}) {
  const groups = getFamilyFocusGroups(bundle);
  const matrix = Array.isArray(payload?.matrix) ? (payload.matrix as number[][]) : null;
  const headline = typeof payload?.headline === "string" ? payload.headline : bundle.manifest.title;

  return (
    <section className="panel-section">
      <header className="panel-section__header">
        <span>Family Focus</span>
        <strong>{bundle.manifest.family}</strong>
      </header>
      <p className="muted-copy">{activeTrace?.studioTips[0] ?? "Use these shortcuts to jump between the most meaningful nodes for this model family."}</p>
      <div className="focus-groups">
        {groups.map((group) => (
          <div key={group.label} className="focus-group">
            <span className="focus-group__label">{group.label}</span>
            <div className="focus-group__chips">
              {group.items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={selection?.id === item.id ? "focus-chip is-active" : "focus-chip"}
                  onClick={() => onSelect({ id: item.id, kind: "node" })}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="family-slice">
        <div className="family-slice__meta">
          <span>{headline}</span>
          <strong>{bundle.manifest.family === "transformer" ? "attention slice" : bundle.manifest.family === "cnn" ? "feature slice" : "decision slice"}</strong>
        </div>
        {matrix ? <MatrixHeatmap matrix={matrix.slice(0, 5).map((row) => row.slice(0, 5))} /> : <p className="muted-copy">No matrix payload.</p>}
      </div>
    </section>
  );
}

function StoryPanel({
  bundle,
  frame,
  chapter,
  activeTraceTitle,
  watchFor
}: {
  bundle: TraceBundle;
  frame: TraceFrame;
  chapter: TraceBundle["narrative"]["chapters"][number] | null | undefined;
  activeTraceTitle: string | null;
  watchFor: readonly string[];
}) {
  return (
    <>
      <section className="panel-section">
        <header className="panel-section__header">
          <span>Story Focus</span>
          <strong>{chapter?.label ?? "Current frame"}</strong>
        </header>
        <p className="story-title">{activeTraceTitle ?? bundle.manifest.summary}</p>
        <p className="muted-copy">{chapter?.description ?? frame.note ?? "No chapter description available."}</p>
      </section>

      <section className="panel-section">
        <header className="panel-section__header">
          <span>Chapter Metrics</span>
          <strong>{frame.metric_refs.length} values</strong>
        </header>
        <div className="metric-grid">
          {frame.metric_refs.map((metric) => (
            <article key={metric.id} className="metric-card">
              <span>{metric.label}</span>
              <strong>{formatMetric(metric.value)}</strong>
            </article>
          ))}
        </div>
      </section>

      <section className="panel-section">
        <header className="panel-section__header">
          <span>What To Watch</span>
          <strong>{watchFor.length} cues</strong>
        </header>
        <KeyList items={watchFor} />
      </section>

      <section className="panel-section">
        <header className="panel-section__header">
          <span>Current Note</span>
          <strong>{frame.phase}</strong>
        </header>
        <div className="detail-card">
          <strong>{bundle.manifest.title}</strong>
          <p>{frame.note ?? "This frame has no additional note."}</p>
        </div>
      </section>
    </>
  );
}

function RenderLens({
  payload,
  family,
  mode
}: {
  payload: Record<string, unknown> | null;
  family: TraceBundle["manifest"]["family"];
  mode: "story" | "studio";
}) {
  const matrix = Array.isArray(payload?.matrix) ? (payload.matrix as number[][]) : null;
  const series = Array.isArray(payload?.series) ? (payload.series as Array<{ label: string; value: number }>) : null;
  const headline = typeof payload?.headline === "string" ? payload.headline : "Render lens";

  return (
    <div className="render-lens">
      <div className="render-lens__header">
        <span>{mode === "story" ? "Story Lens" : "Render Lens"}</span>
        <strong>{family}</strong>
      </div>
      <p className="render-lens__title">{headline}</p>
      {matrix ? <MatrixHeatmap matrix={matrix.slice(0, 6).map((row) => row.slice(0, 6))} /> : null}
      {series ? (
        <div className="render-lens__series">
          {series.slice(0, 3).map((item) => (
            <div key={item.label} className="render-lens__series-item">
              <span>{item.label}</span>
              <strong>{formatMetric(item.value)}</strong>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function LegendPill({ colorClass, label }: { colorClass: string; label: string }) {
  return (
    <span className={`legend-pill ${colorClass}`}>
      <i />
      {label}
    </span>
  );
}

function KeyList({ items }: { items: readonly string[] }) {
  return (
    <div className="key-list">
      {items.map((item) => (
        <article key={item} className="key-list__item">
          <span className="key-list__marker" />
          <p>{item}</p>
        </article>
      ))}
    </div>
  );
}

function getFamilyFocusGroups(bundle: TraceBundle) {
  const nodes = bundle.graph.nodes;
  const take = (ids: string[]) =>
    ids
      .map((id) => nodes.find((node) => node.id === id))
      .filter((node): node is TraceBundle["graph"]["nodes"][number] => Boolean(node))
      .map((node) => ({ id: node.id, label: node.label }));

  if (bundle.manifest.family === "mlp") {
    return [
      { label: "Inputs", items: take(["input-x", "input-y"]) },
      { label: "Hidden", items: take(["hidden-a", "hidden-b", "hidden-c", "mix-a", "mix-b"]) },
      { label: "Readout", items: take(["output", "loss"]) }
    ].filter((group) => group.items.length > 0);
  }

  if (bundle.manifest.family === "cnn") {
    return [
      { label: "Image + Stage", items: take(["image", "stage-1", "stage-2"]) },
      { label: "Feature Maps", items: take(["conv-1", "pool-1", "conv-2", "pool-2"]) },
      { label: "Head", items: take(["dense", "output", "loss"]) }
    ].filter((group) => group.items.length > 0);
  }

  return [
    { label: "Tokens", items: take(["token-bos", "token-neuro", "token-loom", "token-glows"]) },
    { label: "Block", items: take(["embed", "attn", "residual", "mlp", "norm"]) },
    { label: "Decode", items: take(["logits", "decode"]) }
  ].filter((group) => group.items.length > 0);
}

function PayloadView({ payload }: { payload: Record<string, unknown> }) {
  const matrix = Array.isArray(payload.matrix) ? (payload.matrix as number[][]) : null;
  const series = Array.isArray(payload.series) ? (payload.series as Array<{ label: string; value: number }>) : null;

  return (
    <div className="payload-view">
      {series ? (
        <div className="series-bars">
          {series.map((item) => (
            <div key={item.label} className="series-bar">
              <div className="series-bar__meta">
                <span>{item.label}</span>
                <strong>{formatMetric(item.value)}</strong>
              </div>
              <div className="series-bar__track">
                <div className="series-bar__fill" style={{ width: `${Math.max(6, Math.abs(item.value) * 100)}%` }} />
              </div>
            </div>
          ))}
        </div>
      ) : null}
      {matrix ? <MatrixHeatmap matrix={matrix} /> : null}
    </div>
  );
}

function MatrixHeatmap({ matrix }: { matrix: number[][] }) {
  const scale = scaleLinear<string>().domain([-1, 0, 1]).range(["#ffb45b", "#121b2b", "#15f0ff"]);

  return (
    <div
      className="matrix-heatmap"
      style={{
        gridTemplateColumns: `repeat(${matrix[0]?.length ?? 1}, minmax(0, 1fr))`
      }}
    >
      {matrix.flatMap((row, rowIndex) =>
        row.map((value, columnIndex) => (
          <span
            key={`${rowIndex}-${columnIndex}`}
            className="matrix-cell"
            title={String(value)}
            style={{ backgroundColor: scale(value) }}
          />
        ))
      )}
    </div>
  );
}

async function detectBrowserRuntime(): Promise<BrowserRuntimeState> {
  const webgl = canUseWebgl();
  const navigatorWithGpu = globalThis.navigator as Navigator & {
    gpu?: {
      requestAdapter(): Promise<unknown>;
    };
  };

  let webgpu = false;
  if (navigatorWithGpu.gpu) {
    try {
      webgpu = Boolean(await navigatorWithGpu.gpu.requestAdapter());
    } catch {
      webgpu = false;
    }
  }

  if (webgpu) {
    return {
      label: "Browser regen • WebGPU ready",
      detail: "Official traces can be rebuilt locally, and WebGPU is available for future browser-side official runtimes.",
      webgl,
      webgpu
    };
  }

  if (webgl) {
    return {
      label: "Browser regen • WebGL",
      detail: "Official traces can be rebuilt locally in this browser. Scene rendering stays on the stable WebGL path.",
      webgl,
      webgpu
    };
  }

  return {
    label: "Limited browser runtime",
    detail: "Replay still works, but browser-side regeneration and graphics acceleration are constrained in this environment.",
    webgl,
    webgpu
  };
}

function canUseWebgl() {
  if (typeof document === "undefined") return false;
  const canvas = document.createElement("canvas");
  return Boolean(canvas.getContext("webgl") || canvas.getContext("experimental-webgl"));
}

function formatSelectionLabel(bundle: TraceBundle, selection: SelectionState) {
  if (!selection) return null;
  if (selection.kind === "node") {
    return bundle.graph.nodes.find((node) => node.id === selection.id)?.label ?? selection.id;
  }
  return bundle.graph.edges.find((edge) => edge.id === selection.id)?.id ?? selection.id;
}

function parsePayload(raw: string | undefined) {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function formatMetric(value: number) {
  return value.toFixed(Math.abs(value) >= 1 ? 2 : 3);
}
