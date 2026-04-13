import { ReplayEngine, type TraceBundle } from "@neuroloom/core";
import { create } from "zustand";

export type AppMode = "story" | "studio";
export type SelectionState = {
  id: string;
  kind: "node" | "edge";
} | null;

export type RegenerationProgress = {
  frame: number;
  total: number;
  phase: string;
} | null;

type StudioState = {
  mode: AppMode;
  traceId: string | null;
  bundle: TraceBundle | null;
  engine: ReplayEngine | null;
  loadingLabel: string | null;
  error: string | null;
  frameIndex: number;
  playing: boolean;
  selection: SelectionState;
  frozenSelection: SelectionState;
  activeChapterId: string | null;
  regenerationProgress: RegenerationProgress;
  setMode(mode: AppMode): void;
  beginLoading(label: string): void;
  finishLoading(traceId: string, bundle: TraceBundle): void;
  failLoading(message: string): void;
  setFrameIndex(index: number): void;
  step(delta: number): void;
  togglePlaying(): void;
  setPlaying(value: boolean): void;
  setSelection(selection: SelectionState): void;
  toggleFreezeSelection(selection?: SelectionState): void;
  clearFrozenSelection(): void;
  jumpToChapter(chapterId: string): void;
  setRegenerationProgress(progress: RegenerationProgress): void;
};

export const useStudioStore = create<StudioState>((set, get) => ({
  mode: "story",
  traceId: null,
  bundle: null,
  engine: null,
  loadingLabel: null,
  error: null,
  frameIndex: 0,
  playing: false,
  selection: null,
  frozenSelection: null,
  activeChapterId: null,
  regenerationProgress: null,
  setMode(mode) {
    set({ mode });
  },
  beginLoading(label) {
    set({ loadingLabel: label, error: null, playing: false });
  },
  finishLoading(traceId, bundle) {
    const engine = new ReplayEngine(bundle);
    const firstChapter = bundle.narrative.chapters[0] ?? null;
    set({
      traceId,
      bundle,
      engine,
      loadingLabel: null,
      error: null,
      frameIndex: 0,
      playing: false,
      activeChapterId: firstChapter?.id ?? null,
      frozenSelection: null,
      regenerationProgress: null,
      selection: firstChapter?.defaultSelection ? { id: firstChapter.defaultSelection, kind: "node" } : null,
    });
  },
  failLoading(message) {
    set({ error: message, loadingLabel: null, playing: false, regenerationProgress: null });
  },
  setFrameIndex(index) {
    const engine = get().engine;
    if (!engine) return;
    const frameIndex = Math.max(0, Math.min(index, engine.frameCount - 1));
    const chapter = engine.getChapterForFrame(frameIndex);
    set({
      frameIndex,
      activeChapterId: chapter?.id ?? null,
    });
  },
  step(delta) {
    const state = get();
    if (!state.engine) return;
    state.setFrameIndex(state.frameIndex + delta);
  },
  togglePlaying() {
    set((state) => ({ playing: !state.playing }));
  },
  setPlaying(value) {
    set({ playing: value });
  },
  setSelection(selection) {
    set({ selection });
  },
  toggleFreezeSelection(selection) {
    const nextSelection = selection ?? get().selection;
    const frozenSelection = get().frozenSelection;
    if (!nextSelection) {
      set({ frozenSelection: null });
      return;
    }
    if (frozenSelection?.id === nextSelection.id && frozenSelection.kind === nextSelection.kind) {
      set({ frozenSelection: null });
      return;
    }
    set({ frozenSelection: nextSelection });
  },
  clearFrozenSelection() {
    set({ frozenSelection: null });
  },
  setRegenerationProgress(progress) {
    set({ regenerationProgress: progress });
  },
  jumpToChapter(chapterId) {
    const bundle = get().bundle;
    if (!bundle) return;
    const chapter = bundle.narrative.chapters.find((entry) => entry.id === chapterId);
    if (!chapter) return;
    set({
      frameIndex: chapter.frameRange[0],
      activeChapterId: chapter.id,
      frozenSelection: null,
      selection: chapter.defaultSelection ? { id: chapter.defaultSelection, kind: "node" } : null,
      playing: false,
    });
  },
}));
